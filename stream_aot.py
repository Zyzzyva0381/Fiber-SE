from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


MICROSOFT_DOMAIN = "com.microsoft"
STREAM_DOMAIN = "com.setrain.cpu"


def _qkv_grouped(weight: np.ndarray, heads: int) -> np.ndarray:

    if weight.ndim != 2 or weight.shape[1] != 3 * weight.shape[0]:
        raise ValueError(f"expected square QKV projection, got {weight.shape}")
    channels = int(weight.shape[0])
    if channels % heads:
        raise ValueError(f"channels {channels} must be divisible by heads {heads}")
    head_channels = channels // heads
    return np.ascontiguousarray(
        weight.reshape(channels, heads, 3, head_channels)
        .transpose(0, 2, 1, 3)
        .reshape(channels, 3 * channels),
        dtype=np.float32,
    )


def _remove_dead_constants_and_initializers(model: onnx.ModelProto) -> None:

    while True:
        consumed = {name for node in model.graph.node for name in node.input if name}
        outputs = {value.name for value in model.graph.output}
        kept = [
            node
            for node in model.graph.node
            if not (node.op_type == "Constant" and not any(name in consumed | outputs for name in node.output))
        ]
        if len(kept) == len(model.graph.node):
            break
        del model.graph.node[:]
        model.graph.node.extend(kept)
    consumed = {name for node in model.graph.node for name in node.input if name}
    outputs = {value.name for value in model.graph.output}
    kept_initializers = [
        value for value in model.graph.initializer if value.name in consumed or value.name in outputs
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)


def _attention_prefixes(model: onnx.ModelProto) -> list[str]:
    suffix = "qkv/MatMul"
    prefixes = {
        node.name[: -len(suffix)]
        for node in model.graph.node
        if not node.domain and node.op_type == "MatMul" and node.name.endswith(suffix)
    }
    return sorted(prefixes)


def fuse_scaled_dot_product_attention(
    model: onnx.ModelProto,
    *,
    heads: int = 4,
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    for prefix in _attention_prefixes(model):
        nodes = list(model.graph.node)
        targets = [node for node in nodes if node.name.startswith(prefix)]
        by_name = {node.name: node for node in targets}
        qkv = by_name.get(f"{prefix}qkv/MatMul")
        tail = by_name.get(f"{prefix}Reshape_1")
        if qkv is None or tail is None:
            raise ValueError(f"incomplete exported attention region: {prefix}")

        expected_ops = {
            "MatMul",
            "Constant",
            "Reshape",
            "Transpose",
            "Split",
            "Shape",
            "Slice",
            "Cast",
            "Sqrt",
            "Div",
            "Mul",
            "Softmax",
        }
        unexpected = [
            (node.name, node.domain, node.op_type)
            for node in targets
            if node.domain or node.op_type not in expected_ops
        ]
        if unexpected:
            raise ValueError(f"unsupported attention operators under {prefix}: {unexpected}")

        target_outputs = {name for node in targets for name in node.output}
        external_uses = {
            name
            for node in nodes
            if node not in targets
            for name in node.input
            if name in target_outputs
        }
        if external_uses != {tail.output[0]}:
            raise ValueError(
                f"attention region {prefix} has unexpected external values: {sorted(external_uses)}"
            )

        initializers = {value.name: value for value in model.graph.initializer}
        if len(qkv.input) != 2 or qkv.input[1] not in initializers:
            raise ValueError(f"attention QKV weight is not static: {prefix}")
        source_weight = np.asarray(
            numpy_helper.to_array(initializers[qkv.input[1]]), dtype=np.float32
        )
        weight = _qkv_grouped(source_weight, heads)
        channels = int(weight.shape[0])
        weight_name = f"{prefix}stream_aot_qkv_weight"
        bias_name = f"{prefix}stream_aot_qkv_bias"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(weight, name=weight_name),
                numpy_helper.from_array(
                    np.zeros(3 * channels, dtype=np.float32), name=bias_name
                ),
            ]
        )
        fused = helper.make_node(
            "Attention",
            (qkv.input[0], weight_name, bias_name),
            tuple(tail.output),
            name=f"{prefix}StreamingAttention",
            domain=MICROSOFT_DOMAIN,
            num_heads=heads,
            unidirectional=0,
        )
        first_index = min(nodes.index(node) for node in targets)
        replacement = [node for node in nodes if node not in targets]
        replacement.insert(first_index, fused)
        del model.graph.node[:]
        model.graph.node.extend(replacement)
        rows.append(
            {
                "prefix": prefix,
                "input": qkv.input[0],
                "output": tail.output[0],
                "channels": channels,
                "heads": heads,
                "nodes_replaced": len(targets),
            }
        )
    if rows and not any(item.domain == MICROSOFT_DOMAIN for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid(MICROSOFT_DOMAIN, 1))
    return rows


def _constant_array(node: onnx.NodeProto) -> np.ndarray:
    if node.op_type != "Constant" or len(node.attribute) != 1:
        raise ValueError(f"expected one-tensor Constant, got {node.name}")
    return np.asarray(numpy_helper.to_array(node.attribute[0].t))


def _static_shape_from_reshape(
    producer: dict[str, onnx.NodeProto], node: onnx.NodeProto
) -> tuple[int, ...]:
    if node.op_type != "Reshape" or len(node.input) != 2:
        raise ValueError(f"expected static Reshape, got {node.name}")
    shape_node = producer.get(node.input[1])
    if shape_node is None:
        raise ValueError(f"reshape shape is not a Constant: {node.name}")
    shape = tuple(int(value) for value in _constant_array(shape_node).reshape(-1))
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"reshape is not fully static: {node.name} {shape}")
    return shape


def _ancestors(
    value: str,
    producer: dict[str, onnx.NodeProto],
    *,
    prefix: str,
    stop_values: set[str] | None = None,
) -> list[onnx.NodeProto]:
    result: list[onnx.NodeProto] = []
    seen: set[str] = set()
    pending = [value]
    stops = stop_values or set()
    while pending:
        current = pending.pop()
        if current in stops:
            continue
        node = producer.get(current)
        if node is None or not node.name.startswith(prefix) or node.name in seen:
            continue
        seen.add(node.name)
        result.append(node)
        pending.extend(name for name in node.input if name)
    return result


def _slice_start(node: onnx.NodeProto, producer: dict[str, onnx.NodeProto]) -> int:
    if node.op_type != "Slice" or len(node.input) < 2:
        raise ValueError(f"expected Slice, got {node.name}")
    starts = producer.get(node.input[1])
    if starts is None:
        raise ValueError(f"dynamic Slice start: {node.name}")
    values = _constant_array(starts).reshape(-1)
    if values.size != 1:
        raise ValueError(f"non-scalar Slice start: {node.name}")
    return int(values[0])


def fuse_static_grouped_recurrent_layout(model: onnx.ModelProto) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    recurrent_nodes = [
        node
        for node in model.graph.node
        if not node.domain and node.op_type == "GRU" and node.name.endswith("/temporal/GRU")
    ]
    for index, gru in enumerate(recurrent_nodes):
        prefix = gru.name[: -len("temporal/GRU")]
        nodes = list(model.graph.node)
        producer = {name: node for node in nodes for name in node.output}
        pack_reshape = producer.get(gru.input[0])
        if pack_reshape is None or pack_reshape.op_type != "Reshape":
            raise ValueError(f"missing canonical grouped Reshape before {gru.name}")
        pack_nodes = _ancestors(pack_reshape.output[0], producer, prefix=prefix)
        pack_outputs = {name for node in pack_nodes for name in node.output}
        pack_external = {
            name
            for node in pack_nodes
            for name in node.input
            if name and name not in pack_outputs
        }


        if len(pack_external) != 1:
            raise ValueError(f"ambiguous grouped-layout input for {gru.name}: {pack_external}")
        field = next(iter(pack_external))

        pack_shape = _static_shape_from_reshape(producer, pack_reshape)
        if len(pack_shape) != 3:
            raise ValueError(f"unexpected packed shape for {gru.name}: {pack_shape}")
        hidden = int(pack_shape[-1])
        field_reshape = next(
            (
                node
                for node in nodes
                if node.name == f"{prefix}Reshape_2" and node.op_type == "Reshape"
            ),
            None,
        )
        if field_reshape is None:
            raise ValueError(f"missing canonical field Reshape for {gru.name}")
        field_shape = _static_shape_from_reshape(producer, field_reshape)
        channels = int(field_shape[-1])
        if hidden % channels:
            raise ValueError(f"hidden/channels mismatch for {gru.name}: {hidden}/{channels}")
        group = hidden // channels

        pack_concat = next((node for node in pack_nodes if node.op_type == "Concat"), None)
        if pack_concat is None:
            offset = 0
        elif pack_concat.input and pack_concat.input[0] == field:
            offset = 0
        else:
            first_slice = producer.get(pack_concat.input[0])
            if first_slice is None or first_slice.op_type != "Slice":
                raise ValueError(f"cannot derive static cyclic offset for {gru.name}")
            offset = _slice_start(first_slice, producer)
        if not 0 <= offset < group:
            raise ValueError(f"offset {offset} outside group {group} for {gru.name}")

        projection = next(
            (
                node
                for node in nodes
                if node.name == f"{prefix}temporal_projection/MatMul"
                and node.op_type == "MatMul"
            ),
            None,
        )
        projection_add = next(
            (
                node
                for node in nodes
                if node.name == f"{prefix}temporal_projection/Add" and node.op_type == "Add"
            ),
            None,
        )
        residual_add = next(
            (node for node in nodes if node.name == f"{prefix}Add" and node.op_type == "Add"),
            None,
        )
        if projection is None or projection_add is None or residual_add is None:
            raise ValueError(f"missing canonical projection/residual chain for {gru.name}")
        bias_names = [name for name in projection_add.input if name != projection.output[0]]
        if len(bias_names) != 1:
            raise ValueError(f"ambiguous projection bias for {gru.name}")
        bias = bias_names[0]
        unpack_value = next(name for name in residual_add.input if name != field)
        unpack_nodes = _ancestors(
            unpack_value,
            producer,
            prefix=prefix,
            stop_values={projection.output[0]},
        )
        if residual_add not in unpack_nodes:
            unpack_nodes.append(residual_add)
        unpack_nodes = [node for node in unpack_nodes if node is not projection]

        group_name = f"stream_aot.group.{index}"
        offset_name = f"stream_aot.offset.{index}"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray(group, dtype=np.int64), name=group_name),
                numpy_helper.from_array(np.asarray(offset, dtype=np.int64), name=offset_name),
            ]
        )
        pack = helper.make_node(
            "PackFibre",
            (field, group_name, offset_name),
            tuple(pack_reshape.output),
            name=f"{prefix}StreamingGroupedPack",
            domain=STREAM_DOMAIN,
        )
        unpack = helper.make_node(
            "UnpackFibreBiasAdd",
            (field, projection.output[0], bias, group_name, offset_name),
            tuple(residual_add.output),
            name=f"{prefix}StreamingGroupedUnpack",
            domain=STREAM_DOMAIN,
        )
        removed_names = {node.name for node in pack_nodes + unpack_nodes}
        replacement: list[onnx.NodeProto] = []
        for node in nodes:
            if node.name in removed_names:
                if node is pack_reshape:
                    replacement.append(pack)
                elif node is residual_add:
                    replacement.append(unpack)
                continue
            replacement.append(node)
        del model.graph.node[:]
        model.graph.node.extend(replacement)
        rows.append(
            {
                "prefix": prefix,
                "field": field,
                "group": group,
                "offset": offset,
                "packed_shape": list(pack_shape),
                "field_shape": list(field_shape),
                "nodes_replaced": len(removed_names),
            }
        )
    if rows and not any(item.domain == STREAM_DOMAIN for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid(STREAM_DOMAIN, 1))
    return rows


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in dict.fromkeys(node.input):
            if name:
                result.setdefault(name, []).append(node)
    return result


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {name: node for node in model.graph.node for name in node.output}


def _single_consumer_of_type(
    consumers: dict[str, list[onnx.NodeProto]],
    value: str,
    operator: str,
) -> onnx.NodeProto:
    matches = [
        node
        for node in consumers.get(value, [])
        if not node.domain and node.op_type == operator
    ]
    return _only(matches, f"{operator} consumer of {value}")


def _ancestor_nodes(
    value: str,
    producer: dict[str, onnx.NodeProto],
    *,
    stop_values: set[str],
) -> set[str]:
    names: set[str] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if not current or current in stop_values:
            continue
        node = producer.get(current)
        if node is None or node.name in names:
            continue
        names.add(node.name)
        pending.extend(node.input)
    return names


def _descendant_values(
    value: str,
    consumers: dict[str, list[onnx.NodeProto]],
) -> set[str]:
    values = {value}
    pending = [value]
    seen_nodes: set[str] = set()
    while pending:
        current = pending.pop()
        for node in consumers.get(current, []):
            if node.name in seen_nodes:
                continue
            seen_nodes.add(node.name)
            for output in node.output:
                if output not in values:
                    values.add(output)
                    pending.append(output)
    return values


def _retain_shared_ancestors(
    removed: set[str],
    nodes: list[onnx.NodeProto],
    consumers: dict[str, list[onnx.NodeProto]],
    *,
    replacement_values: set[str],
) -> set[str]:

    retained = set(removed)
    by_name = {node.name: node for node in nodes}
    while True:
        shared = {
            name
            for name in retained
            if any(
                output not in replacement_values
                and any(consumer.name not in retained for consumer in consumers.get(output, []))
                for output in by_name[name].output
            )
        }
        if not shared:
            return retained
        retained.difference_update(shared)


def _replace_initializer(
    model: onnx.ModelProto,
    name: str,
    array: np.ndarray,
) -> None:
    index = next(
        (index for index, value in enumerate(model.graph.initializer) if value.name == name),
        None,
    )
    if index is None:
        raise ValueError(f"initializer does not exist: {name}")
    del model.graph.initializer[index]
    model.graph.initializer.insert(
        index,
        numpy_helper.from_array(np.ascontiguousarray(array, dtype=np.float32), name=name),
    )


def _fold_bidirectional_gram_weight(weight: np.ndarray) -> np.ndarray:

    source = np.asarray(weight, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (5, 8):
        raise ValueError(f"expected a [C,5,8] Gram analysis kernel, got {source.shape}")
    folded = np.zeros((source.shape[0], 3, 9), dtype=np.float32)
    folded[:, 0, 1:] = source[:, 0]
    folded[:, 1, :8] = source[:, 1]
    folded[:, 2, :8] = -source[:, 2]
    folded[:, 1, 1:] += source[:, 3]
    folded[:, 2, 1:] += source[:, 4]
    return folded


def _set_conv_kernel_and_padding(
    node: onnx.NodeProto,
    *,
    kernel: int,
    padding: int,
) -> None:
    attributes = {item.name: item for item in node.attribute}
    if "kernel_shape" not in attributes or "pads" not in attributes:
        raise ValueError(f"analysis Conv has no explicit kernel/padding: {node.name}")
    attributes["kernel_shape"].ints[:] = [kernel]
    attributes["pads"].ints[:] = [padding, padding]


def _find_compressed_spectrum(
    model: onnx.ModelProto,
) -> tuple[onnx.NodeProto, onnx.NodeProto]:

    consumers = _consumer_map(model)
    graph_inputs = {value.name for value in model.graph.input}
    body_candidates = [
        node
        for node in model.graph.node
        if not node.domain
        and node.op_type == "Slice"
        and node.input
        and node.input[0] in graph_inputs
    ]
    body = _only(body_candidates, "complex-body Slice")
    square = _only(
        [
            node
            for node in consumers.get(body.output[0], [])
            if not node.domain
            and node.op_type == "Mul"
            and list(node.input) == [body.output[0], body.output[0]]
        ],
        "complex-power square",
    )
    reduce = _single_consumer_of_type(consumers, square.output[0], "ReduceSum")
    clip = _single_consumer_of_type(consumers, reduce.output[0], "Clip")
    root = _single_consumer_of_type(consumers, clip.output[0], "Sqrt")
    exponent = _single_consumer_of_type(consumers, root.output[0], "Pow")
    compressed = _only(
        [
            node
            for node in consumers.get(body.output[0], [])
            if not node.domain
            and node.op_type == "Mul"
            and set(node.input) == {body.output[0], exponent.output[0]}
        ],
        "compressed-complex Mul",
    )
    return body, compressed


def lower_complex_streaming_pipeline(model: onnx.ModelProto) -> dict[str, Any]:

    nodes = list(model.graph.node)
    producer = _producer_map(model)
    consumers = _consumer_map(model)
    initializers = {
        value.name: np.asarray(numpy_helper.to_array(value))
        for value in model.graph.initializer
    }
    body, compressed_node = _find_compressed_spectrum(model)
    compressed = compressed_node.output[0]
    reachable = _descendant_values(compressed, consumers)
    learned_convolutions = [
        node
        for node in nodes
        if not node.domain
        and node.op_type == "Conv"
        and len(node.input) >= 2
        and node.input[0] in reachable
        and node.input[1] in initializers
        and initializers[node.input[1]].ndim == 3
    ]
    analysis = _only(learned_convolutions[:1], "first learned analysis Conv")
    analysis_weight = initializers[analysis.input[1]]

    gram_folded = analysis_weight.shape[1:] == (5, 8)
    magnitude_observation = analysis_weight.shape[1:] == (3, 8)
    if gram_folded:
        attributes = {
            item.name: helper.get_attribute_value(item) for item in analysis.attribute
        }
        if (
            attributes.get("kernel_shape") != [8]
            or attributes.get("pads") != [2, 2]
            or attributes.get("strides") != [4]
        ):
            raise ValueError(f"unsupported Gram analysis geometry: {attributes}")
        _replace_initializer(
            model,
            analysis.input[1],
            _fold_bidirectional_gram_weight(analysis_weight),
        )
        _set_conv_kernel_and_padding(analysis, kernel=9, padding=3)
        feature = analysis.input[0]
        frontend_operator = "CompressComplexSpectrumAndOrientedGram"
    elif magnitude_observation:




        feature = analysis.input[0]
        frontend_operator = "CompressComplexSpectrumAndMagnitude"
    else:
        transposes = [
            node
            for node in consumers.get(compressed, [])
            if not node.domain and node.op_type == "Transpose"
        ]
        transpose = _only(transposes, "raw-complex feature Transpose")
        reshape = _single_consumer_of_type(consumers, transpose.output[0], "Reshape")
        feature = reshape.output[0]
        frontend_operator = "CompressComplexSpectrum"

    frontend_removed = _ancestor_nodes(
        feature,
        producer,
        stop_values={model.graph.input[0].name},
    )
    frontend_removed = _retain_shared_ancestors(
        frontend_removed,
        nodes,
        consumers,
        replacement_values={feature, compressed},
    )
    if compressed_node.name not in frontend_removed:
        raise ValueError("frontend target does not contain the compression chain")
    removed_outputs = {
        output
        for node in nodes
        if node.name in frontend_removed
        for output in node.output
    }
    external_frontend_values = {
        name
        for node in nodes
        if node.name not in frontend_removed
        for name in node.input
        if name in removed_outputs
    }
    if external_frontend_values - {feature, compressed}:
        raise ValueError(
            "complex frontend has unexpected external values: "
            f"{sorted(external_frontend_values)}"
        )
    frontend = helper.make_node(
        frontend_operator,
        (model.graph.input[0].name,),
        (feature, compressed),
        name=f"stream_aot.{frontend_operator}",
        domain=STREAM_DOMAIN,
    )

    graph_outputs = {value.name for value in model.graph.output}
    spectrum_outputs = [name for name in graph_outputs if name not in {n for n in graph_outputs if n.startswith("state_")}]
    enhanced = _only(
        [
            node
            for node in nodes
            if any(output in spectrum_outputs for output in node.output)
        ],
        "enhanced-spectrum producer",
    ).output[0]
    enhanced_ancestors = _ancestor_nodes(enhanced, producer, stop_values={compressed})
    projections = [
        node
        for node in nodes
        if node.name in enhanced_ancestors and not node.domain and node.op_type == "ConvTranspose"
    ]
    projection = _only(projections, "complex-mask ConvTranspose")
    mask = projection.output[0]
    backend_removed = _ancestor_nodes(
        enhanced,
        producer,
        stop_values={compressed, mask},
    )
    backend = helper.make_node(
        "ApplyMaskAndDecompress",
        (compressed, mask),
        (enhanced,),
        name="stream_aot.ApplyMaskAndDecompress",
        domain=STREAM_DOMAIN,
    )

    frontend_index = min(index for index, node in enumerate(nodes) if node.name in frontend_removed)
    projection_index = nodes.index(projection)
    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        if index == frontend_index:
            rewritten.append(frontend)
        if node.name in frontend_removed or node.name in backend_removed:
            continue
        rewritten.append(node)
        if index == projection_index:
            rewritten.append(backend)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    if not any(item.domain == STREAM_DOMAIN for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid(STREAM_DOMAIN, 1))
    return {
        "frontend_operator": frontend_operator,
        "backend_operator": "ApplyMaskAndDecompress",
        "gram_folded": gram_folded,
        "magnitude_observation": magnitude_observation,
        "analysis_node": analysis.name,
        "analysis_weight": analysis.input[1],
        "analysis_weight_before": list(analysis_weight.shape),
        "analysis_weight_after": [int(analysis_weight.shape[0]), 3, 9]
        if gram_folded
        else list(analysis_weight.shape),
        "frontend_nodes_replaced": len(frontend_removed),
        "backend_nodes_replaced": len(backend_removed),
        "compressed_value": compressed,
        "mask_value": mask,
    }


def _only(items: list[onnx.NodeProto], description: str) -> onnx.NodeProto:
    if len(items) != 1:
        raise ValueError(f"expected one {description}, found {len(items)}")
    return items[0]


def _initializer_root(
    name: str,
    initializers: set[str],
    producer: dict[str, onnx.NodeProto],
) -> str:
    seen: set[str] = set()
    while name not in initializers:
        node = producer.get(name)
        if node is None or node.domain or node.op_type != "Identity" or name in seen:
            raise ValueError(f"initializer alias {name!r} has no static root")
        seen.add(name)
        name = node.input[0]
    return name


def _tensor_shape(model: onnx.ModelProto, name: str) -> tuple[int, ...]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    value = next((item for item in values if item.name == name), None)
    if value is None:
        raise ValueError(f"missing static tensor shape: {name}")
    shape = tuple(int(item.dim_value) for item in value.type.tensor_type.shape.dim)
    if not shape or any(item <= 0 for item in shape):
        raise ValueError(f"dynamic tensor shape: {name} {shape}")
    return shape


def _fuse_grouped_temporal_blocks(model: onnx.ModelProto) -> list[dict[str, Any]]:

    initializers = {item.name: item for item in model.graph.initializer}
    producer = {name: node for node in model.graph.node for name in node.output}
    consumers = _consumer_map(model)
    gru_nodes = [
        node
        for node in model.graph.node
        if not node.domain
        and node.op_type == "GRU"
        and (producer.get(node.input[0]) is not None)
        and producer[node.input[0]].domain == STREAM_DOMAIN
        and producer[node.input[0]].op_type == "PackFibre"
    ]
    if not gru_nodes:
        return []
    position_template = next(
        (value for name, value in initializers.items() if name.endswith(".position")), None
    )
    if position_template is None:
        raise ValueError("grouped temporal stack has no position tensor template")
    zero_position = np.zeros_like(numpy_helper.to_array(position_template))

    replacements: dict[str, onnx.NodeProto] = {}
    removed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, gru in enumerate(gru_nodes):
        pack = producer.get(gru.input[0])
        if pack is None or pack.domain != STREAM_DOMAIN or pack.op_type != "PackFibre":
            continue
        attributes = {item.name: helper.get_attribute_value(item) for item in gru.attribute}
        hidden = int(attributes.get("hidden_size", 0))
        state_shape = _tensor_shape(model, gru.input[5])
        input_shape = state_shape
        if (
            attributes != {"hidden_size": hidden, "linear_before_reset": 1}
            or input_shape[0] != 1
            or input_shape[2] != hidden
            or state_shape != (1, input_shape[1], hidden)
            or gru.input[4]
        ):
            raise ValueError(f"unsupported grouped GRU contract: {gru.name}")
        weight = _initializer_root(gru.input[1], set(initializers), producer)
        recurrent = _initializer_root(gru.input[2], set(initializers), producer)
        bias = _initializer_root(gru.input[3], set(initializers), producer)
        squeeze = _only(consumers.get(gru.output[0], []), "grouped GRU Squeeze")
        projection = _only(consumers.get(squeeze.output[0], []), "grouped projection")
        unpack = _only(consumers.get(projection.output[0], []), "grouped unpack")
        if (
            squeeze.domain
            or squeeze.op_type != "Squeeze"
            or projection.domain
            or projection.op_type != "MatMul"
            or unpack.domain != STREAM_DOMAIN
            or unpack.op_type != "UnpackFibreBiasAdd"
            or unpack.input[0] != pack.input[0]
            or unpack.input[3:] != pack.input[1:]
        ):
            raise ValueError(f"unexpected grouped temporal boundary: {gru.name}")
        position_add = next(
            (
                candidate
                for candidate in consumers.get(unpack.output[0], [])
                if not candidate.domain
                and candidate.op_type == "Add"
                and any(name.endswith(".position") for name in candidate.input)
            ),
            None,
        )
        if position_add is None:
            position_name = f"{gru.name}.stream_aot_zero_position"
            model.graph.initializer.append(
                numpy_helper.from_array(zero_position, name=position_name)
            )
            block_output = unpack.output[0]
        else:
            position_name = next(name for name in position_add.input if name.endswith(".position"))
            block_output = position_add.output[0]
            removed.add(position_add.name)
        block = helper.make_node(
            "FibreGRUBlock",
            (
                pack.input[0],
                unpack.input[0],
                weight,
                recurrent,
                bias,
                gru.input[5],
                projection.input[1],
                unpack.input[2],
                position_name,
                pack.input[1],
                pack.input[2],
            ),
            (block_output, gru.output[1]),
            name=f"{gru.name}.StreamingTemporalBlock",
            domain=STREAM_DOMAIN,
        )
        replacements[pack.name] = block
        removed.update((gru.name, squeeze.name, projection.name, unpack.name))
        rows.append(
            {
                "index": index,
                "layout": "static-cyclic-grouped",
                "source_gru": gru.name,
                "hidden": hidden,
                "batch": input_shape[1],
                "weight_root": weight,
                "recurrent_root": recurrent,
                "position_folded": position_add is not None,
                "operator": "FibreGRUBlock",
            }
        )
    nodes = []
    for node in model.graph.node:
        if node.name in removed:
            continue
        nodes.append(replacements.get(node.name, node))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return rows


def _fuse_direct_temporal_blocks(model: onnx.ModelProto) -> list[dict[str, Any]]:

    initializers = {item.name: item for item in model.graph.initializer}
    initializer_names = set(initializers)
    producer = {name: node for node in model.graph.node for name in node.output}
    consumers = _consumer_map(model)
    replacements: dict[str, onnx.NodeProto] = {}
    removed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, gru in enumerate(
        node for node in model.graph.node if not node.domain and node.op_type == "GRU"
    ):
        if producer.get(gru.input[0], None) is not None and producer[gru.input[0]].domain == STREAM_DOMAIN:
            continue
        reshape_in = producer.get(gru.input[0])
        squeeze = _only(consumers.get(gru.output[0], []), "direct GRU Squeeze")
        reshape_out = _only(consumers.get(squeeze.output[0], []), "direct post-GRU Reshape")
        projection = _only(consumers.get(reshape_out.output[0], []), "direct projection")
        projection_bias = _only(consumers.get(projection.output[0], []), "projection bias")
        residual_add = _only(consumers.get(projection_bias.output[0], []), "direct residual")
        if (
            reshape_in is None
            or reshape_in.domain
            or reshape_in.op_type != "Reshape"
            or squeeze.domain
            or squeeze.op_type != "Squeeze"
            or reshape_out.domain
            or reshape_out.op_type != "Reshape"
            or projection.domain
            or projection.op_type != "MatMul"
            or projection_bias.domain
            or projection_bias.op_type != "Add"
            or residual_add.domain
            or residual_add.op_type != "Add"
        ):
            raise ValueError(f"unexpected direct temporal boundary: {gru.name}")
        residual_names = [name for name in residual_add.input if name != projection_bias.output[0]]
        bias_names = [name for name in projection_bias.input if name != projection.output[0]]
        if len(residual_names) != 1 or len(bias_names) != 1:
            raise ValueError(f"ambiguous direct temporal residual/bias: {gru.name}")
        weight = _initializer_root(gru.input[1], initializer_names, producer)
        recurrent = _initializer_root(gru.input[2], initializer_names, producer)
        bias = _initializer_root(gru.input[3], initializer_names, producer)
        projection_weight = _initializer_root(
            projection.input[1], initializer_names, producer
        )
        projection_bias_root = _initializer_root(
            bias_names[0], initializer_names, producer
        )
        position_add = next(
            (
                candidate
                for candidate in consumers.get(residual_add.output[0], [])
                if not candidate.domain
                and candidate.op_type == "Add"
                and any(name.endswith(".pe") for name in candidate.input if name in initializers)
            ),
            None,
        )
        if position_add is None:
            state_shape = _tensor_shape(model, gru.input[5])
            position_name = f"{gru.name}.stream_aot_zero_position"
            model.graph.initializer.append(
                numpy_helper.from_array(np.zeros(state_shape, dtype=np.float32), name=position_name)
            )
            block_output = residual_add.output[0]
        else:
            position_name = next(name for name in position_add.input if name in initializers)
            block_output = position_add.output[0]
            removed.add(position_add.name)
        block = helper.make_node(
            "FastEnhancerGRUBlock",
            (
                gru.input[0],
                residual_names[0],
                weight,
                recurrent,
                bias,
                gru.input[5],
                projection_weight,
                projection_bias_root,
                position_name,
            ),
            (block_output, gru.output[1]),
            name=f"{gru.name}.StreamingTemporalBlock",
            domain=STREAM_DOMAIN,
        )
        replacements[gru.name] = block
        removed.update(
            (
                squeeze.name,
                reshape_out.name,
                projection.name,
                projection_bias.name,
                residual_add.name,
            )
        )
        attributes = {item.name: helper.get_attribute_value(item) for item in gru.attribute}
        state_shape = _tensor_shape(model, gru.input[5])
        rows.append(
            {
                "index": index,
                "layout": "direct",
                "source_gru": gru.name,
                "hidden": int(attributes.get("hidden_size", 0)),
                "batch": state_shape[1],
                "weight_root": weight,
                "recurrent_root": recurrent,
                "position_folded": position_add is not None,
                "operator": "FastEnhancerGRUBlock",
            }
        )
    nodes = []
    for node in model.graph.node:
        if node.name in removed:
            continue
        nodes.append(replacements.get(node.name, node))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return rows


def fuse_symmetric_temporal_blocks(model: onnx.ModelProto) -> list[dict[str, Any]]:
    grouped = _fuse_grouped_temporal_blocks(model)
    direct = _fuse_direct_temporal_blocks(model)
    rows = grouped + direct
    if rows and not any(item.domain == STREAM_DOMAIN for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid(STREAM_DOMAIN, 1))
    return rows


def lower(
    source: Path,
    output: Path,
    *,
    heads: int = 4,
    attention: bool = True,
    complex_pipeline: bool = False,
    temporal_blocks: bool = False,
) -> dict[str, Any]:
    model = onnx.load(str(source), load_external_data=True)
    nodes_before = len(model.graph.node)
    initializers_before = len(model.graph.initializer)
    complex_result = lower_complex_streaming_pipeline(model) if complex_pipeline else None
    attention_regions = fuse_scaled_dot_product_attention(model, heads=heads) if attention else []
    if attention and not attention_regions:
        raise ValueError("no canonical scaled-dot-product attention region found")
    grouped_layout = fuse_static_grouped_recurrent_layout(model)
    temporal = fuse_symmetric_temporal_blocks(model) if temporal_blocks else []
    _remove_dead_constants_and_initializers(model)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return {
        "schema": "fiber-se.streaming-aot-lowering.v2",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "rules": {
            "model_identity_available": False,
            "attention_pattern": "canonical static scaled-dot-product self-attention"
            if attention
            else None,
            "attention_heads": heads if attention else None,
            "complex_pattern": "canonical compressed-complex streaming frontend/backend"
            if complex_pipeline
            else None,
        },
        "complex_pipeline": complex_result,
        "attention_regions": attention_regions,
        "grouped_recurrent_regions": grouped_layout,
        "temporal_blocks": temporal,
        "node_count_before": nodes_before,
        "node_count_after": len(model.graph.node),
        "initializer_count_before": initializers_before,
        "initializer_count_after": len(model.graph.initializer),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--report", type=Path)
    result.add_argument("--heads", type=int, default=4)
    result.add_argument(
        "--no-attention",
        action="store_true",
        help="leave canonical scaled-dot-product attention expanded",
    )
    result.add_argument(
        "--complex-pipeline",
        action="store_true",
        help="fuse complex compression/decompression and exact Gram redundancy",
    )
    result.add_argument(
        "--temporal-blocks",
        action="store_true",
        help="fuse the symmetric GRU/projection/residual/position boundary",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    report = lower(
        source,
        output,
        heads=args.heads,
        attention=not args.no_attention,
        complex_pipeline=args.complex_pipeline,
        temporal_blocks=args.temporal_blocks,
    )
    report_path = args.report.resolve() if args.report else output.with_suffix(".aot.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
