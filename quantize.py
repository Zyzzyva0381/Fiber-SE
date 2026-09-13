from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def _array(initializers: dict[str, onnx.TensorProto], name: str) -> np.ndarray:
    if not name or name not in initializers:
        raise ValueError(f"required GRU initializer is absent: {name!r}")
    return np.asarray(numpy_helper.to_array(initializers[name]), dtype=np.float32)


def _initializer_roots(model: onnx.ModelProto) -> dict[str, str]:

    initializers = {item.name for item in model.graph.initializer}
    identity = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if not node.domain and node.op_type == "Identity" and len(node.input) == len(node.output) == 1
    }
    result: dict[str, str] = {}
    for name in set(initializers) | set(identity):
        current = name
        seen: set[str] = set()
        while current in identity:
            if current in seen:
                raise ValueError(f"initializer Identity cycle at {name}")
            seen.add(current)
            current = identity[current]
        if current in initializers:
            result[name] = current
    return result


def expand_grus(source: Path, output: Path) -> dict[str, Any]:
    model = onnx.load(str(source), load_external_data=True)
    initializers = {item.name: item for item in model.graph.initializer}
    roots = _initializer_roots(model)
    derived: dict[tuple[str, str], str] = {}
    added_initializers: list[onnx.TensorProto] = []
    replacement: list[onnx.NodeProto] = []
    quantize_nodes: list[str] = []
    rows: list[dict[str, Any]] = []

    axes0_name = "stream_w8a32_axes0"
    scalar_one_name = "stream_w8a32_one"
    added_initializers.extend(
        [
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), name=axes0_name),
            numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name=scalar_one_name),
        ]
    )

    def derived_initializer(root: str, kind: str, value: np.ndarray) -> str:
        key = (root, kind)
        if key not in derived:
            safe = root.replace("/", "_").replace(":", "_")
            name = f"stream_w8a32_{safe}_{kind}"
            derived[key] = name
            added_initializers.append(
                numpy_helper.from_array(np.ascontiguousarray(value), name=name)
            )
        return derived[key]

    for node in model.graph.node:
        if node.domain or node.op_type != "GRU":
            replacement.append(node)
            continue
        attributes = _attributes(node)
        hidden = int(attributes.get("hidden_size", 0))
        if attributes != {"hidden_size": hidden, "linear_before_reset": 1}:
            raise ValueError(f"unsupported GRU attributes for {node.name}: {attributes}")
        if len(node.input) < 6 or node.input[4]:
            raise ValueError(f"only unidirectional unmasked one-step GRU is supported: {node.name}")
        w_root = roots.get(node.input[1], node.input[1])
        r_root = roots.get(node.input[2], node.input[2])
        b_root = roots.get(node.input[3], node.input[3])
        weight = _array(initializers, w_root)
        recurrent = _array(initializers, r_root)
        bias = _array(initializers, b_root)
        if weight.shape != (1, 3 * hidden, hidden):
            raise ValueError(f"unexpected W shape for {node.name}: {weight.shape}")
        if recurrent.shape != weight.shape or bias.shape != (1, 6 * hidden):
            raise ValueError(
                f"unexpected R/B shape for {node.name}: {recurrent.shape}, {bias.shape}"
            )

        wt_name = derived_initializer(w_root, "wt", weight[0].T)
        rt_name = derived_initializer(r_root, "rt", recurrent[0].T)
        wb = bias[0, : 3 * hidden].reshape(3, hidden)
        rb = bias[0, 3 * hidden :].reshape(3, hidden)
        wb_names = [derived_initializer(b_root, f"wb{gate}", wb[index]) for index, gate in enumerate("zrh")]
        rb_names = [derived_initializer(b_root, f"rb{gate}", rb[index]) for index, gate in enumerate("zrh")]

        prefix = f"{node.name}.stream_w8a32"
        x2 = f"{prefix}.x2"
        h2 = f"{prefix}.h2"
        x_aff = f"{prefix}.x_aff"
        h_aff = f"{prefix}.h_aff"
        x_parts = [f"{prefix}.x_{gate}" for gate in "zrh"]
        h_parts = [f"{prefix}.h_{gate}" for gate in "zrh"]
        x_mm = f"{prefix}.XMatMul"
        h_mm = f"{prefix}.HMatMul"
        nodes = [
            helper.make_node("Squeeze", (node.input[0], axes0_name), (x2,), name=f"{prefix}.SqueezeX"),
            helper.make_node("Squeeze", (node.input[5], axes0_name), (h2,), name=f"{prefix}.SqueezeH"),
            helper.make_node("MatMul", (x2, wt_name), (x_aff,), name=x_mm),
            helper.make_node("MatMul", (h2, rt_name), (h_aff,), name=h_mm),
            helper.make_node("Split", (x_aff,), tuple(x_parts), name=f"{prefix}.SplitX", axis=1),
            helper.make_node("Split", (h_aff,), tuple(h_parts), name=f"{prefix}.SplitH", axis=1),
        ]
        quantize_nodes.extend((x_mm, h_mm))

        gate_values: dict[str, str] = {}
        for index, gate in enumerate("zr"):
            x_bias = f"{prefix}.{gate}_xb"
            h_bias = f"{prefix}.{gate}_hb"
            total = f"{prefix}.{gate}_pre"
            value = f"{prefix}.{gate}"
            nodes.extend(
                [
                    helper.make_node("Add", (x_parts[index], wb_names[index]), (x_bias,), name=f"{prefix}.{gate}AddXB"),
                    helper.make_node("Add", (h_parts[index], rb_names[index]), (h_bias,), name=f"{prefix}.{gate}AddHB"),
                    helper.make_node("Add", (x_bias, h_bias), (total,), name=f"{prefix}.{gate}Add"),
                    helper.make_node("Sigmoid", (total,), (value,), name=f"{prefix}.{gate}Sigmoid"),
                ]
            )
            gate_values[gate] = value

        xh_bias = f"{prefix}.h_xb"
        hh_bias = f"{prefix}.h_hb"
        reset_h = f"{prefix}.reset_h"
        candidate_pre = f"{prefix}.candidate_pre"
        candidate = f"{prefix}.candidate"
        one_minus_z = f"{prefix}.one_minus_z"
        old_part = f"{prefix}.old_part"
        new_part = f"{prefix}.new_part"
        new_h2 = f"{prefix}.new_h2"
        new_h = node.output[1]
        nodes.extend(
            [
                helper.make_node("Add", (x_parts[2], wb_names[2]), (xh_bias,), name=f"{prefix}.hAddXB"),
                helper.make_node("Add", (h_parts[2], rb_names[2]), (hh_bias,), name=f"{prefix}.hAddHB"),
                helper.make_node("Mul", (gate_values["r"], hh_bias), (reset_h,), name=f"{prefix}.ResetMul"),
                helper.make_node("Add", (xh_bias, reset_h), (candidate_pre,), name=f"{prefix}.CandidateAdd"),
                helper.make_node("Tanh", (candidate_pre,), (candidate,), name=f"{prefix}.CandidateTanh"),
                helper.make_node("Sub", (scalar_one_name, gate_values["z"]), (one_minus_z,), name=f"{prefix}.OneMinusZ"),
                helper.make_node("Mul", (gate_values["z"], h2), (old_part,), name=f"{prefix}.OldMul"),
                helper.make_node("Mul", (one_minus_z, candidate), (new_part,), name=f"{prefix}.NewMul"),
                helper.make_node("Add", (old_part, new_part), (new_h2,), name=f"{prefix}.StateAdd"),
                helper.make_node("Unsqueeze", (new_h2, axes0_name), (new_h,), name=f"{prefix}.UnsqueezeState"),
                helper.make_node("Unsqueeze", (new_h, axes0_name), (node.output[0],), name=f"{prefix}.UnsqueezeSequence"),
            ]
        )
        replacement.extend(nodes)
        rows.append(
            {
                "source_node": node.name,
                "hidden": hidden,
                "weight_root": w_root,
                "recurrent_root": r_root,
                "bias_root": b_root,
                "generated_matmuls": [x_mm, h_mm],
            }
        )

    projection_nodes = [
        node.name
        for node in replacement
        if not node.domain
        and node.op_type == "MatMul"
        and (
            "/temporal_projection/MatMul" in node.name
            or "/rnn_fc/MatMul" in node.name
        )
    ]
    quantize_nodes.extend(projection_nodes)
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    model.graph.initializer.extend(added_initializers)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output))
    return {
        "source": str(source.resolve()),
        "expanded": str(output.resolve()),
        "grus": rows,
        "shared_derived_initializers": len(derived),
        "projection_matmuls": projection_nodes,
        "nodes_to_quantize": quantize_nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expanded", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    expanded = args.expanded or args.output.with_name(f"{args.output.stem}.expanded.onnx")
    report = expand_grus(args.source, expanded)
    quantize_dynamic(
        expanded,
        args.output,
        op_types_to_quantize=["MatMul"],
        nodes_to_quantize=report["nodes_to_quantize"],
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": TensorProto.FLOAT},
    )
    quantized = onnx.load(str(args.output), load_external_data=True)
    onnx.checker.check_model(quantized)
    report.update(
        {
            "output": str(args.output.resolve()),
            "policy": "same per-output-channel W8 weights and dynamic per-row A8 for every GRU affine and temporal projection MatMul",
            "quantized_operator_counts": {
                op: sum(node.op_type == op for node in quantized.graph.node)
                for op in ("DynamicQuantizeLinear", "MatMulInteger", "Mul", "Cast")
            },
        }
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
