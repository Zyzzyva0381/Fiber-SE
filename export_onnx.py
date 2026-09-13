from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from models import build_model


class StreamingModel(nn.Module):


    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, frame: Tensor, *states: Tensor) -> tuple[Tensor, ...]:
        return self.model.forward_frame(frame, *states)


def _checkpoint_state(payload: Any) -> dict[str, Tensor]:
    if not isinstance(payload, dict):
        return payload
    for key in ("model", "model_state_dict", "state_dict"):
        if key in payload:
            return payload[key]
    return payload


def export_onnx(
    *,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    model_entry: str | None = None,
    opset: int = 17,
    verify_frames: int = 8,
    strict: bool = True,
) -> dict[str, Any]:
    config = OmegaConf.load(config_path)
    model_name = model_entry or str(config.model.name)
    model = build_model(model_name).cpu().eval()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(_checkpoint_state(payload), strict=strict)

    prepare = getattr(model, "remove_weight_reparameterizations", None)
    if prepare is not None:
        prepare()
    model.float().eval()
    wrapper = StreamingModel(model).eval()

    frame = torch.zeros(1, model.n_fft // 2 + 1, 1, 2, dtype=torch.float32)
    states = model.initialize_state(1, device="cpu", dtype=torch.float32)
    input_names = ["frame", *(f"state_{index}" for index in range(len(states)))]
    output_names = ["enhanced", *(f"state_{index}_out" for index in range(len(states)))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (frame, *states),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)

    maximum_error = 0.0
    if verify_frames:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(output_path), options, providers=["CPUExecutionProvider"]
        )
        generator = np.random.default_rng(1)
        torch_states = states
        ort_states = [state.numpy() for state in states]
        with torch.inference_mode():
            for _ in range(verify_frames):
                frame_array = generator.standard_normal(frame.shape, dtype=np.float32) * 0.1
                expected = wrapper(torch.from_numpy(frame_array), *torch_states)
                feeds = {input_names[0]: frame_array}
                feeds.update(dict(zip(input_names[1:], ort_states)))
                actual = session.run(output_names, feeds)
                for expected_value, actual_value in zip(expected, actual):
                    error = np.max(np.abs(expected_value.numpy() - actual_value))
                    maximum_error = max(maximum_error, float(error))
                torch_states = tuple(expected[1:])
                ort_states = actual[1:]

    report = {
        "model": model_name,
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "onnx": str(output_path.resolve()),
        "opset": opset,
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": [list(frame.shape), *(list(state.shape) for state in states)],
        "verify_frames": verify_frames,
        "verify_max_abs_error": maximum_error,
        "missing_checkpoint_keys": list(incompatible.missing_keys),
        "unexpected_checkpoint_keys": list(incompatible.unexpected_keys),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model-entry", help="model module name; defaults to the config")
    result.add_argument("--opset", type=int, default=17)
    result.add_argument("--verify-frames", type=int, default=8)
    result.add_argument(
        "--non-strict", action="store_true", help="allow partial checkpoint loading"
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = export_onnx(
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        output_path=args.output.resolve(),
        model_entry=args.model_entry,
        opset=args.opset,
        verify_frames=args.verify_frames,
        strict=not args.non_strict,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
