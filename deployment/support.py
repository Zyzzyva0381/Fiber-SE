from __future__ import annotations

import os
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ORT_VERSION = "1.19.2"
ORT_HEADER_NAMES = (
    "onnxruntime_c_api.h",
    "onnxruntime_cxx_api.h",
    "onnxruntime_cxx_inline.h",
    "onnxruntime_float16.h",
    "onnxruntime_lite_custom_op.h",
)
SIDECAR_MAGIC = b"FIBERPW1"


def write_initializer_sidecar(model: Path, output: Path) -> dict[str, Any]:

    import numpy as np
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(model), load_external_data=True)
    rows: list[tuple[str, int, tuple[int, ...], bytes]] = []
    for initializer in graph.graph.initializer:
        array = np.ascontiguousarray(numpy_helper.to_array(initializer))
        if array.dtype.hasobject:
            raise TypeError(f"string/object initializer is unsupported: {initializer.name}")
        rows.append(
            (
                initializer.name,
                int(initializer.data_type),
                tuple(int(value) for value in array.shape),
                array.tobytes(order="C"),
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(SIDECAR_MAGIC)
        stream.write(struct.pack("<I", len(rows)))
        for name, element_type, shape, data in rows:
            encoded = name.encode("utf-8")
            stream.write(struct.pack("<IIIQ", len(encoded), element_type, len(shape), len(data)))
            stream.write(encoded)
            stream.write(struct.pack(f"<{len(shape)}q", *shape))
            stream.write(data)

    return {
        "path": str(output.resolve()),
        "format": "fiber-deployment-initializers-v1",
        "initializer_count": len(rows),
        "tensor_bytes": sum(len(row[3]) for row in rows),
        "file_bytes": output.stat().st_size,
    }


def read_initializer_sidecar(path: Path) -> dict[str, Any]:

    with path.open("rb") as stream:
        if stream.read(len(SIDECAR_MAGIC)) != SIDECAR_MAGIC:
            raise ValueError("invalid deployment-initializer sidecar magic")
        (count,) = struct.unpack("<I", stream.read(4))
        names = []
        tensor_bytes = 0
        for _ in range(count):
            name_size, _element_type, rank, data_size = struct.unpack("<IIIQ", stream.read(20))
            names.append(stream.read(name_size).decode("utf-8"))
            stream.seek(rank * 8 + data_size, os.SEEK_CUR)
            tensor_bytes += data_size
        if stream.read(1):
            raise ValueError("trailing bytes in deployment-initializer sidecar")
    return {"initializer_count": count, "tensor_bytes": tensor_bytes, "names": names}


def ensure_ort_headers(cache_root: Path | None = None) -> Path:
    if cache_root is None:
        cache_root = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        ) / "fiber-se" / "onnxruntime" / ORT_VERSION
    cache_root.mkdir(parents=True, exist_ok=True)
    base_url = (
        "https://raw.githubusercontent.com/microsoft/onnxruntime/"
        f"v{ORT_VERSION}/include/onnxruntime/core/session"
    )
    for name in ORT_HEADER_NAMES:
        destination = cache_root / name
        if destination.is_file():
            continue
        with urllib.request.urlopen(f"{base_url}/{name}", timeout=30) as response:
            payload = response.read()
        destination.write_bytes(payload)
    return cache_root


def ensure_ort_header(cache_root: Path | None = None) -> Path:
    return ensure_ort_headers(cache_root) / "onnxruntime_c_api.h"


def ort_library(python: str = sys.executable) -> Path:
    code = (
        "import pathlib, onnxruntime; "
        "root=pathlib.Path(onnxruntime.__file__).parent/'capi'; "
        "print(next(root.glob('libonnxruntime.so.*')))"
    )
    value = subprocess.check_output([python, "-c", code], text=True).strip()
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def compile_runner(
    *, source: Path, header: Path, output: Path, library: Path, cxx: str = "g++"
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime_link = output.parent / "libonnxruntime.so.1"
    if not runtime_link.exists():
        runtime_link.symlink_to(library)
    subprocess.run(
        [
            cxx,
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-pedantic",
            str(source),
            f"-I{header.parent}",
            str(library),
            "-Wl,-rpath,$ORIGIN",
            "-o",
            str(output),
        ],
        check=True,
    )
