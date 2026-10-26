"""An S-Pass on disk: a directory holding `spass.safetensors` (one float16 tensor `spass` of
shape `(tokens, width)` in the authorized model's native embedding coordinates),
`plain.safetensors` (the skill's own embeddings, the `plain` presentation) and `spass.json`,
the sidecar recording how it was made. The format is written and read in `numpy` alone."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPASS_FILE = "spass.safetensors"
PLAIN_FILE = "plain.safetensors"
SIDECAR_FILE = "spass.json"
TENSOR_NAME = "spass"
PLAIN_NAME = "plain"

REQUIRED_FIELDS = (
    "construction",
    "skill",
    "model",
    "revision",
    "tokens",
    "width",
    "seed",
    "decoy",
    "source_sha256",
)

_DTYPES = {"F16": np.float16, "F32": np.float32}
_NAMES = {np.dtype(np.float16): "F16", np.dtype(np.float32): "F32"}


def save_file(tensors: Mapping[str, np.ndarray], path: str | Path) -> None:
    header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        if array.dtype not in _NAMES:
            raise ValueError(f"a tensor is float16 or float32, got {array.dtype}")
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))
        raw = array.tobytes()
        header[name] = {
            "dtype": _NAMES[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<Q", len(text)))
        handle.write(text)
        for raw in chunks:
            handle.write(raw)


def load_file(path: str | Path) -> dict[str, np.ndarray]:
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path} is not a safetensors file")
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + length].decode("utf-8"))
    body = data[8 + length :]
    out: dict[str, np.ndarray] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec["dtype"] not in _DTYPES:
            raise ValueError(f"{path}: tensor {name!r} has dtype {spec['dtype']}")
        start, end = spec["data_offsets"]
        array = np.frombuffer(
            body[start:end], dtype=np.dtype(_DTYPES[spec["dtype"]]).newbyteorder("<")
        )
        out[name] = array.reshape(spec["shape"]).astype(_DTYPES[spec["dtype"]])
    return out


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SPass:
    tensor: np.ndarray
    sidecar: Mapping[str, Any]


def write_spass(directory: Path, tensor: np.ndarray, sidecar: Mapping[str, Any]) -> Path:
    """Write `tensor` as float16 and `sidecar` as JSON under `directory`."""
    array = np.asarray(tensor)
    if array.ndim != 2:
        raise ValueError(f"an S-Pass is (tokens, width), got shape {array.shape}")
    missing = [f for f in REQUIRED_FIELDS if f not in sidecar]
    if missing:
        raise ValueError(f"the sidecar lacks {missing}")
    if (int(sidecar["tokens"]), int(sidecar["width"])) != array.shape:
        raise ValueError(
            f"the sidecar says {sidecar['tokens']} x {sidecar['width']}, got {array.shape}"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SPASS_FILE
    save_file({TENSOR_NAME: np.ascontiguousarray(array.astype(np.float16))}, path)
    (directory / SIDECAR_FILE).write_text(
        json.dumps(dict(sidecar), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_spass(directory: Path) -> SPass:
    directory = Path(directory)
    path, side = directory / SPASS_FILE, directory / SIDECAR_FILE
    if not path.is_file() or not side.is_file():
        raise FileNotFoundError(f"{directory} holds no S-Pass")
    sidecar = json.loads(side.read_text(encoding="utf-8"))
    missing = [f for f in REQUIRED_FIELDS if f not in sidecar]
    if missing:
        raise ValueError(f"{side} lacks {missing}")
    tensors = load_file(path)
    if TENSOR_NAME not in tensors:
        raise ValueError(f"{path} holds no tensor {TENSOR_NAME!r}")
    tensor = tensors[TENSOR_NAME]
    if tensor.dtype != np.float16:
        raise ValueError(f"{path} holds a {tensor.dtype} tensor; an S-Pass is float16")
    if tensor.shape != (int(sidecar["tokens"]), int(sidecar["width"])):
        raise ValueError(f"{path} is {tensor.shape}, the sidecar says otherwise")
    return SPass(tensor, sidecar)


def write_plain(directory: Path, tensor: np.ndarray) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PLAIN_FILE
    save_file({PLAIN_NAME: np.ascontiguousarray(np.asarray(tensor).astype(np.float16))}, path)
    return path


def read_plain(directory: Path) -> np.ndarray:
    path = Path(directory) / PLAIN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{directory} holds no {PLAIN_FILE}")
    tensors = load_file(path)
    if PLAIN_NAME not in tensors:
        raise ValueError(f"{path} holds no tensor {PLAIN_NAME!r}")
    return tensors[PLAIN_NAME]
