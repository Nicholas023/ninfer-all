#!/usr/bin/env python3
"""Read a GGUF header without loading the model: metadata, tensor census, layer map.

A zero-dependency (standard library only) reader for the GGUF container header — the
`general.*` / architecture metadata, the per-tensor types, and the byte cost of each
tensor family. It never touches the tensor data section, so it works on a multi-gigabyte
file in milliseconds and needs no GPU, no `gguf` wheel and no `numpy`.

Useful when a release's `general.file_type` (for example `MOSTLY_IQ2_M`) hides a mixed
recipe, when you need the routed-expert share of the file before deciding what fits on a
card, or when a fine-tune silently drops a component (MTP head, Vision tower) that its
`config.json` still declares.

    python tools/inspect_gguf_metadata.py MODEL.gguf
    python tools/inspect_gguf_metadata.py MODEL.gguf --json > census.json
    python tools/inspect_gguf_metadata.py MODEL.gguf --families 30

See `docs/gguf.md` for the block formats the runtime multiplies in place and
`docs/qwen36-35b-a3b-gguf-notes.md` for a worked example of the output.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF metadata value-type tags (`gguf_metadata_value_type`).
_META_SCALARS = {
    0: ("<B", 1),   # UINT8
    1: ("<b", 1),   # INT8
    2: ("<H", 2),   # UINT16
    3: ("<h", 2),   # INT16
    4: ("<I", 4),   # UINT32
    5: ("<i", 4),   # INT32
    6: ("<f", 4),   # FLOAT32
    7: ("<?", 1),   # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}
_META_STRING = 8
_META_ARRAY = 9

# ggml tensor type -> (block element count, block byte size). Names match `ggml_type_name`.
# Types the runtime does not use (removed or backend-specific) are absent on purpose.
_GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
}


class _Reader:
    """Sequential little-endian reader over a binary stream."""

    def __init__(self, handle) -> None:
        self._handle = handle

    def raw(self, count: int) -> bytes:
        data = self._handle.read(count)
        if len(data) != count:
            raise EOFError(f"unexpected end of file: wanted {count} bytes, got {len(data)}")
        return data

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        length = self.unpack("<Q")
        return self.raw(length).decode("utf-8", errors="replace")


def _read_metadata_value(reader: _Reader, type_id: int):
    if type_id in _META_SCALARS:
        return reader.unpack(_META_SCALARS[type_id][0])
    if type_id == _META_STRING:
        return reader.string()
    if type_id == _META_ARRAY:
        element_type = reader.unpack("<I")
        count = reader.unpack("<Q")
        return [_read_metadata_value(reader, element_type) for _ in range(count)]
    raise ValueError(f"unknown GGUF metadata value type {type_id}")


def _tensor_bytes(type_id: int, dims: list[int]) -> tuple[int, bool]:
    """Return (payload bytes, exact?) for one tensor of `type_id` and `dims`."""
    if type_id not in _GGML_TYPES:
        return 0, False
    _name, block_elems, block_bytes = _GGML_TYPES[type_id]
    elements = 1
    for dim in dims:
        elements *= dim
    quotient, remainder = divmod(elements, block_elems)
    if remainder:
        return 0, False
    return quotient * block_bytes, True


def read_gguf(path: str | Path) -> dict:
    """Parse the header of `path` into metadata, tensors and aggregate censuses."""
    with open(path, "rb") as handle:
        reader = _Reader(handle)
        magic = reader.unpack("<I")
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic 0x{magic:08x})")
        version = reader.unpack("<I")
        if version < 2:
            raise ValueError(f"GGUF version {version} is too old to read")
        tensor_count = reader.unpack("<Q")
        metadata_count = reader.unpack("<Q")

        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            key = reader.string()
            metadata[key] = _read_metadata_value(reader, reader.unpack("<I"))

        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.unpack("<I")
            dims = [reader.unpack("<Q") for _ in range(n_dims)]
            type_id = reader.unpack("<I")
            offset = reader.unpack("<Q")
            size, exact = _tensor_bytes(type_id, dims)
            type_name = _GGML_TYPES.get(type_id, (f"type#{type_id}", 0, 0))[0]
            tensors.append({
                "name": name,
                "dims": dims,
                "type_id": type_id,
                "type": type_name,
                "offset": offset,
                "bytes": size,
                "exact": exact,
            })

    by_type: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_family: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "bytes": 0, "types": defaultdict(int), "shape": None}
    )
    by_layer: dict[int, list[str]] = defaultdict(list)
    total_bytes = 0

    for tensor in tensors:
        total_bytes += tensor["bytes"]
        entry = by_type[tensor["type"]]
        entry["count"] += 1
        entry["bytes"] += tensor["bytes"]

        family = re.sub(r"^blk\.\d+\.", "", tensor["name"])
        family = re.sub(r"\.\d+$", "", family)
        record = by_family[family]
        record["count"] += 1
        record["bytes"] += tensor["bytes"]
        record["types"][tensor["type"]] += 1
        record["shape"] = tensor["dims"]

        match = re.match(r"^blk\.(\d+)\.(.+)$", tensor["name"])
        if match:
            by_layer[int(match.group(1))].append(tensor["name"])

    return {
        "path": str(path),
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "tensors": tensors,
        "total_bytes": total_bytes,
        "by_type": dict(by_type),
        "by_family": {key: {**value, "types": dict(value["types"])} for key, value in by_family.items()},
        "layers": {index: sorted(names) for index, names in by_layer.items()},
    }


def _layer_kind(names: list[str]) -> str:
    """Classify one block by the tensor names it holds (hybrid attention models)."""
    joined = " ".join(names)
    linear = any(tag in joined for tag in ("ssm_", "attn_gate", "conv1d"))
    full = any(re.search(r"(^|\.)attn_q(\.|$)", name) or ".attn_k" in name for name in names)
    if full and linear:
        return "hybrid?"
    if full:
        return "full-attention"
    if linear:
        return "linear-attention"
    return "?"


def _fmt_gib(num_bytes: float) -> str:
    return f"{num_bytes / (1 << 30):.3f} GiB"


def render(census: dict, families: int) -> str:
    lines: list[str] = []
    metadata = census["metadata"]

    general = {key: value for key, value in metadata.items() if key.startswith("general.")}
    arch = {key: value for key, value in metadata.items()
            if not key.startswith("general.") and not key.startswith("tokenizer.")
            and "." in key}
    tokenizer = {key: value for key, value in metadata.items() if key.startswith("tokenizer.")}

    lines.append(f"file            : {census['path']}")
    lines.append(f"GGUF version    : {census['version']}")
    lines.append(f"tensor_count    : {census['tensor_count']}")
    lines.append(f"metadata (KV)   : {census['metadata_count']}")
    lines.append(f"tensor payload  : {_fmt_gib(census['total_bytes'])}")
    lines.append("")

    def block(title: str, items: dict) -> None:
        lines.append(f"== {title} ==")
        for key in sorted(items):
            value = items[key]
            text = str(value)
            if len(text) > 120:
                text = text[:117] + "..."
            lines.append(f"  {key:<46} = {text}")
        lines.append("")

    block("metadata: general.*", general)
    block("metadata: architecture", arch)
    lines.append("== metadata: tokenizer.* (counts only) ==")
    for key in sorted(tokenizer):
        value = tokenizer[key]
        text = str(value)
        if len(text) > 80:
            text = f"<{type(value).__name__} len={len(text)}>"
        lines.append(f"  {key:<46} = {text}")
    lines.append("")

    lines.append("== quantized type distribution ==")
    for name, entry in sorted(census["by_type"].items(), key=lambda kv: -kv[1]["bytes"]):
        share = 100.0 * entry["bytes"] / census["total_bytes"] if census["total_bytes"] else 0.0
        lines.append(f"  {name:<10} {entry['count']:>5}  {_fmt_gib(entry['bytes']):>12}  {share:5.1f}%")
    lines.append("")

    lines.append(f"== tensor families (top {families} by bytes) ==")
    ordered = sorted(census["by_family"].items(), key=lambda kv: -kv[1]["bytes"])[:families]
    for family, record in ordered:
        types = ", ".join(f"{name} x{count}" for name, count in
                          sorted(record["types"].items(), key=lambda kv: -kv[1]))
        lines.append(f"  {family:<44} x{record['count']:<4} {_fmt_gib(record['bytes']):>12}"
                     f"  shape={record['shape']}  {types}")
    lines.append("")

    if census["layers"]:
        lines.append("== per-layer kind ==")
        for index in sorted(census["layers"]):
            lines.append(f"  blk.{index:<3} {_layer_kind(census['layers'][index])}")
        lines.append("")

    present = " ".join(tensor["name"] for tensor in census["tensors"])
    for component, pattern in (("MTP / nextn draft head", r"nextn\."),
                               ("Vision tower", r"v\.(blk|patch_embd)|mmproj")):
        lines.append(f"  {component:<26}: {'present' if re.search(pattern, present) else 'absent'}")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="path to a .gguf file")
    parser.add_argument("--json", action="store_true", help="emit the raw census as JSON")
    parser.add_argument("--families", type=int, default=25,
                        help="how many tensor families to list (default: 25)")
    parser.add_argument("--no-tensors", action="store_true",
                        help="omit the per-tensor list from JSON output")
    args = parser.parse_args(argv)

    try:
        census = read_gguf(args.model)
    except (OSError, ValueError, EOFError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        payload = dict(census)
        if args.no_tensors:
            payload.pop("tensors", None)
            payload.pop("layers", None)
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render(census, args.families))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
