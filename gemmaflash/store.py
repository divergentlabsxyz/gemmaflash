"""Write the routed experts of a quantized Gemma 4 MoE checkpoint to a per-layer store.

One file per MoE layer, one fixed-size record per expert, each record holding
that expert's gate/up/down weight, scales and biases bytes back to back
(padded to 16 KiB so a record is one aligned read). The bytes are copied
verbatim from the safetensors shards, so streamed experts are the same
weights. Usage: python -m gemmaflash.store <hf-snapshot-dir> <out-dir>
"""

import fcntl
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

PROJ = ("gate_proj", "up_proj", "down_proj")
PART = ("weight", "scales", "biases")
ALIGN = 16384


def tensors(snapshot):
    for f in sorted(Path(snapshot).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                yield k, f, 8 + n + v["data_offsets"][0], v


def write_uncached(path, buf):
    """Write without leaving the pages in the OS file cache (F_NOCACHE)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        mv = memoryview(buf).cast("B")
        for i in range(0, len(mv), 64 << 20):
            os.write(fd, mv[i:i + (64 << 20)])
    finally:
        os.close(fd)


def main(snapshot, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    by_layer = {}
    for k, f, off, v in tensors(snapshot):
        if ".experts.switch_glu." not in k or "vision" in k:
            continue
        layer = int(k.split(".layers.")[1].split(".")[0])
        proj, part = k.split("switch_glu.")[1].split(".")
        by_layer.setdefault(layer, {})[(proj, part)] = (f, off, v)
    layout = None
    for layer in sorted(by_layer):
        t = by_layer[layer]
        n_exp = t[("gate_proj", "weight")][2]["shape"][0]
        fields, pos = [], 0
        for proj in PROJ:
            for part in PART:
                f, off, v = t[(proj, part)]
                size = (v["data_offsets"][1] - v["data_offsets"][0]) // n_exp
                fields.append(dict(proj=proj, part=part, dtype=v["dtype"], shape=v["shape"][1:], offset=pos, nbytes=size))
                pos += size
        rec = -(-pos // ALIGN) * ALIGN
        layout = layout or dict(num_experts=n_exp, record_bytes=rec, fields=fields)
        assert layout["fields"] == fields and layout["record_bytes"] == rec
        buf = np.zeros((n_exp, rec), np.uint8)
        for fld in fields:
            f, off, v = t[(fld["proj"], fld["part"])]
            raw = np.fromfile(f, np.uint8, count=n_exp * fld["nbytes"], offset=off).reshape(n_exp, -1)
            buf[:, fld["offset"]:fld["offset"] + fld["nbytes"]] = raw
        write_uncached(out / f"L{layer:02d}.bin", buf)
        print(f"layer {layer}: {n_exp} experts x {rec} B", flush=True)
    layout["layers"] = sorted(by_layer)
    (out / "layout.json").write_text(json.dumps(layout, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
