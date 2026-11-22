#!/usr/bin/env python3
"""cpu_experts_real.py GGUF_DIR_OR_SHARD LAYER EXPERT CARD OUT [NTOK]: one card's half of a real routed expert, laid out as
the engine's host copy (model.cpp expert_rows2 / expert_cols: gate rows ff0..ff1 then up rows ff0..ff1; down's N rows of the
blocks of columns ff0..ff1), NTOK random inputs, and the half's outputs computed in double precision from gguf-py's own
dequantizer. tools/cpu_experts_test check OUT compares the CPU path's result with them. Two cards: card 0 is ff 0..319."""
import struct
import sys
from pathlib import Path

import numpy as np

import gguf
from gguf import quants

src, layer, expert, card, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ntok = int(sys.argv[6]) if len(sys.argv) > 6 else 3
shards = sorted(src.glob("*.gguf")) if src.is_dir() else sorted(src.parent.glob(src.name.rsplit("-", 3)[0] + "-*.gguf"))
want = {f"blk.{layer}.ffn_{k}_exps.weight": k for k in ("gate", "up", "down")}
t = {}
for sh in shards:
    for x in gguf.GGUFReader(str(sh)).tensors:
        if x.name in want:
            t[want[x.name]] = x
assert len(t) == 3, f"layer {layer}'s expert tensors not found in {len(shards)} files"
N, FF, NEXP = (int(v) for v in t["gate"].shape)   # gate / up: [N][FF][NEXP]
assert tuple(int(v) for v in t["down"].shape) == (FF, N, NEXP)
ff0, ff1 = card * FF // 2, (card + 1) * FF // 2
ffs = ff1 - ff0


def rows(x, r0, r1, row_values):
    """rows r0..r1 of expert `expert`: raw bytes and float64 values"""
    bs, ts = gguf.GGML_QUANT_SIZES[x.tensor_type]
    rb = row_values // bs * ts
    per = int(x.shape[1]) * rb
    raw = np.asarray(x.data).reshape(-1).view(np.uint8)[expert * per + r0 * rb: expert * per + r1 * rb]
    return raw, quants.dequantize(raw.reshape(r1 - r0, rb), x.tensor_type).astype(np.float64).reshape(r1 - r0, row_values), rb, bs, ts


g_raw, G, _, _, _ = rows(t["gate"], ff0, ff1, N)
u_raw, U, _, _, _ = rows(t["up"], ff0, ff1, N)
d_raw, D, drb, dbs, dts = rows(t["down"], 0, N, FF)
assert dbs == 32, "down is not in 32-value blocks"
b0, b1 = ff0 // 32, ff1 // 32
d_half = d_raw.reshape(N, drb)[:, b0 * dts: b1 * dts]
rng = np.random.default_rng(layer * 1000 + expert)
x = rng.standard_normal((ntok, N)).astype(np.float32)
g, u = x.astype(np.float64) @ G.T, x.astype(np.float64) @ U.T
h = g / (1.0 + np.exp(-g)) * u
o = h @ D[:, ff0:ff1].T
with open(out, "wb") as f:
    f.write(struct.pack("<5i", int(t["gate"].tensor_type), int(t["down"].tensor_type), N, ffs, ntok))
    f.write(g_raw.tobytes()), f.write(u_raw.tobytes()), f.write(np.ascontiguousarray(d_half).tobytes())
    f.write(x.tobytes()), f.write(o.astype(np.float64).tobytes())
print(f"layer {layer} expert {expert} card {card}: {t['gate'].tensor_type.name} gate|up, {t['down'].tensor_type.name} down, {ntok} tokens -> {out}")
