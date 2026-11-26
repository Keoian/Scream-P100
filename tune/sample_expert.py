#!/usr/bin/env python3
"""Writes one card's half of a routed expert out of the model's GGUF files, as raw blocks, for the CPU kernels' bench
(engine/tools/cpu_exact_test.cpp, mode `real`; tune/check.py runs both).

    python3 tune/sample_expert.py --models /data/flash-next/models --out /tmp/experts 10:7 30:200
    python3 tune/sample_expert.py --files a-00001-of-00004.gguf a-00002-of-00004.gguf ... --out DIR 10:7 --card 1

LAYER:EXPERT names the expert. Two files a pair: gu-LAYER-EXPERT.bin (the card's 320 of the expert's 640 gate rows, then its
320 up rows, each 2560 weights wide) and dn-LAYER-EXPERT.bin (the expert's 2560 down rows, each the card's 320 of its 640
weights). Card 0 holds the first half of the 640, card 1 the second. Needs only Python 3 (it reads the GGUF headers itself);
the tensors' types are printed and must be ones the CPU path has kernels for (gate|up Q4_K or Q5_K; down IQ4_NL, Q5_1 or
Q8_0)."""
import argparse
import glob
import os
import struct
import sys

# ggml type -> (name, values a block, bytes a block)
BLOCK = {7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 20: ("IQ4_NL", 32, 18)}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def tensors(path):
    """{name: (type, dims, offset in the file)} of one GGUF file"""
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

        def string():
            return f.read(rd("Q")[0])

        def value(t):
            if t in SCALAR:
                return rd(SCALAR[t])[0]
            if t == 8:
                return string()
            if t == 9:
                et, n = rd("IQ")
                if et in SCALAR:   # skipped in one read
                    f.seek(n * struct.calcsize("<" + SCALAR[et]), os.SEEK_CUR)
                    return None
                return [value(et) for _ in range(n)]
            raise ValueError(f"{path}: metadata type {t}")

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version, nt, nkv = rd("IQQ")
        if version < 2:
            raise ValueError(f"{path}: GGUF version {version}")
        align = 32
        for _ in range(nkv):
            key = string()
            v = value(rd("I")[0])
            if key == b"general.alignment":
                align = int(v)
        info = []
        for _ in range(nt):
            name = string().decode()
            nd = rd("I")[0]
            dims = rd(f"{nd}Q")
            ty, off = rd("IQ")
            info.append((name, ty, dims, off))
        base = (f.tell() + align - 1) // align * align
    return {name: (ty, dims, base + off) for name, ty, dims, off in info}


def sample(files, out, experts, card=0, log=print):
    """experts: (layer, expert) pairs; returns [(gate|up file, down file, gate|up type, down type)]. Raises ValueError with the reason."""
    where = {}
    for p in files:
        for name, t in tensors(p).items():
            if "_exps." in name:
                where[name] = (p,) + t
    os.makedirs(out, exist_ok=True)
    done = []
    for L, e in experts:
        parts = {}
        for kind in ("gate", "up", "down"):
            name = f"blk.{L}.ffn_{kind}_exps.weight"
            if name not in where:
                raise ValueError(f"{name}: not in {len(files)} file(s) (a model whose experts are stored another way)")
            parts[kind] = where[name]
        (pg, tg, dg, og), (pu, tu, du, ou), (pd, td, dd, od) = parts["gate"], parts["up"], parts["down"]
        if tg != tu or tg not in (12, 13) or td not in (7, 8, 20):
            raise ValueError(f"layer {L}: gate|up type {tg}/{tu}, down type {td}: the CPU path's kernels are for gate|up Q4_K or Q5_K and "
                             "down IQ4_NL, Q5_1 or Q8_0")
        n, ff, nexp = dg   # innermost first: an expert is ff rows of n weights; down: n rows of ff weights
        if tuple(dd) != (ff, n, nexp) or tuple(du) != tuple(dg) or not 0 <= e < nexp or ff % 64 or n % 256:
            raise ValueError(f"layer {L}: gate {dg}, up {du}, down {dd}, expert {e}: not the shapes expected")
        half = ff // 2
        rb = n // BLOCK[tg][1] * BLOCK[tg][2]       # a gate or up row
        rbd = ff // BLOCK[td][1] * BLOCK[td][2]     # a down row
        gu = bytearray()
        for p, off in ((pg, og), (pu, ou)):
            with open(p, "rb") as f:
                f.seek(off + e * ff * rb + card * half * rb)
                gu += f.read(half * rb)
        with open(pd, "rb") as f:
            f.seek(od + e * n * rbd)
            rows = f.read(n * rbd)
        dn = b"".join(rows[i * rbd + card * (rbd // 2):i * rbd + (card + 1) * (rbd // 2)] for i in range(n))
        fg, fd = os.path.join(out, f"gu-{L}-{e}.bin"), os.path.join(out, f"dn-{L}-{e}.bin")
        with open(fg, "wb") as f:
            f.write(gu)
        with open(fd, "wb") as f:
            f.write(dn)
        log(f"layer {L} expert {e}, card {card}: gate|up {BLOCK[tg][0]} {2 * half} x {n} ({len(gu)} bytes), down {BLOCK[td][0]} {n} x {half} ({len(dn)} bytes)")
        done.append((fg, fd, BLOCK[tg][0], BLOCK[td][0]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="a folder with the model's *.gguf files")
    ap.add_argument("--files", nargs="+", help="the model's GGUF files instead of --models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", type=int, default=0, choices=(0, 1))
    ap.add_argument("experts", nargs="+", help="LAYER:EXPERT")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob(os.path.join(a.models or ".", "*.gguf")))
    if not files:
        sys.exit("no GGUF files")
    try:
        sample(files, a.out, [tuple(int(v) for v in spec.split(":")) for spec in a.experts], a.card)
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
