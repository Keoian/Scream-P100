#!/usr/bin/env python3
"""UNVERIFIED PROOF OF CONCEPT, its answers on four cards are not verified: compares the first-token logits of two runs, for example the
4-card engine against the 2-card one on the same machine (docs/POC-4X.md, "Turning it into a verified build").

    python3 engine-4x/compare_logits.py /tmp/logits-2 /tmp/logits-4

Each run is started with FNX_DUMP_FIRST_LOGITS=PREFIX in its profile's "env" and sent the same requests (temperature 0) in the
same order; the engine writes each request's first-token logits (float32, the whole vocabulary) to PREFIX.0, PREFIX.1, ...
For every pair it prints whether the top token is the same, the largest difference of a logit and KL(first || second) of the
softmaxes in nats. The 4-card engine adds the cards' parts in another order and with other roundings than the 2-card one
(four fp16-rounded addends per sum instead of two), so small differences are expected; another top token, or a KL far above
the pairs' usual, is a fault to report.
"""
import math
import sys
from array import array
from pathlib import Path


def load(p):
    a = array("f")
    data = p.read_bytes()
    if len(data) % 4:
        sys.exit(f"{p}: not float32 values")
    a.frombytes(data)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def log_softmax(a):
    m = max(a)
    s = math.log(math.fsum(math.exp(x - m) for x in a))
    return [x - m - s for x in a]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1])
    pa, pb = sys.argv[1], sys.argv[2]
    n = same = 0
    worst = 0.0
    for i in range(10000):
        fa, fb = Path(f"{pa}.{i}"), Path(f"{pb}.{i}")
        if not fa.exists() or not fb.exists():
            break
        a, b = load(fa), load(fb)
        if len(a) != len(b):
            sys.exit(f"{fa} and {fb}: {len(a)} and {len(b)} values")
        ta, tb = max(range(len(a)), key=a.__getitem__), max(range(len(b)), key=b.__getitem__)
        la, lb = log_softmax(a), log_softmax(b)
        kl = math.fsum(math.exp(x) * (x - y) for x, y in zip(la, lb))
        d = max(abs(x - y) for x, y in zip(a, b))
        worst = max(worst, kl)
        same += ta == tb
        n += 1
        print(f"{i}: top {ta} / {tb} {'same' if ta == tb else 'DIFFERENT'}, largest logit difference {d:.4f}, KL {kl:.6f}")
    if not n:
        sys.exit(f"no pairs {pa}.0 / {pb}.0")
    print(f"{n} requests: the same top token in {same}, KL at most {worst:.6f}")


if __name__ == "__main__":
    main()
