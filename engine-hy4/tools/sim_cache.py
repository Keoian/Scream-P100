#!/usr/bin/env python3
"""sim_cache.py SLED.bin [SLOTS ...] (2026-11-20): hy4_v1's expert cache under other policies, on a routing trace (HY4_TRACE:
int32 layer, int32 T, then T * 16 picks; layer -1: a prompt-path chunk, whose staging empties 1024 slots and leaves layers 40
and 41 cached). Misses a token for: the engine's LRU, LFU with decay, a 2-segment LRU, and Belady's MIN (the bound: it knows
the future). A window's distinct picks are never evicted for each other."""
import heapq, random, sys
from collections import OrderedDict

import numpy as np

NEXP, TOPK = 512, 16


def read(path):
    a = np.fromfile(path, np.int32)
    ev, i = [], 0
    while i < len(a):
        il, t = int(a[i]), int(a[i + 1])
        i += 2
        if il < 0:
            ev.append(("pf", t))
            continue
        ids = a[i:i + t * TOPK]
        i += t * TOPK
        keys = list(dict.fromkeys(int(il) * NEXP + int(e) for e in ids))   # distinct, in order
        ev.append(("win", il, t, keys))
    return ev


def tokens(ev):   # writing windows of one token count once a layer: tokens = windows at layer 1 (T counted)
    return sum(e[2] for e in ev if e[0] == "win" and e[1] == 1)


def pf_keys():
    return [il * NEXP + e for il in (40, 41) for e in range(NEXP)]


def lru(ev, S, rng):
    c = OrderedDict()
    miss = 0
    for e in ev:
        if e[0] == "pf":
            for k in rng.sample(list(c.keys()), min(len(c), 2 * NEXP)):
                del c[k]
            for k in pf_keys():
                c[k] = 1
                c.move_to_end(k)
                if len(c) > S:
                    c.popitem(last=False)
            continue
        keys = e[3]
        cur = set(keys)
        for k in keys:
            if k in c:
                c.move_to_end(k)
                continue
            miss += 1
            if len(c) >= S:
                for v in c:
                    if v not in cur:
                        del c[v]
                        break
            c[k] = 1
    return miss


def slru(ev, S, rng, prot=0.8):   # probation (new keys) and protected (hit at least twice) segments, both LRU
    pr, po = OrderedDict(), OrderedDict()
    miss = 0
    P = int(S * prot)
    for e in ev:
        if e[0] == "pf":
            allk = list(pr.keys()) + list(po.keys())
            for k in rng.sample(allk, min(len(allk), 2 * NEXP)):
                pr.pop(k, None)
                po.pop(k, None)
            for k in pf_keys():
                if k not in po:
                    pr[k] = 1
                    pr.move_to_end(k)
            while len(pr) + len(po) > S:
                pr.popitem(last=False) if pr else po.popitem(last=False)
            continue
        cur = set(e[3])
        for k in e[3]:
            if k in po:
                po.move_to_end(k)
                continue
            if k in pr:
                del pr[k]
                po[k] = 1
                while len(po) > P:   # demote the coldest protected to probation's top
                    v, _ = po.popitem(last=False)
                    pr[v] = 1
                continue
            miss += 1
            while len(pr) + len(po) >= S:
                done = False
                for seg in (pr, po):
                    for v in seg:
                        if v not in cur:
                            del seg[v]
                            done = True
                            break
                    if done:
                        break
            pr[k] = 1
    return miss


def lfu(ev, S, rng, half=200.0):
    """score = count decayed with a half-life in tokens; all decay alike, so log(score) + lam * stamp orders them for good until
    the key is used again: a lazy min-heap on that"""
    pri, cnt, stamp = {}, {}, {}
    heap, now, miss = [], 0.0, 0
    lam = np.log(2) / half

    def touch(k, c):
        cnt[k], stamp[k] = c, now
        pri[k] = np.log(c) + lam * now
        heapq.heappush(heap, (pri[k], k))

    def evict(cur):
        held = []
        while heap:
            p, v = heapq.heappop(heap)
            if v in pri and pri[v] == p:
                if v in cur:
                    held.append((p, v))
                    continue
                del pri[v], cnt[v], stamp[v]
                break
        for h in held:
            heapq.heappush(heap, h)
    for e in ev:
        if e[0] == "pf":
            for k in rng.sample(list(pri.keys()), min(len(pri), 2 * NEXP)):
                del pri[k], cnt[k], stamp[k]
            for k in pf_keys():
                touch(k, 1.0)
            while len(pri) > S:
                evict(set())
            continue
        il, t, keys = e[1], e[2], e[3]
        if il == 1:
            now += t
        cur = set(keys)
        for k in keys:
            if k in pri:
                touch(k, cnt[k] * np.exp(-lam * (now - stamp[k])) + 1.0)
                continue
            miss += 1
            if len(pri) >= S:
                evict(cur)
            touch(k, 1.0)
    return miss


def belady(ev, S):
    seq = []   # (event index, key) of every access; pf events handled as in the others but deterministic (drop farthest)
    for i, e in enumerate(ev):
        if e[0] == "win":
            for k in e[3]:
                seq.append((i, k))
    nxt = [0] * len(seq)
    last = {}
    for j in range(len(seq) - 1, -1, -1):
        k = seq[j][1]
        nxt[j] = last.get(k, 1 << 60)
        last[k] = j
    cache, heap, miss = {}, [], 0
    j = 0
    for i, e in enumerate(ev):
        if e[0] == "pf":
            continue   # the bound ignores the staging
        keys = e[3]
        cur = set(keys)
        for k in keys:
            n = nxt[j]
            j += 1
            if k in cache:
                cache[k] = n
                heapq.heappush(heap, (-n, k))
                continue
            miss += 1
            if len(cache) >= S:
                held = []
                while heap:
                    nn, v = heapq.heappop(heap)
                    if v in cache and cache[v] == -nn:
                        if v in cur:
                            held.append((nn, v))
                            continue
                        del cache[v]
                        break
                for h in held:
                    heapq.heappush(heap, h)
            cache[k] = n
            heapq.heappush(heap, (-n, k))
    return miss


def main():
    ev = read(sys.argv[1])
    slots = [int(s) for s in sys.argv[2:]] or [7100]
    nt = tokens(ev)
    acc = sum(len(e[3]) for e in ev if e[0] == "win")
    print(f"{nt} tokens, {acc} distinct picks, {sum(1 for e in ev if e[0] == 'pf')} prompt-path chunks")
    for S in slots:
        rng = random.Random(1)
        r = {"LRU": lru(ev, S, rng), "SLRU": slru(ev, S, random.Random(1)), "LFU200": lfu(ev, S, random.Random(1)),
             "MIN": belady(ev, S)}
        print(f"S={S}: " + ", ".join(f"{k} {v / nt:.1f}/tok ({100 * (1 - v / acc):.1f} % hits)" for k, v in r.items()))


if __name__ == "__main__":
    main()
