#!/usr/bin/env python3
"""loop_check.py: the server's loop guard (serve/ink2_server.py's periodic_tail and Loop) on texts, without the model. A
reasoning that falls into repeating two lines (as Inkling's did on 22 Nov after ~20 tool calls) must be cut after the
lines' first copy; ordinary long texts (this branch's docs and the server's own source, read as a reasoning) must never be.
Texts go in as ~4-character tokens, as the server sees a reply."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "serve/ink2_server.py").read_text(encoding="utf-8")
ns = {}
exec(src[src.index("def periodic_tail("):src.index("# what closes a looping reasoning")], ns)
Loop = ns["Loop"]


def cut(text):
    """(characters read when the guard cut, characters kept) or None."""
    g, toks = Loop(), [text[i:i + 4] for i in range(0, len(text), 4)]
    for n, t in enumerate(toks, 1):
        k = g.step("r", [("r", t)], n)
        if k is not None:
            return sum(map(len, toks[:n])), sum(map(len, toks[:k]))
    return None


head = ("The user wants server GPUs better than the P100 in Europe. From the searches: A100 80GB refurbished 7,479 EUR, a DGX "
        "A100 with 8x A100 40GB 50,000 EUR, V100 32GB 425-1,099 USD on eBay, the P100 itself 180-289 EUR.\n\n")
block = ("Let me also check the HydraGPU page via search snippet - it says \"NVIDIA A100 80GB - Refurbished - 7,479 EUR\". "
         "That's a good price.\n\nLet me also check the server-parts.eu DGX A100 offer - it's 50,000 EUR for 8x A100 40GB.\n\n")
bad = 0
r = cut(head + block * 40)
print(f"looping reasoning ({len(head + block * 40)} characters): " + (f"cut after {r[0]}, {r[1]} kept" if r else "NOT CUT"))
# one copy kept (its start can slip back a few characters, into the end of head)
bad += not r or not (len(head) + len(block) - 16 <= r[1] <= len(head) + len(block))
for f in sorted((ROOT.parent / "docs").glob("*.md")) + [ROOT / "serve/ink2_server.py", ROOT / "README.md"]:
    t = f.read_text(encoding="utf-8")
    r = cut(t)
    print(f"{f.relative_to(ROOT.parent)} ({len(t)} characters): " + (f"CUT after {r[0]}" if r else "not cut"))
    bad += bool(r)
print("OK" if not bad else "FAILED")
sys.exit(1 if bad else 0)
