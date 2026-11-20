#!/usr/bin/env python3
# UNVERIFIED PROOF OF CONCEPT: its answers on four cards are not verified (engine-4x/README.md).
#
# rows_tail_patch.py (2026-11-20): the decode windows' row kernel (k_moe_rows, p100_moe_rows.cuh) for a down slice whose count of
# 32-column runs leaves one run in its last group. The down types with four runs a group (Q4_0, IQ4_NL, Q5_1, Q8_0) check runs
# 2 and 3 against the slice's end, but not run 1. With 160 columns (five runs, the even four-card split) the last group is run 4
# alone, and the kernel added run 5 too: 32 columns past the slice, decoded from the next row's bytes, times the next token's
# activations (or whatever lies past the last one). That is the wrong first-token logits of the 160-column build on four P100,
# short prompts included (they are read through the decode windows), and their run-to-run difference (KaneQuah, issue #1).
# Two cards (320 columns, ten runs), three (224 and 192), six (128 and 96) and eight (96) never leave run 1 past the end, so
# only four cards' even split met it. With ff64_patch.py's slices (192 and 128) the new check is always true: the same sums.
# Run in the engine's source folder (the one holding src/).
import sys
from pathlib import Path

S = Path("src")
if not (S / "p100_moe_rows.cuh").exists():
    sys.exit("rows_tail_patch.py: run it in the folder that holds src/")


def rep(name, old, new):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != 1:
        sys.exit(f"rows_tail_patch.py: {name}: {n} matches, want 1:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


if "rows_tail_patch" in (S / "p100_moe_rows.cuh").read_text(encoding="utf-8"):
    sys.exit("rows_tail_patch.py: applied already")

rep("p100_moe_rows.cuh",
    "                row_run<TY, TT, 0>(W, tab, t, r0, xp, n, s);\n"
    "                row_run<TY, TT, 1>(W, tab, t, r0 + 1, xp, n, s);\n"
    "                if (r0 + 2 < nruns) row_run<TY, TT, 2>(W, tab, t, r0 + 2, xp, n, s);\n",
    "                row_run<TY, TT, 0>(W, tab, t, r0, xp, n, s);\n"
    "                if (r0 + 1 < nruns) row_run<TY, TT, 1>(W, tab, t, r0 + 1, xp, n, s);   // rows_tail_patch.py: 160 columns end on run 4\n"
    "                if (r0 + 2 < nruns) row_run<TY, TT, 2>(W, tab, t, r0 + 2, xp, n, s);\n")
print("rows_tail_patch.py: the row kernel's last group checks run 1 against the slice's end")
