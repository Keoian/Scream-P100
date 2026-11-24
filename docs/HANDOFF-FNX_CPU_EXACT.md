# `FNX_CPU_EXACT`: the CPU's missing experts, bit for bit the card's

Branch `fnx-cpu-exact`: PR #8's branch `fnx-cpu-experts-exact` on main, which has `FNX_CPU_EXPERTS` since PR #7.
`FNX_CPU_EXPERTS=1 FNX_CPU_EXACT=1` computes routed experts that are not in VRAM on the CPU with the card's own
arithmetic, so the engine's output is the card-only build's, byte for byte, whatever is in VRAM at a window. Off by
default; without the switch the CPU path is PR #7's fp32 one. `FNX_CPU_SHARE=A,B` beside it shares each window's missing
experts between the copy engine and the CPU (A per cent on the CPU in a window of 1 row, B in one of 16); any split
gives the same bits. On the server it was measured on (one card in a x8 slot, fast CPUs) it is faster than no CPU path
on every writing test and slower than the fp32 path; on the maintainer's (both cards on x16, slow CPUs) it holds the
bits and loses to the cards alone ("Verified"):

| tok/s | no CPU path (#5's build) | fp32 path (#7) | exact (this branch) | exact against no CPU path |
|---|---:|---:|---:|---:|
| **File A** (bitlamas' Q4_K_XL-IQ4NL): fresh writing | 47.6 | 59.7 | 56.2 | +18 % |
| edit py | 53.9 | 110.1 | 101.0 | +87 % |
| edit md | 54.9 | 117.5 | 105.6 | +92 % |
| writing after a 38,862-token read | 41.9 | 47.2 | 45.1 | +8 % |
| **File B** (Unsloth's UD-Q4_K_XL): fresh writing | 36.8 | 52.3 | 50.9 | +38 % |
| edit py | 37.5 | 92.8 | 82.0 | +118 % |
| edit md | 37.5 | 97.4 | 84.0 | +124 % |
| writing after a 38,862-token read | 34.1 | 44.2 | 42.5 | +25 % |

(Means of three rounds, every run a fresh load; "Speed" below has the spread. File A's fresh writing without a CPU path is
its undisturbed run's: two of the three had stalled requests.)

`tune/tune.py` finds the shares on another machine and checks the bits of what it writes
([tune/README.md](../tune/README.md#the-exact-cpu-path)).

Two Claude sessions wrote it on 23 and 24 Nov 2026: the arithmetic and its CPU tests on an i7-7700 without a card (the
first commit), then the run on two Tesla P100 with what that found and changed, and the speed (the commits after it; a
2 x E5-2696 v3 server, one card in a x8 slot). This file is the record: the spec, how the CPU gets the card's bits and
how it got fast, what is verified and how, the speed, what was tried and not kept, what is left.

## Why

With `FNX_CPU_EXPERTS=1` the CPU sums in fp32 and the cards' row kernels in fp16, so the CPU's rows are close to the cards'
but not the same bits, and which experts are in VRAM at a window depends on timing: a run with the switch differs from a
run without it, and from the next run with it. The maintainer's profile is bit-exact and stays so. With `FNX_CPU_EXACT=1`
the CPU computes exactly what the card would have.

## What the card computes (the spec the CPU now follows)

From `p100_moe_rows.cuh` (`k_moe_rows`, `row_run`) and `p100_iq_gemm.cuh` (`kq_finish`, `finish_s`, `fold_one`):

- weights, rounded to fp16 once: Q4_K/Q5_K `fp16(q * a + b)` with `a = fp16(d * sc)`, `b = fp16(-dmin * m)` (one HFMA2);
  IQ4_NL `fp16(kvalue * d)`, Q8_0 `fp16(q * d)` (HMUL2); Q5_1 `fp16(q * d + m)` (HFMA2);
- each 32-value run: two fp16 FMA chains (lane .x the even values 0, 2, ..., 30, lane .y the odd ones), 16 steps each,
from +0,
  then into fp32 as `s = s + (lo + hi)`;
- gate|up rows: `FNX_ROW_G` lanes a row (2 by default), part p the 256-value groups `[p*ngr/G, (p+1)*ngr/G)`, the parts
added by
  the xor shuffles (lane 0: `((s0 + s1) + (s2 + s3)) + ...`); down rows one lane;
- SwiGLU: `__float2half(silu_over(g) * u)` with `silu_over(x) = x / (1 + __expf(-x))` under `--use_fast_math` (MUFU.EX2
and an
  approximate division: not reproducible on a CPU).

## What the branch adds

| File | What |
|---|---|
| `engine/src/cpu_experts.cpp/.hpp` | `Work::exact`, `Work::gu_lanes`, `Expert::gu_out`; an oracle doing IEEE binary16 on the integers (`f16_fma_ref`, `f16_from_f32_ref`) and the card's row sums with it (`gu_row_card_ref`, `dn_row_card_ref`); AVX2 kernels with the card's arithmetic (`gu16_exact`, `dn16_exact`: 16 rows at a time, a row a lane; the step `fma16z`, eight chains in flight in `chain64`, `fold_runs` with the oracle for a run that left fp16's range); `Pool::run_gu` / `run_down` in chunks of one kernel call; `Pool::stats()`; the pool's threads beyond the cores on the cores' second hyperthreads (`core_siblings`) |
| `engine/src/moe.cuh`, `kernels.hpp`, `p100_moe_rows.cuh` | the plan's split (`cache_plan` with `cpu == 2`): the layer's misses with the fewest rows listed for the CPU, slots and the copy list for the others, both published by the router's last block; a job of the CPU's list is marked so that neither phase of the row kernels takes it (`JOB_CPU`) |
| `engine/src/p100_cpu.cuh` | `FNX_CPU_EXACT=1`: the host thread computes gate\|up and writes the sums to mapped host memory; on the card's third stream one block (`k_cpu_swiglu1`) waits for them, fetches them with a sum check, computes h with the very expression of `k_swiglu_after` into VRAM and posts it to mapped host memory with its sum and the list's number; the host thread checks the sum, reads h and computes down. The mailbox's post (`k_cpu_post`) is one block with a sum too. `FNX_CPU_CHECK` counts the values whose bits differ (`CpuDev::bitdiff`) and, with the exact switch, the stage: the CPU's gate\|up sums and the SwiGLU made of them against the card's own (`k_cpu_check_gh`). The report (`FNX_PROF=1`) has the list's stages by the window's rows and the pool's counters. Measuring only: `FNX_CPU_GCOPY` (the sums by a copy and three launches, as first written), `FNX_CPU_POST_ROWS`, `FNX_CPU_HB_DIRECT`, `FNX_CPU_DNPROBE`, `FNX_CPU_REUSE`, and `FNX_CPU_WARM` |
| `engine/src/engine.cu` | `FNX_CPU_SHARE`; with a split the copy engine's second phase and then the CPU's take; `p100_cpu_swiglu` after `p100_cpu_post` on `st3`; under `FNX_CPU_CHECK` a fork of `st` to `st3` after the publish, joined before the take, then `p100_cpu_check_stages` |
| `engine/tools/cpu_exact_test.cpp` | CPU only: the oracle against F16C, `fma16z` against the oracle (finite operands: the oracle's bits; infinite ones: its bits or NaN, which the run's end catches), every kernel against the card's steps (direct and through the pool, with runs that go through the oracle), a `bench` mode and `real GU.bin DN.bin NTOK THREADS REPS`: the kernels on an expert sampled from the model's files |
| `engine/tools/cpu_exact_gpu_test.cu` | on a P100: HFMA2 / HMUL2 / `__float2half_rn` against the oracle; random experts of every covered type, 1-16 tokens: the card's row kernels against the CPU's, the SwiGLU kernels against each other (the one-block one among them), down of the card's h |
| `engine/CMakeLists.txt`, `CMakeLists.txt` | targets `split_cpu_exact_test` and `split_cpu_exact_gpu_test` (`-DSTRATA_BUILD_TESTS=ON`); the GPU test gets `p100_compat.h` like the engine |
| `tune/sample_expert.py` | one card's half of a routed expert out of the GGUF files, as raw blocks (plain Python) |
| `tune/check.py` | the exact kernels built with the host compiler and timed on two sampled experts (one core, a card's pool), beside the copy of the same bytes to each card; from the two a first guess for the 16-row share |
| `tune/tune.py` | the group `cpu_exact` (no option: the bits are the baseline's): the shares, `FNX_CPU_MIN_T`, the pool on both hyperthreads; a profile that keeps the path is loaded once more with `FNX_CPU_CHECK=1` and written only if no value differed |
| `docs/SWITCHES.md`, `docs/GUIDE.md`, `tune/README.md` | the switches' rows, the section, the tuner's part |

Covered types: gate|up Q4_K or Q5_K, down IQ4_NL, Q5_1 or Q8_0 (every layer), `FNX_ROW_G` 1/2/4/8; otherwise the engine says
why and keeps the fp32 path. Requires `FNX_ROWS_G2` unset, as the profiles have it (with it set, phase 2 sums in another
order and the card-only run itself depends on what is in VRAM). The two files it ran on have all five types between them:
both have Q4_K gate|up in 47 layers and Q5_K in one; ours has IQ4_NL down rows, Unsloth's UD-Q4_K_XL Q5_1 in 43 layers
and Q8_0 in five.

## How the CPU gets the card's bits

- A product of two fp16 values has at most 22 significant bits: exact in fp32.
- `fma16z`, the kernels' step, with MXCSR rounding toward zero (the kernels set it and put it back): `a = fma(w, x, c)` is
  the fp32 sum truncated; `d = c - a`; `e = fma(w, x, d)` is +0 exactly when `a` is the sum, and its bits are not 0 when it
  is not (a miss would need the truncation's error under one fp32 place of `w x`, but that error is a multiple of `w x`'s
  last bit, two places higher); `a` with its last bit set where `e` is not 0 is the sum rounded to odd, and `vcvtps2ph`
  (to nearest by its own immediate, whatever MXCSR says) of that is the correctly rounded fp16 sum (24 bits >= 11 + 2:
  Boldo, Melquiond). Nine vector uops, no test, no second pass. The first version (`fma16f`: one fp32 rounding, then a
  test for "inexact and on an fp16 midpoint", and the run again by an exact form where a lane flagged) took 15 uops a
  step and redid 3 % of the runs; it is gone.
- An infinite operand gives NaN where the card says infinity. A chain meets one only after it left fp16's range, stays
  NaN to its run's end, and `fold_runs` (which adds the runs into the rows' fp32 sums, back in round-to-nearest, in a
  function of its own so that the compiler cannot move those adds among the chains' steps) does such a run again by
  the oracle, a row at a time. The kernel test's wide ranges go through that path (137,000 runs of 41,472 rows'
  worth); on the model's files it has not happened (0 runs in every engine run).
- Decode: `q a + b` with q < 32 is exact in fp32 when a and b are normal fp16 values with exponents at most 7 apart (or
  either is 0): then one rounding of the FMA, the same in any rounding mode; otherwise `fma16z`. On two experts of our
  file every one of 12,800 groups took the first way.
- The xor-shuffle order of the lanes, the run order, `s + (lo + hi)`: as listed above. No `-ffast-math` anywhere; every op
  is an IEEE op, so the results are the same on any x86 with AVX2/FMA/F16C.

## How it is fast

Where the time went first (23 Nov, the server below): a list of a layer's missing experts took the CPU 3.7 times the fp32
path's time (4.3 ms for 23 experts in a 16-row window) and the cards waited for it; with every window on the CPU the
exact path wrote fresh text at 32 tok/s against 47.5 with no CPU path at all. Five changes, each measured:

1. **The step with no test, eight chains in flight.** Stage timers in the kernel on real experts of the file: decoding a
   run (32 values x 16 rows) took 324 cycles and its chains 361 cycles a row; the test was the cost, not the rounding
   (without it, which is not exact: 208 cycles with four chains, 127-141 with eight). With `fma16z` a step is 9 uops but
   some 25 cycles long, so two runs of the same 16 rows go through their steps together (`chain64`: eight chains; four
   were slower than the old code, sixteen no better than eight). One core, one expert half: 1,355 -> 1,173 us for one
   row, 3,284 -> 2,539 us for four (the chains 643 -> 455 us a row; the decode unchanged at about 715 us).
2. **Chunks of one kernel call.** The pool's chunks were the fp32 path's (10 + 10 an expert half, sized for a path bound by
   memory); they are one call of `gu16_exact` or four of `dn16_exact` now (40 + 40). With the first change, one expert on
   16 threads: 199 -> 116 us for one row, 431 -> 257 us for four.
3. **A window's misses shared between the copy engine and the CPU** (`FNX_CPU_SHARE=A,B`). The planner gives a part of the
   layer's misses slots, as without the CPU path, and lists the others for the CPU; the copies, the card's second phase
   for the copied experts and the CPU's list run at the same time, and both give the card's bits, so any split is exact.
   The CPU's part is the misses picked by the fewest rows (a copy costs the same however many rows use the expert, the
   CPU's chains cost by the row), A per cent of them in a window of 1 row and B in one of 16, a line between, to the
   nearest whole expert. The copied experts stay in VRAM, so the cache follows the text again (a 16-row window had 23
   misses a layer with every miss on the CPU, about 16 with the split).
4. **The hand-off off the copy queue.** The CPU's gate|up sums went to the card by `cudaMemcpyAsync`, behind the copier's
   experts already queued on the same link: 726 us a 16-row list on the x8 card (36-38 us with nothing copied). The card
   now fetches them from mapped host memory, checking their sum as the host checks the card's posts.
5. **The hand-off as one block.** Wait, fetch, SwiGLU and post were three launches; the later two waited for a slot among
   the row kernels (67-185 us a list). One block that is launched with the list and is running when the sums come:
   38-53 us beside small windows, 73 at 16 rows. `split_cpu_exact_gpu_test` runs it against the 40-block kernel and the
   card's own two.

Then the settings: the shares and the threads. On file A the 1-row share is flat between 50 and 70 per cent (30 loses 6
% of fresh writing, 80 loses 6 % of the writing after a read); the 16-row share has a sharp best value, 75 with 32
threads (65 and 85 lose 3-4 % of the edits, 60 loses 7 %); the pool on both hyperthreads of its cores
(`FNX_CPU_THREADS=32`) gains 4-7 % on the edits at a 16-row share of 75 and nothing at 60. `tune/tune.py` finds these
for a machine: `FNX_CPU_SHARE=70,75 FNX_CPU_THREADS=32` for file A here, `70,70` with 32 threads for UD-Q4_K_XL. The
prefetch stays as the profile has it (off loses 12 % of the edits, four guesses a row 10 %).

## Verified

Everything below is the final build (the engine as of the last engine commit; sha256 `ba7ceb4a...` of the binary
measured), on the server: two Tesla P100 16 GB capped at 150 W (card 0 on PCIe 3.0 x8), one card per socket of 2 x Xeon
E5-2696 v3, sm_60, CUDA 12.8, 23 and 24 Nov 2026. File A is bitlamas' Q4_K_XL-IQ4NL (7,778 of 24,576 experts in VRAM at
262,144 positions) with `FNX_CPU_SHARE=70,75 FNX_CPU_THREADS=32`, file B Unsloth's UD-Q4_K_XL with this repository's
512k profile (6,691 in VRAM) and `FNX_CPU_SHARE=70,70 FNX_CPU_THREADS=32`: the settings `tune.py` wrote for each.
"Card-only" is PR #5's build (no CPU path).

The CPU alone, `split_cpu_exact_test 8` (the first version also on an i7-7700 without a card): the oracle's known
answers (including the double-rounding case 1.5 x 0.6669921875 + 2^-24 = 1.0009765625) and its rounding against F16C on
8 M floats and around every half; `fma16z` on 9,015,936 triples of finite operands bit for bit the oracle's (192 of
them where fp32-then-fp16 would have been wrong) and 32,768 with an infinite or NaN operand (the oracle's bits or NaN);
41,472 row sums of every covered type with 1, 2, 4 lanes and 1 or 3 tokens bit for bit the card's steps, directly and
through the pool, 136,803 and 622,767 of their runs going past fp16's range and through the oracle.

On each card, `split_cpu_exact_gpu_test`: 8,388,608 x 2 HFMA2 lanes (every half as a first operand), as many HMUL2,
8,388,608 float -> half: 0 / 0 / 0 differ from the oracle; row kernels gate|up 276,480 values, the SwiGLU kernels
138,240 (the one-block kernel against the 40-block one inside the engine's own probe), down 1,105,920: 0 not the card's.

The engine:

| | File A | File B (UD-Q4_K_XL) |
|---|---|---|
| `FNX_CPU_CHECK=1` (the copy engine and the cards compute every window as without the path; each row the CPU computed is compared with the card's own) | 3,172,127 rows a card in four runs, 1,596,630 of them at the final settings: 0 values not bit for bit, by stage 0 gate\|up sums and 0 SwiGLU values, every post whole | 3,293,093 rows a card in three runs (1,113,675 + 1,082,504 at the final settings, 1,096,914 in the tuner's check): 0 values not bit for bit, by stage 0 gate\|up sums and 0 SwiGLU values, every post whole |
| First-token logits of `bench.py`'s 27 requests, the CPU's rows used | byte for byte the card-only build's in each of three rounds; the three runs each other's | byte for byte the card-only build's in each of three rounds; the three runs each other's (the card-only build's three runs repeat too; the fp32 path's differ from each other in 26 of 27 files) |
| The decode-path yardstick (4,995 positions teacher-forced in windows of 16 rows; a TSV row a position) | the card-only run's file byte for byte, twice | the file of this file's own run without a CPU switch, byte for byte, twice |
| HumanEval, 164 problems, greedy | all 164 answers the card-only build's, text for text (158 passed in both; 37,295 completion tokens in both) | all 164 answers the card-only build's, text for text (158 passed in both; 37,471 completion tokens in both); median writing 60.6 tok/s against 31.2 |
| `FNX_VRAM_CHURN=600` with `--vram-elastic` (eight greedy answers) | the eight texts and the eight logit files the card-only build's without the hook | the eight texts and the eight logit files the card-only build's without the hook (670 lends and give-backs) |
| Pictures on demand (`--vram-elastic`, one sequence) | its six answers the card-only build's, text for text | its six answers the card-only build's, text for text |
| A control vector (`FNX_CVEC`, `project` on layers 4-44) with pictures on demand, the check | 771,285 rows a card, 0 values | 1,068,144 rows a card, 0 values |

`tune.py`'s own check of the profile it writes is among the rows counted (792,199 rows a card on file A, 1,096,914 on
file B).

**On another machine** (the maintainer's, 24 Nov, their numbers from PR #8's thread, the branch at `26fe28f`: two P100 on
x16, 2 x Xeon Bronze 3106 with 8 cores a socket at 1.7 GHz, UD-Q4_K_XL at 512k): `FNX_CPU_CHECK=1` 0 values on both cards,
every exact arm's first-token logits the cards-only run's byte for byte, and with no CPU switch 27 of 27 logits and the
speed of main. The speed is the other way round there: cards only 51.8-54.8 tok/s fresh and 64 on the edits; exact with
`FNX_CPU_SHARE=25,40` 41.9 and 49-51, with `40,40` 37.3 and 49-50, with `60,60` 28.9 and 36-38. A list costs those CPUs
about five times what it costs ours, and the copies half. So the path holds its bits on a second machine and is for
machines like the first: a card on a narrow link, cores to spare.

## What the cards found: the pool, not the arithmetic

The first engine runs were not bit-exact: with the check, 5 and then 10 rows of ~785,000 a run on the card in the x8 slot
(never on the other) were not the card's, and the yardstick left the card-only file in every segment. The check's stage
counters (added for this) put it at the CPU's gate|up sums, 16 at a time: the output of one pass of `gu16_exact`. The cause
was in `cpux::Pool`, PR #7's code, and so in the fp32 path too: a round's count of chunks was a word of its own, written
before the round's ticket. A worker that had just finished a round's last chunk read that round's spent ticket and then
the NEXT round's larger count (the host thread stores it at that moment); if its compare-and-swap also came before the new
ticket, it took a chunk beyond the old round, computed that chunk of the new round and counted it done there, and the new
round ended one chunk early: the host thread went on with rows a worker was still writing. The rows' round (a short chunk
a row, windows of more than 4 rows) before the gate|up round is where it happened, about once in six thousand such lists
on that card. `fnx-cpu-experts` has the fix as its own commit (the round's number and count in one word, `limit`; a ticket
is only measured against its own round's count; such workers are turned away and counted: over a hundred a card in one
check run). With it every check above is clean.

The hand-off itself was not at fault: the first version's posts to mapped host memory (the mailbox from a block a row,
h from the SwiGLU kernel's 40 blocks, the list's number from the last block to finish) arrived whole in every one of
113,604 lists a card when measured (`FNX_CPU_POST_ROWS=1 FNX_CPU_HB_DIRECT=1`, kept for that). They are one block each
now, and carry a sum the host thread checks, so that "the CPU read what the card wrote" is a line of the report
(`mailbox 0 lists read again, 0 never whole; SwiGLU rows 0 read again, 0 never whole`) and not an assumption.

## Speed

Three rounds in a Latin square (B C X / X B C / C X B), every run a fresh load, `bench/bench.py`, tok/s, mean
(min-max). B is PR #5's build, C is PR #7's fp32 path with the pool's fix, X this branch with the settings above.

**File A:**

| | B: #5's build (no CPU path) | C: #7's fp32 path | X: `FNX_CPU_EXACT=1` with the split |
|---|---:|---:|---:|
| fresh writing | 47.6, 44.0, 38.7 (see below) | 59.7 (59.5-60.0) | 56.2 (56.1-56.4) |
| edit py | 53.9 (53.9-53.9) | 110.1 (109.7-110.6) | 101.0 (100.8-101.2) |
| edit md | 54.9 (54.9-54.9) | 117.5 (117.2-117.8) | 105.6 (105.6-105.8) |
| the 38,862-token prompt: reading | 884 (876-891) | 890 (889-890) | 890 (889-890) |
| writing after it | 41.9 (41.3-42.4) | 47.2 (47.2-47.3) | 45.1 (44.5-45.5) |

Two of B's three runs had requests stalled on the host (their fresh writing reads 44.0 and 38.7: one and two of 12
requests at half speed, with page faults in every round of those requests and the cards idle meanwhile; the same
runs' other tests are at their usual values, and no run of C or X had one). B's undisturbed fresh writing is 47.6 (its
first run; 47.8 in a fourth run after the rounds; 47.5-47.6 in every baseline of the tuner the same night), which is
what X is held against: **+18 % on fresh text, +87 % and +92 % on the edits, +8 % after the long read; reading is
unchanged.** Against the fp32 path it is 6 % behind on fresh text and 8-10 % on the edits.

**File B:**

| | B: #5's build (no CPU path) | C: #7's fp32 path | X: `FNX_CPU_EXACT=1` with the split |
|---|---:|---:|---:|
| fresh writing | 36.8 (36.2-37.6) | 52.3 (50.9-54.1) | 50.9 (50.7-51.0) |
| edit py | 37.5 (37.5-37.6) | 92.8 (91.5-93.5) | 82.0 (81.6-82.2) |
| edit md | 37.5 (37.4-37.6) | 97.4 (96.2-98.5) | 84.0 (83.7-84.2) |
| the 38,862-token prompt: reading | 818 (765-900) | 898 (897-898) | 896 (895-896) |
| writing after it | 34.1 (33.9-34.5) | 44.2 (40.0-52.6) | 42.5 (41.8-43.3) |

**+38 % on fresh text, +118 % and +124 % on the edits, +25 % after the long read.** Against the fp32 path it is 3 %
behind on fresh text and 12-14 % on the edits. This file gains more than file A because its expert halves are larger
(1.54-1.79 MB against 1.38), so each copy costs more (284 µs to card 0 against 236) while the CPU's time is about the
same, and fewer of its experts are in VRAM with the 512k profile (6,691 against 7,778).

B's loads came in two states on this file: one read the long prompt at 900 tok/s and wrote fresh text at 37.6, two read
at 765 and 788 and wrote at 36.2 and 36.6, every request alike (the tuner's two baselines the night before were the same
pair). C and X read at 895-898 in every run. We did not trace it; `FNX_CPU_EXPERTS` binds each card's RAM copy of the
experts to the card's own CPU (`FNX_NUMA_EXPERTS`) and B does not, which may be the difference. Against B's better run X
is +35 % on fresh text and +23 % after the long read; the edits are the same in all three.

Where the time goes (`FNX_PROF=1`, one run an arm in one window, file A; ms a window by the window's rows; "cards
wait" is what the cards wait for the CPU's rows; the CPU's list is a layer's experts for the CPU on card 0):

| rows | windows (none / fp32 / exact) | no CPU path | fp32 path | exact | fp32: cards wait | exact: cards wait | CPU's list, fp32: experts, µs | CPU's list, exact: experts, µs |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 17 / 17 / 17 | 23.4 | 23.0 | 25.0 | 4.4 | 3.2 | 2.6, 154 | 1.2, 217 |
| 2 | 888 / 558 / 576 | 35.6 | 29.5 | 32.3 | 4.6 | 3.1 | 3.8, 190 | 1.6, 256 |
| 3 | 632 / 803 / 706 | 46.1 | 34.2 | 37.9 | 6.1 | 3.0 | 5.2, 251 | 2.2, 312 |
| 4 | 243 / 357 / 373 | 59.1 | 41.0 | 45.0 | 8.5 | 3.5 | 7.3, 334 | 2.8, 378 |
| 5 | 53 / 56 / 69 | 88.0 | 51.6 | 57.9 | 11.2 | 4.2 | 10.0, 452 | 4.3, 524 |
| 6 | 29 / 31 / 26 | 126.4 | 61.2 | 72.4 | 15.9 | 7.0 | 12.0, 579 | 6.0, 731 |
| 7 | 7 / 7 / 7 | 120.9 | 64.5 | 76.5 | 13.4 | 6.1 | 13.4, 588 | 6.9, 783 |
| 8 | 15 / 20 / 18 | 133.2 | 69.5 | 79.0 | 15.3 | 5.9 | 14.0, 635 | 6.7, 769 |
| 10 | 8 / 5 / 7 | 165.7 | 90.5 | 101.0 | 21.2 | 8.2 | 17.6, 876 | 8.9, 1015 |
| 12 | 6 / 4 / 6 | 243.2 | 105.3 | 118.9 | 22.1 | 7.2 | 21.5, 998 | 10.4, 1159 |
| 14 | 3 / 3 / 3 | 395.8 | 146.8 | 194.6 | 44.3 | 16.2 | 33.2, 1706 | 22.8, 2445 |
| 16 | 395 / 436 / 388 | 282.2 | 129.5 | 142.3 | 28.3 | 12.5 | 24.9, 1234 | 13.1, 1449 |

The exact path is ahead of the copies from 2 rows up and takes half their time at 16 rows. Its list is half the fp32
path's (the copy engine has the other half) and costs 111 us an expert at 16 rows against the fp32 kernels' 50. At 1
row it is 1.6 ms behind the copies (17 windows a run).

File B:

| rows | windows (none / fp32 / exact) | no CPU path | fp32 path | exact | fp32: cards wait | exact: cards wait | CPU's list, fp32: experts, µs | CPU's list, exact: experts, µs |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 17 / 17 / 17 | 25.8 | 25.8 | 27.5 | 6.3 | 3.4 | 3.4, 190 | 1.4, 238 |
| 2 | 877 / 663 / 609 | 44.5 | 33.1 | 36.1 | 6.2 | 3.2 | 4.3, 232 | 1.9, 300 |
| 3 | 487 / 681 / 635 | 59.9 | 39.5 | 42.9 | 9.1 | 3.3 | 6.1, 321 | 2.6, 375 |
| 4 | 268 / 311 / 348 | 81.7 | 48.0 | 52.9 | 12.4 | 3.9 | 8.4, 431 | 3.8, 490 |
| 5 | 87 / 54 / 88 | 118.1 | 61.1 | 68.9 | 16.8 | 4.0 | 12.1, 600 | 5.5, 666 |
| 6 | 27 / 37 / 28 | 180.0 | 71.0 | 89.4 | 20.6 | 5.9 | 13.5, 714 | 7.3, 892 |
| 7 | 7 / 8 / 10 | 177.7 | 77.1 | 85.8 | 21.7 | 4.4 | 15.9, 787 | 7.5, 875 |
| 8 | 15 / 14 / 17 | 175.1 | 79.9 | 96.0 | 22.0 | 4.1 | 15.7, 797 | 8.3, 961 |
| 10 | 11 / 5 / 8 | 248.6 | 118.7 | 111.1 | 35.1 | 4.4 | 23.6, 1310 | 9.3, 1096 |
| 14 | 3 / 5 / 6 | 513.7 | 156.6 | 192.2 | 43.7 | 3.5 | 33.4, 1785 | 17.7, 1952 |
| 16 | 398 / 397 / 428 | 407.3 | 152.2 | 177.1 | 44.1 | 4.4 | 29.1, 1594 | 15.6, 1748 |

The same picture: ahead of the copies from 2 rows up (36.1 against 44.5 ms), well under half their time at 16 rows (177
against 407 ms), 1.7 ms behind at 1 row. The CPU's list costs 112 µs an expert at 16 rows; the cards wait 4 ms of such a
window for the CPU's rows (44 with the fp32 path, whose list is twice as long).

The kernels alone, one expert half of file A (blocks sampled from the file, `cpu_exact_test real`), the box idle:

| | 1 row | 2 rows | 4 rows |
|---|---:|---:|---:|
| one core, before (the step with a test, four chains) | 1,355 us | 1,994 | 3,284 |
| one core, now | 1,173 | 1,620 | 2,539 |
| 16 threads, before | 199 | 262 | 431 |
| 16 threads, now | 116 | 166 | 257 |

Decoding an expert half to fp16 is about 715 us of the one-core time and did not change; the chains went from 643 to
455 us a row. With all 16 cores of a socket busy a core is about 1.4 times slower than alone.

## Tried and not kept

Five of these were built into the engine behind measuring switches and measured there, on an experimental build that is
not in this branch (`FNX_CPU_RND`, `FNX_CPU_TAB`, `FNX_CPU_DCACHE`, `FNX_CPU_DNCARD`, `FNX_CPU_RPULL`; the default
kernels are template
instantiations of their own and do not change with them). File A, the final settings, 24 Nov; "the default" is that
build's own default path, three rounds: 55.9 / 100.2 / 104.6 tok/s fresh and edits, 44.8 after the long read (the
row fetch's window: 56.1 / 100.2 / 104.5 / 45.1). Every such run's first-token logits were the card-only build's, and
each variant's `FNX_CPU_CHECK=1` run found 0 values in some 780,000 rows a card.

| What | Measured | Why not |
|---|---|---|
| Rounding to fp16 inside the fp32 register: to nearest even at bit 13 with integer operations (5 uops) in place of `vcvtps2ph` / `vcvtph2ps` (4), a run in which a lane's sum was zero or under 2^-14 done again with the round trip | **in the engine, one run: 27.0 / 33.7 / 34.9 / 21.4** (-52 % to -67 %). Kernel bench, one core, an expert half, 1 and 4 rows: 2,200 and 6,469 us against 1,293 and 2,773; the bare rounding without the tracking (not exact) 1,388 and 3,271 | below 2^-14 fp16 rounds at a place that moves with the exponent, which the fixed bit cannot do; 93-94 % of the runs (16 rows x 32 partial sums) have such a sum or a zero, so nearly every run is done twice. And the bare rounding is already 7-17 % slower than the round trip: one uop more in a step that the core's window bounds |
| Per-row tables: a row's 16 possible weights once, looked up with `pshufb`, the two byte planes transposed | **in the engine, three rounds: 55.7 / 99.5 / 103.7 / 44.3** (-0.4 % to -1.0 %). Kernel bench: +4 to +6 % an expert half on one core; the decoder alone 153-154 against 120-126 cycles for 8 rows x 32 values | the lanes are rows, so each lane has its own table; the planes' way out alone costs 97-102 cycles |
| A cache of decoded weights (fp16 lanes, 4.7 MiB an expert half; a cache a card, the clock's second chance) | **in the engine, 4 GiB a card (873 expert halves), three rounds: 48.6 / 71.6 / 74.8 / 38.5** (-13 % to -28 %); 20.9 % of the run's 486,680 listed experts were found in it. Kernel bench: the same expert again (hot) 0.58 of the default; the pass that writes an entry +13 % on one core, +17-24 % on 16 threads | four fifths of the listed experts are not in it and pay for the write; a found one is read cold from RAM, 4.7 MiB where the decode reads 1.4 MB of blocks |
| Only the down weights copied, the card doing SwiGLU and down (`FNX_CPU_DNCARD=1`: the CPU computes gate\|up only; each listed expert's down weights go to a place of their own in VRAM, 70 MiB a card, on a stream of their own; after its SwiGLU the card computes the down rows with its own row kernel) | **in the engine, three rounds: 52.1 / 83.7 / 87.6 / 41.5** against that window's default 55.9 / 100.5 / 104.9 / 44.6 (-6.7 %, -16.7 %, -16.4 %, -7.1 %); one run each at shares 70,90: 52.3 / 89.7 / 94.9 / 41.2, and at 100,100: 46.6 / 78.9 / 86.4 / 33.9. Its check: 0 values in 778,909 rows a card. A probe had bounded it before it was built (the down weights copied as well, the two cards' lists paired a layer, the card's own work counted as nothing): at most +1.0 % on fresh text and -11 % on the edits at 70,75, less at four other shares; the real thing came in under that | the down weights' copies load the link that the path is there to relieve (they alone cost 7 % of fresh writing and 16 % of the edits). Not measured with both cards on x16 |
| The CPU's finished rows fetched by the card from mapped host memory (one block, as the gate\|up sums are fetched) in place of their copy on the card's copy queue | **in the engine, three rounds: 55.8 / 100.2 / 103.6 / 44.5** against 56.1 / 100.2 / 104.5 / 45.1 (-0.5 %, 0, -0.9 %, -1.3 %). `FNX_CPU_ROWPROBE=1`: the rows are on the card 43-55 us after they are computed in windows of 1-4 rows and about 100 us at 16 rows (x8 card), the callback's own delay included | there is little to take away, and the fetch costs as much |
| Eight chains with the first version's step | 391 against 361 cycles a run and row | that step is bound by its uops, not its length |
| A cheap look at each step and an exact check of the steps that need it | 386 against 361 cycles; 1.8 steps a run have a lane on a midpoint | too many branches |
| Sixteen chains (four runs at once) | 256-346 against 242-317 cycles for eight (a busy box) | no better; the core's window holds about seven steps |
| Comparing `d` with `-(w x)` bit for bit instead of the second FMA | within 1-2 % | one uop more for two cycles less; the 9-uop form is kept |
| `FNX_CPU_WARM=1` (the pool's waiting threads run 256-bit FMAs) | kernel bench: a list after 3 ms idle 157 -> 125 us; in the engine +0.5 %, +0.8 %, -0.6 %, +0.2 % on the four tests, and within 1 % in each of the tuner's three runs | no gain in the engine; our reading (not measured): with the split nearly every layer has a list, so the pool does not idle that long |
| `FNX_CPU_SHARE` with the fp32 path (no `FNX_CPU_EXACT`) | one run each, file A, tok/s fresh and edits: 58.9 / 108.7 / 112.5 with `70,75` and 32 threads against 58.6 / 109.5 / 118.3 without | no gain at the exact path's shares; a share nearer 100 was not searched |
| `FNX_PREFETCH_WANT=1`, `FNX_CPU_MIN_T=3` beside the split | one guess a row: within 1.5 % on every test; no CPU below 3 rows: -1.5 % to +0.2 % (the mean of the tests) in the tuner's three runs | under 3 % on every test |

## Left

- **A share for each card.** Both cards get the same share, because the two halves of a layer keep the same slot map.
  On a machine like ours the card in the x8 slot would rather compute more and the one on x16 copy more (a copy of an
  expert half costs 236 us on the first and 122 on the second; the CPU's time is the same for both), and a layer waits
  for the slower card. Estimated from the by-rows report, not built: about an eighth off a 16-row window's time for its
  misses. It needs the cards' slot maps to differ, or the faster link's copies to be ordered ahead.
- **Both cards on x16 with fast CPUs.** Not measured. The maintainer's box has both cards on x16 and slow CPUs, and
  there the path loses at every share tried ("Verified"). With fast CPUs the copies are twice as fast on the card that
  limits us, so the CPU's best share is smaller; `tune.py` finds it. The one rejected idea that may pay there: only the
  down weights copied and the card doing SwiGLU and down (built and measured here, where it loses; on our x16 card the
  down weights were there 988 us after a 16-row list arrived, the CPU's whole list after 1,542).
- **The decode** is now the larger part of a list of few rows (715 of 1,173 us an expert half for one row on one core).
  Two ways to make it cheaper were built into the engine and lost (tables, a cache: the table above); none is known
  that keeps the bits and wins.
- **AVX-512**: twice the lanes a step. No machine here; the kernels are AVX2/FMA/F16C only.
- **Prompts** (reading) do not use the CPU path: it is for decode windows.
- The measuring switches (`FNX_CPU_GCOPY`, `FNX_CPU_POST_ROWS`, `FNX_CPU_HB_DIRECT`, `FNX_CPU_DNPROBE`, `FNX_CPU_REUSE`)
  can go; `FNX_CPU_WARM` gained nothing in the engine here and is off.
- Other weight types, and `FNX_ROWS_G2`.
