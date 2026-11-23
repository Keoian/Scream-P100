# `FNX_CPU_EXACT`: the CPU's missing experts, bit for bit the card's

Branch `fnx-cpu-experts-exact`, on top of `fnx-cpu-experts` (PR #7). `FNX_CPU_EXPERTS=1 FNX_CPU_EXACT=1` computes the routed
experts that are not in VRAM on the CPU with the card's own arithmetic, so the engine's output is the card-only build's,
byte for byte, whatever is in VRAM at a window. Off by default; without the switch the CPU path is PR #7's fp32 one.
It costs speed (below): use it with `FNX_CPU_MIN_T=12` on a server like the one it was measured on.

Two Claude sessions wrote it on 23 Nov 2026: the arithmetic and its CPU tests on an i7-7700 without a card (the first
commit), the run on two Tesla P100 with what that found and changed (the commits after it; a 2 x E5-2696 v3 server, one
card in a x8 slot). This file is the record of both: the spec, what is verified and how, what it costs, what is left.

## Why

With `FNX_CPU_EXPERTS=1` the CPU sums in fp32 and the cards' row kernels in fp16, so the CPU's rows are close to the cards'
but not the same bits, and which experts are in VRAM at a window depends on timing: a run with the switch differs from a
run without it, and from the next run with it. The maintainer's profile is bit-exact and stays so. With `FNX_CPU_EXACT=1`
the CPU computes exactly what the card would have.

## What the card computes (the spec the CPU now follows)

From `p100_moe_rows.cuh` (`k_moe_rows`, `row_run`) and `p100_iq_gemm.cuh` (`kq_finish`, `finish_s`, `fold_one`):

- weights, rounded to fp16 once: Q4_K/Q5_K `fp16(q * a + b)` with `a = fp16(d * sc)`, `b = fp16(-dmin * m)` (one HFMA2);
  IQ4_NL `fp16(kvalue * d)`, Q8_0 `fp16(q * d)` (HMUL2); Q5_1 `fp16(q * d + m)` (HFMA2);
- each 32-value run: two fp16 FMA chains (lane .x the even values 0, 2, ..., 30, lane .y the odd ones), 16 steps each, from +0,
  then into fp32 as `s = s + (lo + hi)`;
- gate|up rows: `FNX_ROW_G` lanes a row (2 by default), part p the 256-value groups `[p*ngr/G, (p+1)*ngr/G)`, the parts added by
  the xor shuffles (lane 0: `((s0 + s1) + (s2 + s3)) + ...`); down rows one lane;
- SwiGLU: `__float2half(silu_over(g) * u)` with `silu_over(x) = x / (1 + __expf(-x))` under `--use_fast_math` (MUFU.EX2 and an
  approximate division: not reproducible on a CPU).

## What the branch adds

| File | What |
|---|---|
| `engine/src/cpu_experts.cpp/.hpp` | `Work::exact`, `Work::gu_lanes`, `Expert::gu_out`; an oracle doing IEEE binary16 on the integers (`f16_fma_ref`, `f16_from_f32_ref`) and the card's row sums with it (`gu_row_card_ref`, `dn_row_card_ref`); AVX2 kernels with the card's arithmetic (`gu16_exact`, `dn16_exact`: 16 rows at a time, a row a lane); `Pool::run_gu` / `run_down` |
| `engine/src/p100_cpu.cuh` | `FNX_CPU_EXACT=1`: the host thread computes gate\|up, copies the sums to the card and writes `CpuDev::gflag`; on the card's third stream `k_cpu_wait_g`, `k_cpu_swiglu` (h with the very expression of `k_swiglu_after`, into VRAM) and `k_cpu_hpost` (one block: h to mapped host memory with the sum of its words, then the list's number); the host thread checks the sum, reads h and computes down. The mailbox's post (`k_cpu_post`) is one block with a sum too. `FNX_CPU_CHECK` counts the values whose bits differ (`CpuDev::bitdiff`) and, with the exact switch, the stage: the CPU's gate\|up sums and the SwiGLU made of them against the card's own (`k_cpu_check_gh`) |
| `engine/src/engine.cu` | `p100_cpu_swiglu` after `p100_cpu_post` on `st3`; under `FNX_CPU_CHECK` a fork of `st` to `st3` after the publish, joined before the take, then `p100_cpu_check_stages` |
| `engine/tools/cpu_exact_test.cpp` | CPU only: the oracle against F16C, the fp16 FMA against the oracle, the chains' fast form (every lane it does not flag is the oracle's bits), every kernel against the card's steps (direct and through the pool), and a `bench` mode |
| `engine/tools/cpu_exact_gpu_test.cu` | on a P100: HFMA2 / HMUL2 / `__float2half_rn` against the oracle; random experts of every covered type, 1-16 tokens: the card's row kernels against the CPU's, the three SwiGLU kernels against each other, down of the card's h |
| `engine/CMakeLists.txt`, `CMakeLists.txt` | targets `split_cpu_exact_test` and `split_cpu_exact_gpu_test` (`-DSTRATA_BUILD_TESTS=ON`); the GPU test gets `p100_compat.h` like the engine |
| `docs/SWITCHES.md`, `docs/GUIDE.md` | the switch's rows and a paragraph |

Covered types: gate|up Q4_K or Q5_K, down IQ4_NL, Q5_1 or Q8_0 (every layer), `FNX_ROW_G` 1/2/4/8; otherwise the engine says
why and keeps the fp32 path. Requires `FNX_ROWS_G2` unset, as the profiles have it (with it set, phase 2 sums in another
order and the card-only run itself depends on what is in VRAM).

## How the CPU gets the card's bits

- A product of two fp16 values has at most 22 significant bits: exact in fp32.
- `fma16` (exact form): the fp32 sum `s`, its exact error `e` (TwoSum), `s` rounded to odd where `e != 0` (one ulp toward the
  exact sum when its last bit is 0), then to fp16 (F16C). Rounding to odd with 24 >= 11 + 2 bits and then to nearest is the
  correctly rounded result (Boldo-Melquiond). The subtractions are written as FMAs with 1.0 (Haswell adds on one port only).
- `fma16f` (the chains' fast form): one fp32 rounding, then fp16; wrong only where the fp32 sum was inexact and landed on an
  fp16 midpoint. Every fp16 midpoint (normal, subnormal, 65520) has at most 12 significant bits, so its float's low 12 bits are
  0; inexactness is `s - p != c || s - c != p` (for the larger operand that subtraction is Fast2Sum's, exact). A run whose
  lanes flag anything is redone with `fma16` (measured: exact midpoints are 97 % of midpoints and must not trigger the redo).
- Decode: `q a + b` with q < 32 is exact in fp32 when a and b are normal fp16 values with exponents at most 7 apart (or either
  is 0): then one rounding of the FMA; otherwise the fast form with its test, else `fma16`.
- The xor-shuffle order of the lanes, the run order, `s + (lo + hi)`: as listed above. No `-ffast-math` anywhere; every op is an
  IEEE op, so the results are the same on any x86 with AVX2/FMA/F16C (run on Kaby Lake and on Haswell).


## Verified

On the i7-7700 (no card) and again on the server (Haswell), `split_cpu_exact_test 8`: the oracle's known answers
(including the double-rounding case 1.5 x 0.6669921875 + 2^-24 = 1.0009765625) and its rounding against F16C on 8 M floats
and around every half; `fma16` against the oracle on 8.5 M triples (191 of them where fp32-then-fp16 is wrong); the
chains' fast form on as many (every lane it does not flag is the oracle's bits; 169 flagged lanes where it would have been
wrong); 41,472 row sums of every covered type with 1, 2, 4 lanes and 1 or 3 tokens bit for bit the card's steps, directly
and through the pool. Mutation checks on the i7 (temporary copies): the naive fp32-then-fp16 FMA, the fast form without
its redo, and the plain decode everywhere each make the kernel test fail.

On the two P100 (server, 23 Nov 2026; sm_60, CUDA 12.8):

- `split_cpu_exact_gpu_test` on each card: 8,388,608 x 2 HFMA2 lanes (every half as a first operand), as many HMUL2,
  8,388,608 float -> half: 0 / 0 / 0 differ from the oracle; row kernels gate|up 276,480 values, the three SwiGLU kernels
  138,240, down 1,105,920: 0 not the card's.
- The engine with `FNX_CPU_CHECK=1` (the copy engine and phase 2 run as without the switch; the CPU's rows are compared):
  on a Q4_K_XL file with IQ4_NL down rows 1,568,584 rows a card, on Unsloth's UD-Q4_K_XL (Q4_K gate|up, Q5_1 down)
  1,078,758 rows a card: 0 values not bit for bit, on both cards, and by stage 0 gate|up sums and 0 SwiGLU values. Every
  post whole by its sum.
- Without the check (the CPU's rows are the ones used): the decode-path yardstick (4,995 positions, teacher-forced,
  windows of 16 rows) is the card-only run's file byte for byte, twice; the card-only run itself repeats byte for byte
  (three runs over two days). The first-token logits of `bench/bench.py`'s 27 requests are the card-only build's byte
  for byte in four runs of four.
- HumanEval (164 problems, greedy): all 164 answers identical in text to #5's build's (158 passed in both; 37,295 completion tokens in both); median writing 43.9 tok/s against 43.0

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

## Cost

Server: two Tesla P100 16 GB capped at 150 W (card 0 on PCIe 3.0 x8), one card per socket of 2 x Xeon E5-2696 v3, 16
pool threads a card. Our file (Q4_K gate|up, IQ4_NL down; 7,778 of 24,576 experts in VRAM at 262,144 positions), three
rounds in a Latin square, every run a fresh load, `bench/bench.py`, tok/s, mean (min-max):

| | #5's build (no CPU path) | #7's fp32 path (with the pool's fix) | `FNX_CPU_EXACT=1` |
|---|---:|---:|---:|
| fresh writing | 47.5 (47.2-47.6) | 60.0 (59.6-60.5) | 32.0 (31.5-32.8) |
| edit py | 53.9 (53.9-54.0) | 112.0 (111.4-112.7) | 52.3 (51.5-53.0) |
| edit md | 54.8 (54.7-54.9) | 118.9 (118.5-119.2) | 58.8 (58.1-59.5) |
| a cold 38,862-token prompt: reading | 889 (888-889) | 886 (885-886) | 887 (886-887) |
| writing after it | 43.0 (42.9-43.2) | 46.9 (45.9-48.1) | 23.0 (22.7-23.2) |

So on this server the exact path is slower than no CPU path at all on fresh text and equal on edits. Where the time
goes (`FNX_PROF=1`, ms a window by the window's rows; the CPU's list is a layer's missing experts on one card):

| rows | no CPU path | fp32 path | exact | fp32: the cards wait | exact: the cards wait | CPU's list, fp32: experts, µs | CPU's list, exact: experts, µs |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 23.6 | 22.7 | 38.1 | 4.3 | 19.3 | 2.6, 148 | 2.7, 518 |
| 2 | 35.6 | 29.8 | 53.8 | 4.6 | 27.1 | 3.8, 188 | 3.8, 691 |
| 3 | 46.4 | 34.4 | 65.7 | 6.1 | 35.7 | 5.1, 247 | 5.1, 899 |
| 4 | 58.9 | 41.0 | 83.9 | 8.3 | 49.5 | 7.1, 326 | 7.2, 1207 |
| 6 | 126.9 | 61.1 | 132.7 | 15.6 | 86.0 | 11.9, 566 | 12.2, 2022 |
| 8 | 132.7 | 68.8 | 147.9 | 14.7 | 94.1 | 13.9, 624 | 14.3, 2300 |
| 10 | 164.8 | 91.1 | 180.0 | 21.8 | 109.6 | 18.0, 891 | 16.6, 2753 |
| 12 | 242.5 | 99.9 | 212.5 | 17.8 | 131.8 | 19.9, 888 | 20.2, 3288 |
| 14 | 395.3 | 144.7 | 334.3 | 42.6 | 227.7 | 32.8, 1637 | 33.4, 5542 |
| 16 | 282.9 | 128.1 | 275.3 | 26.9 | 174.8 | 24.8, 1205 | 25.2, 4285 |

(The first column is this branch with both switches off, the second #7's build with the pool's fix, both measured the same day; 12 and 14 rows are 3 to 8 windows a run.)

The exact sums take the CPU 3.7 times the fp32 sums' time for a list, and the cards wait for it (27 ms of a 2-row
window, 175 ms of a 16-row one); the round trip for the SwiGLU is 30-53 us a list and is not the cost. By rows the
exact path is slower than the copies below 12 rows and a little faster from there. `FNX_CPU_MIN_T=12` beside the switch
uses the exact path from 12 rows up and the copy engine below, with the same bits whatever the limit: one run: fresh 47.0 tok/s (47.5 without the CPU path), edits 66.6 and 68.4 (53.9 and 54.8: +23 % and +25 %), reading 886, writing after the 38.9K read 42.5 (43.0), and the 27 first-token logits byte for byte the card-only build's. The small windows' copies keep the cache following the text, so the 16-row windows' lists are shorter than when the CPU takes every window.

One core, one expert half of our file (blocks sampled from the file): about 1.4 ms for one token and 0.64 ms for each
further token of the same expert (measured while another run used the box, so approximate). Decoding a run's weights
to fp16 costs more than one token's chains, and the chains run at about 29 cycles a step: the dependency path through
each step's rounding to fp16, not the ports' limit.

## Left

- Speed. The decode of a run's weights costs more than the chains of one token, and the kernel is not bound by its ports;
  nothing was gained by moving the test's compares to integers. Not tried: the weights decoded in row order with byte
  shuffles (16 values a block from a 16-entry table) and transposed as floats; a cache of hot experts' decoded weights; the
  misses of a small window split between the copy engine and the CPU (any split is bit-exact).
- `FNX_CPU_POST_ROWS` and `FNX_CPU_HB_DIRECT` are measuring switches for the first version's posts; they can go.
- Other weight types, and `FNX_ROWS_G2`.
