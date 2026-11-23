# Hand-off: `FNX_CPU_EXACT`, the CPU's missing experts bit for bit the card's (prototype)

Branch `fnx-cpu-experts-exact`, on top of `fnx-cpu-experts` (PR #7, ef0e74d). Written 23 Nov 2026 by a Claude session working
on another box (an i7-7700, no card free for it there); for the Claude that works on this repository
on the E5-2696 v3 server. Nothing here has run on a card yet: read "Not verified" before relying on any of it.

## Why

The owner: with `FNX_CPU_EXPERTS=1` some benchmark tests fail reliably against the GPU-only run. The CPU path sums in fp32, the
cards' row kernels in fp16, so the CPU's rows are close but not the same bits, and which experts are in VRAM at a window
depends on timing, so the output changes with residency. The goal: the CPU computes exactly what the card would have.

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
| `engine/src/p100_cpu.cuh` | `FNX_CPU_EXACT=1`: the host thread computes gate|up, copies the sums to the card and writes `CpuDev::gflag`; `k_cpu_wait_g` + `k_cpu_swiglu` on the card's third stream compute h with the same expression as `k_swiglu_after` into mapped host memory (`hb`: the list's number, then f16 rows); the host thread reads h and computes down; `FNX_CPU_CHECK` now also counts values whose bits differ (`CpuDev::bitdiff`, in the report line); `p100_cpu_swiglu_probe` for the test |
| `engine/src/engine.cu` | `p100_cpu_swiglu` after `p100_cpu_post` on `st3`; under `FNX_CPU_CHECK` a fork of `st` to `st3` after the publish, joined before the take |
| `engine/tools/cpu_exact_test.cpp` | CPU only: the oracle against F16C, the fp16 FMA against the oracle, every kernel against the card's steps (direct and through the pool), and a `bench` mode |
| `engine/tools/cpu_exact_gpu_test.cu` | on a P100: HFMA2/HMUL2/`__float2half_rn` against the oracle; random experts of every covered type, 1-16 tokens: the card's row kernels against the CPU's, the three SwiGLU kernels against each other, down of the card's h |
| `engine/CMakeLists.txt`, `CMakeLists.txt` | targets `split_cpu_exact_test` and `split_cpu_exact_gpu_test` (`-DSTRATA_BUILD_TESTS=ON`); the GPU test gets `p100_compat.h` like the engine |
| `docs/SWITCHES.md`, `docs/GUIDE.md` | the switch's row and a paragraph |

Off by default: without `FNX_CPU_EXACT` the CPU path is PR #7's, unchanged (`split_cpu_experts_test` still passes).
Covered types: gate|up Q4_K or Q5_K, down IQ4_NL, Q5_1 or Q8_0 (every layer), `FNX_ROW_G` 1/2/4/8; otherwise the engine says why
and keeps the fp32 path. Requires `FNX_ROWS_G2` unset, as the profiles have it (with it set, phase 2 sums in another order and
the GPU-only run itself depends on residency).

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
  IEEE op, so the results are the same on any x86 with AVX2/FMA/F16C (Kaby Lake here, Haswell there).

## Verified (here, i7-7700, no GPU)

- `split_cpu_exact_test 4`: oracle known answers (including the double-rounding case 1.5 x 0.6669921875 + 2^-24 = 1.0009765625),
  its rounding against F16C on 4 M floats and around every half; `fma16` against the oracle on 4.5 M triples (113 of them where
  fp32-then-fp16 is wrong); 41,472 row sums of every covered type with 1, 2, 4 lanes and 1 or 3 tokens bit for bit the card's
  steps, directly and through the pool.
- Mutation checks (temporary copies, not committed): the naive fp32-then-fp16 FMA, the fast form without its redo, and the plain
  decode everywhere each make the kernel test fail (1 to 16 of 41,472 row sums).
- `split_cpu_experts_test` (PR #7's) passes; `strata-split-2` builds with the changes.
- `split_cpu_exact_gpu_test`: its own file compiles; the whole target was not built here. The first try failed in the engine's
  kernels (`__dp4a`, `__nanosleep` undefined) because the target lacked the force-included `p100_compat.h`; the top-level
  CMakeLists.txt now adds it, as for the engine, but that build was not repeated.

## Not verified

- Nothing on a card: the claim that the card's HFMA2, its row kernels and its three SwiGLU kernels match these steps bit for
  bit rests on reading the code. Run `split_cpu_exact_gpu_test [DEV]` first (it needs about 100 MB of VRAM and seconds).
- The engine with the switch: never run (no model file of these types on the box). The handoff's ordering was checked by
  reading: the CPU reads h only after the card's last swiglu block wrote the list's number; the card's next swiglu waits for the
  CPU's next gflag; graph capture sees the st3 fork joined.
- Speed on the server.

## Cost

`split_cpu_exact_test bench NTOK THREADS REPS`, one expert half (640 x 2560 Q4_K gate|up, 2560 x 320 IQ4_NL down), random
blocks, weights in cache, i7-7700 at up to 4.2 GHz:

| | fp32 (PR #7) | exact |
|---|---:|---:|
| 1 token, 1 thread | 0.32 ms | 1.55 ms |
| 2 tokens, 1 thread | 0.46 ms | 2.28 ms |
| 1 token, 3 threads | 0.14 ms | 0.62 ms |

Roughly 1.2 cycles a weight a token for the chains and the rest for the decode (random scales send many decodes to the slower
forms; a real Q4_K file's scales are closer together, so expect less). On the server the fp32 path was bound by memory
(153 us a list of 2.6 experts at 1 row); the exact path will be bound by compute, plus a round trip to the card a layer. A guess:
about 3 x the CPU's time a list, which may cost the 1-row windows most of their gain while the 8-row edit windows keep part of
theirs. Measure before deciding anything.

## Next steps, in order

1. `cmake -DSTRATA_BUILD_TESTS=ON ...` (as `build.sh`), build `split_cpu_exact_test split_cpu_exact_gpu_test`; run both. If the
   GPU test reports a difference, its first lines name the stage (ops, gate|up, SwiGLU, down) and the value.
2. The engine with `FNX_CPU_EXPERTS=1 FNX_CPU_EXACT=1 FNX_CPU_CHECK=1`: the check's report line should say
   `0 values not bit for bit the card's`.
3. The failing benchmark tests with `FNX_CPU_EXPERTS=1 FNX_CPU_EXACT=1` against GPU-only.
4. Speed: `bench/bench.py` and `FNX_PROF=1` as in GUIDE.md's table, the three arms (switch off, fp32 CPU path, exact).
5. If it is too slow: the chains are about 15 vector ops a step against one FMA; ideas not tried: more rows in flight per
   pass (latency), skipping the inexact test where both operands are provably close, caching a hot expert's decoded weights.
