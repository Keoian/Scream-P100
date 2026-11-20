# engine-m40: the 2 x Tesla M40 24 GB test build's sources

> **TEST BUILD.** Built and measured on one machine only: our second server, two Tesla M40 **24 GB** (Maxwell, sm_52),
> which we have since sold. This tree is the source of the engine that ran there, put together again from our files (below)
> and **not compiled again since**. We cannot measure it any more. See [docs/M40.md](../docs/M40.md).

`bash build.sh --m40` builds this folder as it is, for sm_52, into `build-m40/split/strata-split-2`. `engine/` and the P100
builds are not touched.

## Why a tree of its own

The M40 engine was made on 6 Nov 2026 from **the P100 tree of that afternoon**, eleven days before `engine/`. The P100
port has changed a great deal since then, so the M40 changes are not a patch on `engine/`: this folder is the whole tree.

| Here | What it is |
|---|---|
| `src/` | the engine's sources: the 6 Nov P100 tree with `patches/1-*` and `patches/2-*` applied |
| `p100_compat.h` | the P100's compatibility header of 6 Nov plus `__hfma2` for sm < 53 (Maxwell has no f16 arithmetic) |
| `CMakeLists.txt`, `tools/` | the split engine's own, unchanged (byte for byte the same as `engine/`'s) |
| `patches/1-m40-20261106.patch` | 6 Nov: the Maxwell kernels and the M40's memory plan (diff against the P100 tree, 13 files) |
| `patches/2-m40-20261115.patch` | 15 Nov: the P100's prompt-path changes of 10-13 Nov brought over, a new prompt expert kernel, cuBLAS algorithm maps (8 files) |
| `tests/` | the two kernel checks we ran on the M40 (`build_test.sh`) |
| `SHA256SUMS` | `src/`, `p100_compat.h` and `CMakeLists.txt` as published here |

## What the M40 changes do

- **Writing:** `src/m40_moe_rows.cuh`, a row kernel in f32 for the M40 model's expert types (IQ2_XXS, IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S, IQ4_NL, Strata's Q2_0 and Q4_0): the P100's row kernel is built around paired f16 FMA (HFMA2), which
  Maxwell does not have.
- **Reading prompts:** `src/m40_pf_gemm.cuh` (`FNX_M40_PF=1`): each thread decodes its rows straight into f32 weights and
  sums in f32; 1.5-2x the P100's f16-tile kernel on the M40. The P100 kernel (`k_gemm4_job`) is still there when it is off.
- **New types:** IQ2_XXS and IQ2_XS in the prompt kernels and the type checks; Q2_0 kept as Q2_0 (`FNX_Q2_0_NATIVE`, on below
  sm_60) instead of being widened to Q4_0 at load, which would not fit in 32 GB of RAM; IQ3_S in `tensor_convert.hpp`.
- **Memory on a 32 GB machine:** `FNX_EXPERT_DYN_GIB` (only the coldest experts on the cards keep a copy in RAM; the rest are
  built straight in VRAM and never move) and `FNX_PLE_RAM=0` (the 26.8 GiB lookup table stays in the file and is read
  through the page cache).
- **cuBLAS on Maxwell:** `FNX_M40_HC_ALGO` and `FNX_M40_DENSE_ALGO` pick explicit algorithms measured on our M40
  (`FNX_PF_SP` needs an algorithm whose bits on half a chunk equal the whole chunk's, and cuBLAS's default choice differs on
  Maxwell).
- The Maxwell paths are behind `FNX_ARCH < 53` (Q2_0 kept native: `< 60`) or a switch. This tree has only been built for
  sm_52; for the P100 use `engine/`.

## How this tree was put together (20 Nov 2026)

The server is gone, so `src/` was rebuilt from what we kept:

1. **The base:** the sources frozen beside our P100 production build `a2b6f771` (6 Nov), plus the four scripts run on the
   P100 tree that afternoon before the M40 copy was taken (`onehop_patch.py` 14:36, `rr_var_patch.py`, `rr_var_patch2.py`,
   `rr_final_patch.py` 15:16). Check: this base with the next P100 step (`cvec_patch.py`, 7 Nov) gives, file for file, the
   sources frozen beside the next production build (`cvec`).
2. **`patches/1-m40-20261106.patch`** applied with no fuzz and no offset (the M40 engine of 6 Nov, `strata-split-2`
   `02cae8822df696ba...`).
3. **`patches/2-m40-20261115.patch`** applied with no fuzz and no offset: the M40 production engine from 15 Nov until the
   sale, `strata-split-2` SHA-256 `d21a22c825473d11...`. This patch is the diff our switch-over script wrote beside that
   binary (`diff -ruN` of the tree before and after 15 Nov's work).

Both patches are kept in `patches/` with their original paths (`varne/p100-izvor/`, `varne-20261115/src/` against
`upstream/src/split/src/`). The P100 tree they started from is not in this repository; step 1 is our record of it.

## Building

```bash
bash build.sh --m40          # CUDA 12.x (12.4 here), gcc 13, sm_52, prompt-path jobs of 96 rows
```

The binary must not call any CUDA peer-access function (the cards exchange their sums through pinned host memory), and
`build.sh` checks that as for the P100 build.

## Credits

The engine is [Strata](https://github.com/Niko1221/Strata)'s split engine (`src/split`), which Strata ported from John Fink's
[sFNX](https://github.com/johnfink8/sfnx); the block formats are ggml's / llama.cpp's. Our P100 port and the Maxwell changes
on top of it are ours (MIT, as Strata). See [NOTICE.md](../NOTICE.md).
