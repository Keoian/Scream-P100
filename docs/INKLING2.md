# Scream P100 · Inkling 2: an engine of its own for Inkling 2 Mini on two Tesla P100

> ⚠️ Built and measured on **one machine** (2× Tesla P100 16 GB, each on its own PCIe 3.0 x16 link, 99 GiB RAM), written on
> 21 Nov 2026. Nobody else has run it yet. Reports are welcome.

Thinking Machines' **Inkling 2 Mini** (26.4B MoE: 24 layers, layers 6-17 run three times for 48 passes, 128 experts of which 8
per token, a 1,024-token window on most layers, one full-attention layer in six, NVFP4 experts) on the same two P100s as
Scream P100's Flash-Next, with an **engine of its own**, `ink2` (`engine-inkling2/`): Strata's split engine does not know the
looped layers, the depth mixing, the loop controllers and messages, the short convolutions, the relative position bias with
a sink logit, or the DSpark drafter, so we wrote one with the same idea. **Both cards work on every layer at once**, and
here the **whole model sits in VRAM** (8.9 GB a card): no RAM tier, no PCIe streaming of experts.

## Speed

Same two P100 16 GB, temperature 0, everything exact (f32 sums, only their order differs from llama.cpp's).

| | llama.cpp `690b87d` (same cards, layer split) | **ink2** |
|---|---|---|
| Fresh writing, no drafts | 13.0 tok/s | **56.6** |
| Fresh writing, DSpark (MTP) 3 + prompt lookup | — | **78.8** (code, 256 tokens) |
| Edits (the model repeats a file with a change) | — | **100.7** (700 tokens) |
| Reading a 4,096 / 14,336 / 30,000-token prompt | 143 / 119 / — | **794 / 757 / 709** |
| VRAM a card | 14.0-14.5 GB | 8.9 GB writing only, 11.8 GB with the drafter at 32k |
| Context | 32k measured | up to **262,144** (15.2 GiB a card with the drafter) |

On the **test set of our Flash-Next builds** (the same prompts, through the OpenAI API, `INK2` with its defaults), against
Scream P100's Flash-Next UD-Q4_K_XL on the same machine:

| Test | **Inkling 2 Mini, ink2 (exact)** | Flash-Next Q4_K_XL, exact | Flash-Next Q4_K_XL, fast (lossy) |
|---|---|---|---|
| Fresh writing (12 prompts, thinking low) | 66.3 | 59.3 | 68.9 |
| Edits, Python / Markdown | 114.3 / **126.8** | 100.9 / 92.2 | 124.3 / 125.6 |
| Reading 36k / writing after it | 646 / 55.4 | 802 / 61.8 | 987 / 51.3 |

Inkling thinks before it answers even at effort 0; in the Python edit (a file without the variable it is asked to rename) it
reasons for over 1,600 tokens, so that number measures writing its reasoning, not copying. Long reads are bound by f32
arithmetic: the P100 has no tensor cores, and Flash-Next's fast numbers use f16 there.

**Long context:** a word hidden in the first line of a **239,114-token** prompt was found (read in 876 s, 273 tok/s on average,
writing after it at 42.2 tok/s with the drafter); also at 120k. The KV cache costs 16 KB a token on each card, so 1M tokens
(the model's limit) cannot fit in 2× 16 GB; about 262k with the drafter, about 290k without.

**Quality:** greedy outputs of 32 tokens on English, Slovenian and code prompts are the same as llama.cpp's; KL of the last
row 0.0001 / 0.025 / 0.00001; on a 4k prompt the first token is the same in 13 of the last 16 rows (in the other three the
top two swap places, KL 0.0000-0.040, 0.014 on average; llama.cpp quantizes its own inputs to q8_1). On the task sets of our
Flash-Next builds, through the server with its defaults: **44 of 45** functions with hidden tests (Flash-Next UD-Q4_K_XL 45),
written at 85.6 tok/s (73.6); English questions **16/16** (16); Slovenian questions 12/15, the same three missed by llama.cpp
on the same file (two authors of Slovenian classics
and one clock sum missed); 0 of 18 Klepet-style answers in Croatian or Serbian, 0 of 6 open Slovenian prompts with foreign
words. Inkling reasons before it answers even at the lowest effort, and once spent a whole 1,200-token answer on it.

## Run it

```bash
git clone -b inkling2 https://github.com/Megla123/Scream-P100.git && cd Scream-P100
bash engine-inkling2/build.sh                 # ink2_v1 (nvcc) + llama.cpp's llama-tokenize on the CPU
bash engine-inkling2/setup.sh /data/inkling2  # the model (17.6 GB), the DSpark drafter (0.4 GB), a venv
bash engine-inkling2/run.sh /data/inkling2    # OpenAI-compatible API on http://127.0.0.1:8080/v1
```

`run.sh DATA [CTX] [PORT]`: CTX up to 262144. `python3 engine-inkling2/tools/bench.py` measures fresh writing, an edit and a
long read through the API. Checked from a clean clone of this branch on our machine (21 Nov, CUDA 12.4, `HOST_CXX=g++-13`):
`build.sh`, `setup.sh`, `run.sh DATA 65536`, ready in 48 s, then `tools/bench.py`: fresh writing 70.8 tok/s, the edit 105.9, a
34,699-token read at 684.8 tok/s and writing after it at 55.8. If your default g++ is newer than CUDA 12.4 accepts, set
`HOST_CXX=g++-13`.

Files: Xylolabs's [`Inkling-2-Mini-NVFP4-Q8_0.gguf`](https://huggingface.co/Xylolabs/Inkling-2-Mini-GGUF) (the original
NVFP4 experts with a scale per expert, the last four layers Q8_0, the rest Q8_0 / F32) and Unsloth's
[`dspark-Inkling-2-Mini-Q8_0.gguf`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), the model's own MTP drafter. The
model is Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini); its license applies to
the weights.

The server (`serve/ink2_server.py`) answers `/v1/chat/completions` (streamed or not; the thinking part as
`reasoning_content`), `/v1/models`, `/health`, `/props`, `/slots`, `/tokenize` and `/apply-template`, so llama-server
clients work. `reasoning_effort` (none … xhigh) sets Inkling's "Thinking effort level". **Tool calls** work: OpenAI `tools` go in
as the template's `tool_declare`, the model's calls come back as `tool_calls` (streamed too), tool results as its tool messages;
the prompt equals the GGUF template's to the character (`tools/template_check.py`), and a web search tool in a chat app and a
coding agent's write / bash tools worked on our machine. Limits: **greedy only** (no sampler yet), no pictures, one request at a time. A follow-up turn of a conversation reads only the new part (the
engine keeps a snapshot of its state before the last prompt token).

## How it works

- **Split:** query heads 12 + 12 and KV heads 4 + 4 per card; every expert's FF split in half (gate and up rows, down
  columns); the output vocabulary split in half. Two sums per pass (after attention, after the experts) through pinned
  host memory, card 0's part first on both cards, so both hold the same bits. 96 exchanges per token. No NVLink, no P2P.
- **Writing path:** NVFP4 experts stored planar per expert (all codes, then all scales) and read by row kernels with the
  input in shared memory padded against bank conflicts (`k_fp4_rows7` / `k_fp4_rows6`); fused kernels for the exchange +
  post-convolution + residual, the K/V convolutions + clips + cache write, the router + job list; attention by KV head with
  the three query heads together.
- **Reading path:** chunks of 2,048 tokens; dense matrices as Q8 → f32 + cuBLAS; the experts sorted by expert and run by a
  tile GEMM that decodes NVFP4 in shared memory (~4.7 TFLOPS); window attention for all query blocks and heads in one
  batched call; full-attention layers with keys in 32k blocks and P·V split by 1,024-key pieces at long context.
- **Drafts:** the DSpark drafter (3 layers, a Markov head) on both cards, run with k + 1 block rows, a reduced 32k draft
  vocabulary (`data/draft_vocab_32768.bin`, `tools/draft_vocab.py`) and a probability gate (0.7, like Flash-Next's
  `FNX_DRAFT_MIN_P`); prompt lookup takes the longest earlier match of the last tokens (3-16) and wins over the drafter at
  6+. The target checks every draft in one window. A window's sums are not bit-identical to single tokens (KL 6·10⁻⁵ after
  ~2,000 tokens), so a greedy answer can part from the undrafted one at a near tie.
- **Model details we had to get right:** the relative position bias and the sink logit are **not** scaled by 1/√128 (the
  reference code adds them after the scaling; a llama.cpp graph line suggests otherwise, its CUDA output does not); the
  bias for distances ≥ 1,024 is zero; τ scaling of Q and R on the full layers past 8,192 tokens.

## Switches (`INK2_*`, environment of the engine)

Defaults in brackets. Presence switches (on when set at all): `INK2_STARE` (the first kernels), `INK2_FUSE_OFF`,
`INK2_ATTN_OLD`, `INK2_MT`, `INK2_GRAF`, `INK2_ROWS2`, `INK2_FP4_ILV`, `INK2_PF_OFF` (no reading path). Value switches:
`INK2_T1TT` (1), `INK2_WIN_SEG` (0), `INK2_PERM` (0), `INK2_FEED_T` (16), `INK2_PF_MIN` (17), `INK2_PF_CHUNK` (2048),
`INK2_PF_QSF` (1024, smaller on its own past 96k), `INK2_PF_SMB` (MB for the score matrix), `INK2_PF_ATTB` (1),
`INK2_PF_RB` (0), `INK2_PV_SPLIT` (1), `INK2_DS_T1` (1), `INK2_DS_ROWS` (1 = k + 1 rows, 0 = all 8), `INK2_DS_PMIN` (0.7),
`INK2_DS_RING` (on past 64k), `INK2_NG_MIN` (6), `INK2_SNAP` (1). Engine flags: `--model`, `--mtp`, `--draft-vocab`, `--ctx`,
`--serve SOCK`, or `--tokens FILE [--gen N] [--logits F] [--logits-all F] [--dump DIR]` for tests (`--logits-all`: every row of the last window).

## What did not help

A CUDA graph per token (no gain), reading an expert once per draft window (slower), 128×128 expert tiles on the reading path
(slower), a custom R·B kernel (same as cuBLAS), job order by expert in a window (same speed).
