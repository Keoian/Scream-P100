# Scream P100 · Inkling 2: an engine of its own for Inkling 2 Mini on two Tesla P100

> ⚠️ Built and measured on **one machine** (2× Tesla P100 16 GB, each on its own PCIe 3.0 x16 link, 99 GiB RAM), written on
> 21-22 Nov 2026. Nobody else has run it yet. Reports are welcome.

**Options:** a context of **256k** (the KV cache all in f16) or **512k** (the long part of the KV cache in int8, the first 32k
tokens exactly as at 256k, the same speed), and **pictures** (the model's vision projector on the CPU) in both.

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
| Context | 32k measured | **262,144** (KV f16) or **524,288** (KV int8 past 32k), 15.2 GiB a card with the drafter |
| Pictures | mmproj (on a card) | mmproj on the CPU, ~20 s for a 1600×900 picture (920 rows), then read at ~640 tok/s |

On the **test set of our Flash-Next builds** (the same prompts, through the OpenAI API, `INK2` with its defaults), against
Scream P100's Flash-Next UD-Q4_K_XL on the same machine (22 Nov, at the context each option runs with):

| Test | **ink2, 256k (exact)** | **ink2, 512k** | Flash-Next Q4_K_XL, exact | Flash-Next Q4_K_XL, fast (lossy) |
|---|---|---|---|---|
| Fresh writing (12 prompts, thinking low) | 67.4 | 67.2 | 59.3 | 68.9 |
| Edits, Python / Markdown | 115.5 / **127.8** | 115.5 / **127.8** ¹ | 100.9 / 92.2 | 124.3 / 125.6 |
| Reading 37.6k / writing after it | 578 / 57.2 | 581 / 54.2 | 802 / 61.8 | 987 / 51.3 |

¹ The edit prompts stay under 32k tokens, where the 512k mode is the 256k mode bit for bit (an earlier 512k build that read
those rows with the int8 kernel too: 114.9 / 127.2).

The reading path's blocks get smaller as the context grows (the KV cache takes their room): at CTX 65,536 the same 37.6k read
runs at 646 tok/s (21 Nov: 66.3 / 114.3 / 126.8 / 646 / 55.4). Writing after the read is lower at 512k only because the int8
rows past 32k change the answer a little and with it the drafts taken: with f16 up to 40,960 it is 57.18, the 256k number.

Inkling thinks before it answers even at effort 0; in the Python edit (a file without the variable it is asked to rename) it
reasons for over 1,600 tokens, so that number measures writing its reasoning, not copying. Long reads are bound by f32
arithmetic: the P100 has no tensor cores, and Flash-Next's fast numbers use f16 there.

**Long context:** a word hidden in the first line of a **239,114-token** prompt was found at 256k (read in 876 s, 273 tok/s on
average, writing after it at 42.2 tok/s with the drafter), and of a **496,073-token** prompt at 512k (read in 3,041 s, 163
tok/s on average, writing after it at 24.2). The f16 KV cache costs 16 KB a token on each card, so 256k is the most that fits
in f16 with the drafter; see [256k or 512k](#256k-or-512k) for how 512k fits. 1M tokens (the model's limit) cannot fit even
in int8 (8.5 GiB of KV a card).

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
bash engine-inkling2/build.sh                 # ink2_v1 (nvcc), ink2_vis + llama.cpp's llama-tokenize on the CPU
bash engine-inkling2/setup.sh /data/inkling2  # the model (17.6 GB), the DSpark drafter (0.4 GB), the projector (0.64 GB), a venv
bash engine-inkling2/run.sh /data/inkling2 256k   # or 512k; OpenAI-compatible API on http://127.0.0.1:8080/v1
```

`run.sh DATA [CTX] [PORT]`: CTX `256k` (= 262144), `512k` (= 524288) or any number up to 524288 (default 32768); pictures are
on when DATA holds the projector (`setup.sh DATA --no-pictures` leaves it out, `INK2_PICTURES=0` turns them off).
`python3 engine-inkling2/tools/bench.py` measures fresh writing, an edit and a long read through the API,
`python3 engine-inkling2/tools/picture_check.py` sends a picture (by default this branch's card picture) and asks for its title.
Checked from a clean clone of this branch on our machine (22 Nov, CUDA 12.4, `HOST_CXX=g++-13`): `build.sh` in 3.5 min
(llama.cpp downloaded and built anew), `setup.sh`, then for each option `run.sh`, `tools/picture_check.py` and `tools/bench.py`:

| | `run.sh DATA 256k` | `run.sh DATA 512k` |
|---|---|---|
| Ready, VRAM a card | 48 s, 15.19 GiB | 48 s, 15.24 GiB |
| Picture: the title | read word for word (30.6 s with the encoding, 952-token prompt in 1.5 s, writing 77.0 tok/s) | the same, 30.9 s |
| Fresh writing | 70.4 tok/s | 70.4 |
| The edit | 77.8 tok/s | 77.8 |
| 34,699-token read / writing after it | 594.9 / 58.7 tok/s | 600.3 / 59.1 |

The edit test repeats `serve/ink2_server.py`, which grew with the pictures (21 → 26 KB) and now makes Inkling reason longer
first: its 21 Nov copy gives 99.1 tok/s on the same engine at 256k (105.9 on 21 Nov at CTX 65536). On 21 Nov at CTX 65536:
fresh 70.8, a 34,699-token read at 684.8 (bigger reading blocks; see [Speed](#speed)). If your default g++ is newer than CUDA
12.4 accepts, set `HOST_CXX=g++-13`.

Files: Xylolabs's [`Inkling-2-Mini-NVFP4-Q8_0.gguf`](https://huggingface.co/Xylolabs/Inkling-2-Mini-GGUF) (the original
NVFP4 experts with a scale per expert, the last four layers Q8_0, the rest Q8_0 / F32) and Unsloth's
[`dspark-Inkling-2-Mini-Q8_0.gguf`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), the model's own MTP drafter, and
Xylolabs's `mmproj-Inkling-2-Mini-BF16.gguf` (same repository as the model), its vision projector. The
model is Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini); its license applies to
the weights.

The server (`serve/ink2_server.py`) answers `/v1/chat/completions` (streamed or not; the thinking part as
`reasoning_content`), `/v1/models`, `/health`, `/props`, `/slots`, `/tokenize` and `/apply-template`, so llama-server
clients work. `reasoning_effort` (none … xhigh) sets Inkling's "Thinking effort level". **Tool calls** work: OpenAI `tools` go in
as the template's `tool_declare`, the model's calls come back as `tool_calls` (streamed too), tool results as its tool messages;
the prompt equals the GGUF template's to the character (`tools/template_check.py`), and a web search tool in a chat app and a
coding agent's write / bash tools worked on our machine. **Pictures**: `image_url` parts (a data URL, an http(s) URL or a
local path; checked with one picture a message). Limits: **greedy only** (no sampler yet), one request at a time. A follow-up turn of a
conversation reads only the new part (the engine keeps a snapshot of its state before the last prompt token).

## 256k or 512k

At 256k a card holds the experts (7.5 GiB), the full layers' KV cache in f16 (4.1 GiB), the reading path's buffers (1.8 GiB),
the drafter (0.65 GiB) and the rest (1.4 GiB), with ~0.2 GiB free. The f16 cache at 512k would need ~3.8 GiB more. Moving the
experts to RAM would free it, but they would cross PCIe on every token (30-50 % slower writing), so the 512k mode does this
instead, by itself when CTX > 262144:

- **The full layers' KV rows from 32,768 on are int8**, in groups of 32 values with an f16 scale each (absmax / 127); the
  rows below stay f16. The window layers (1,024 tokens) are unchanged. Writing reads the f16 part with the 256k kernel and
  the int8 part with its own, so up to 32k tokens the **output is bit for bit the 256k mode's** (checked at 14k) and as fast.
- **Buffers that never run at once share memory:** the reading path's score matrix lives in the writing path's attention
  parts, the reading path's f32 K / V rows, R·B and P·V pieces in the MoE's scratch (a layer's attention and its experts run
  one after the other on a card's single stream). 15.24 GiB a card in all.
- **Cost past 32k:** KL of int8 against f16 over the last 16 rows 4.8·10⁻³ at 4k (int8 from row 0 for the test), 4.1·10⁻³
  at 14k, 3.4·10⁻³ at 40k; for scale, our exact f16 against llama.cpp is 1.24·10⁻² at 4k already.
- Switches: `INK2_KV8` (0 / 1, forces the mode off or on), `INK2_KV8_F16` (32768, the rows kept in f16), `INK2_KV8_PART`
  (3: 1 = K only, 2 = V only), `INK2_KV8_ROT` (0; a Hadamard rotation before quantizing: correct, 70× smaller error on a
  13-token prompt, but worse on long ones, since Inkling's K has a few huge components beside small ones). `INK2_VRAM_LOG=1`
  prints the VRAM by part after loading.

## Pictures

`build/ink2_vis` (`src/ink2_vis.cpp`) runs llama.cpp's mtmd at `690b87d`, the code llama-server uses for `--mmproj`: the
projector's preprocessing and ViT on the CPU (the cards are full at long contexts; `CUDA_VISIBLE_DEVICES=` keeps it off
them), ~20 s for a 1600×900 picture, 920 rows of 2,304 values. The server keeps one `ink2_vis` running, caches rows by the
picture's SHA-256 (`INK2_IMG_DIR`, default `/tmp/ink2-img`) and gives each picture's rows ids from 2²⁸ up, made from that
hash, so a conversation's cached state never mixes two pictures. The engine puts the rows in place of the template's
placeholder and, as llama.cpp does, skips `token_embd_norm` for them. Checked on this branch's card picture: the title read
word for word (the 960-token prompt read in 1.5 s, the answer written at 85.5 tok/s).

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
