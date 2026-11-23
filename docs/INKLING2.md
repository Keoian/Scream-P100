# Scream P100 · Inkling 2: an engine of its own for Inkling 2 Mini on two Tesla P100

> ⚠️ Built and measured on **one machine** (2× Tesla P100 16 GB, each on its own PCIe 3.0 x16 link, 99 GiB RAM), written on
> 21-22 Nov 2026. Nobody else has run it yet. Reports are welcome.

**Options:** a context of **256k** (the KV cache all in f16) or **512k** (the long part of the KV cache in int8, the first 32k
tokens exactly as at 256k, the same speed), and **pictures** (the model's vision projector on the CPU) in both. **Reading
30-37 % faster since 22 Nov** (p19, [below](#reading-faster-p19)): 790 tok/s on a 37.6k prompt at 256k or 512k, was 578.
**A conversation keeps its place** (22 Nov evening, p21-p23, [below](#conversations-keep-their-place-p21-p23)): a chat's next
turn reads only its new messages (first byte after 0.76 s instead of ~20 s on a 16k conversation), a conversation that
another one pushed off the cards comes back from RAM in under a second, writing is 10 % faster without drafts and 2 % with
them, and code in a prompt is tokenized as written. **Short pieces 18-22 % faster** (23 Nov, p25-p27,
[below](#short-pieces-read-faster-p25-p27)): a chat turn's 64 or 128 new tokens are read in 251 / 336 ms instead of 322 /
409, the experts of the last 4 layers no longer go through f32 copies.

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
| Reading a 4,096 / 14,336 / 40,960-token prompt | 143 / 119 / — | **~965 / 917 / 798** (at CTX 256k or 512k; ~940 / 887 / 776 before p25) |
| VRAM a card | 14.0-14.5 GB | 8.9 GB writing only, 11.8 GB with the drafter at 32k |
| Context | 32k measured | **262,144** (KV f16) or **524,288** (KV int8 past 32k), 15.2 GiB a card with the drafter |
| Pictures | mmproj (on a card) | mmproj on the CPU, ~20 s for a 1600×900 picture (920 rows), then read at ~640 tok/s |

On the **test set of our Flash-Next builds** (the same prompts, through the OpenAI API, `INK2` with its defaults), against
Scream P100's Flash-Next UD-Q4_K_XL on the same machine (22 Nov, at the context each option runs with):

| Test | **ink2, 256k (exact)** | **ink2, 512k** | Flash-Next Q4_K_XL, exact | Flash-Next Q4_K_XL, fast (lossy) |
|---|---|---|---|---|
| Fresh writing (12 prompts, thinking low) | 67.3 | 67.2 | 59.3 | 68.9 |
| Edits, Python / Markdown ¹ | 75.3 / 114.3 | 75.3 / 114.4 | 100.9 / 92.2 | 124.3 / 125.6 |
| Koda's 26 edit tasks (tool calls, hidden tests) | **97.4** tok/s, 24/26 right | — | — | — |
| Reading 37.6k / writing after it | **790** / 53.2 | **790** / 56.7 | 802 / 61.8 | 987 / 51.3 |

¹ Inkling thinks before it answers even at effort 0, and both of these prompts are 1,600 tokens of reasoning (in the Python
one the file has no variable of the name it is asked to rename), so they time its reasoning, not copying, and follow the
text: with p19's reading path the sums of the prompt come out in another order, the greedy reasoning takes another turn,
and the two numbers went from 115.5 / 127.8 to 75.3 / 114.3, while the same build with every p19 switch off gives the old
numbers exactly (`INK2_PF_QSF=64 INK2_SHARE=0 INK2_GEMM5=0 INK2_XSPLIT=1`: 115.2 / 127.4). On Koda's 26 edit tasks the
speed is the same before and after (97.3 / 97.4 tok/s). Writing after the long read differs between 256k and 512k for the
same reason (the int8 rows past 32k change the answer a little). Long reads are bound by f32 arithmetic: the P100 has no
tensor cores, and Flash-Next's fast numbers use f16 there.

**Long context:** a word hidden in the first line of a **239,114-token** prompt was found at 256k (read in 876 s, 273 tok/s on
average, writing after it at 42.2 tok/s with the drafter), and of a **496,073-token** prompt at 512k (read in 3,041 s, 163
tok/s on average, writing after it at 24.2). The f16 KV cache costs 16 KB a token on each card, so 256k is the most that fits
in f16 with the drafter; see [256k or 512k](#256k-or-512k) for how 512k fits. 1M tokens (the model's limit) cannot fit even
in int8 (8.5 GiB of KV a card).

**Quality** (21 Nov, before p19; p19 in [Reading faster](#reading-faster-p19)): greedy outputs of 32 tokens on English, Slovenian and code prompts are the same as llama.cpp's; KL of the last
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
bash engine-inkling2/build.sh                 # ink2_v1 (nvcc); ink2_vis, ink2_tok + llama.cpp's llama-tokenize on the CPU
bash engine-inkling2/setup.sh /data/inkling2  # the model (17.6 GB), the DSpark drafter (0.4 GB), the projector (0.64 GB), a venv
bash engine-inkling2/run.sh /data/inkling2 256k   # or 512k; OpenAI-compatible API on http://127.0.0.1:8080/v1
```

`run.sh DATA [CTX] [PORT]`: CTX `256k` (= 262144), `512k` (= 524288) or any number up to 524288 (default 32768); pictures are
on when DATA holds the projector (`setup.sh DATA --no-pictures` leaves it out, `INK2_PICTURES=0` turns them off).
`python3 engine-inkling2/tools/bench.py` measures fresh writing, an edit and a long read through the API,
`python3 engine-inkling2/tools/picture_check.py` sends a picture (by default this branch's card picture) and asks for its title,
`python3 engine-inkling2/tools/conversation_check.py` runs a chat with a long system message and a tool the way a chat app
sends it and says how many tokens each request read.
Checked from a clean clone of this branch on our machine (22 Nov, CUDA 12.4, `HOST_CXX=g++-13`): `build.sh` in 3.5 min
(llama.cpp downloaded and built anew), `setup.sh`, then for each option `run.sh`, `tools/picture_check.py` and `tools/bench.py`;
in the morning with the pictures and the 512k option, at noon again with p19's faster reading (the same clone, `build.sh`
again in 46 s):

| | `run.sh DATA 256k` | `run.sh DATA 512k` |
|---|---|---|
| Ready, VRAM a card | 48 s, 15.19 GiB → **14.76** (p19) | 48 s, 15.24 GiB |
| Picture: the title | read word for word (30.6 s with the encoding, 952-token prompt in 1.5 s → **1.2 s**) | the same, 30.9 s |
| Fresh writing | 70.4 tok/s (p19: 70.4) | 70.4 (70.5) |
| The edit | 77.8 tok/s (p19: 87.8) | 77.8 (88.0) |
| 34,699-token read / writing after it | 594.9 / 58.7 → **789.6** / 57.3 tok/s | 600.3 / 59.1 → **800.0** / 58.5 |

The edit test repeats `serve/ink2_server.py`, which grew with the pictures (21 → 26 KB) and makes Inkling reason first, so
its number follows the text of that reasoning (its 21 Nov copy gave 99.1 tok/s on the morning's engine at 256k, 105.9 on 21
Nov at CTX 65536). On 21 Nov at CTX 65536: fresh 70.8, a 34,699-token read at 684.8. If your default g++ is newer than CUDA
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
conversation reads only the new part, and so does a prompt that parts from the conversation earlier or comes back after
another one ([checkpoints and parked conversations](#conversations-keep-their-place-p21-p23)).

## Conversations keep their place (p21-p23)

22 Nov, evening. The kernels were not where a chat lost its time. The engine kept one snapshot, at the end of the last
prompt, and a request read its whole prompt again from token 0 in four everyday cases:

| What happened | Measured before | Why |
|---|---|---|
| Every turn of a chat in our app | first byte after 9.2 s; all ~7,200 tokens read each turn | the app moves a note about the environment from the previous user message to the new one, so the prompt parts from the conversation 13 tokens **before** the old prompt's end, where the snapshot stood |
| A client shortened its history | 54 s for 40.8k tokens | the same: it parted before the snapshot |
| Any answer over 3,056 tokens | 83 s for a 58k conversation, at every request after it | the snapshot was good only while the windows' ring (4,096 rows) had not been written over |
| Another conversation in between | everything read again | the cards hold one conversation |

And every request waited 0.9 s before reading at all: `llama-tokenize` started anew and read the vocabulary.

**Checkpoints in RAM.** A checkpoint is what it takes to go back to a place in the conversation: the convolutions' states,
the last 1,024 rows of every window slot's ring and the drafter's rings (278 MiB for both cards, page-locked RAM; the full
layers' rows stay where they are, they are kept by position). The server tells the engine where the prompt's messages
start; the engine takes a checkpoint at the last of them, at earlier ones 4,096 tokens apart, every 16,384 tokens of a long
read and before the prompt's last token. A request goes back to the latest checkpoint its prompt still shares. Taking one
costs 15-30 ms (~150 ms the first time a buffer is page-locked), going back 30 ms.

**Parked conversations.** When a request shares less than half of the conversation on the cards, that conversation's full
layers' rows (32.8 KB a token for both cards, its int8 rows and scales in the 512k mode) go to RAM with its tokens and its
first and last three checkpoints. A later request that shares more with a parked conversation than with the one on the
cards gets its rows copied back: the whole conversation when the prompt goes on from its end, a copy up to a checkpoint
when it parts earlier (a new chat with the same system prompt and tools). RAM is taken as it is needed, up to
`INK2_CKPT_MB` (24576) for checkpoints and `INK2_PARK_MB` (32768) for parked conversations, the least recently used out
first: **set them lower on a machine with less RAM** (ours has 99 GiB).

**A tokenizer that stays up.** `build/ink2_tok` is llama.cpp's tokenizer with the same flags as `llama-tokenize --no-escape`,
as a process the server starts once. The prompt is cut where its messages' parts start (special tokens, so each piece
tokenizes by itself as in the whole) and the pieces' tokens are cached by their text: ~10 ms a request instead of ~900.
**And the text is now tokenized as written:** the server ran `llama-tokenize` without `--no-escape`, which turns a written
`\n`, `\t` or `\\` (two characters, as in source code) into the character itself, so code in a prompt reached the model with
broken string literals.

Through the server, a chat with a 16.2k-token system message and a tool, the note moving each turn
(`tools/conversation_check.py` does the same with a 13k one):

| Request | Read before | Read now | First byte now |
|---|---|---|---|
| The first turn | 16,226 tokens | 16,226 | 20.7 s |
| The tool's round | 16,654 | 429 | 1.6 s |
| The second turn | 16,281 | 106 | 0.76 s |
| The third turn | 16,347 | 184 | 0.86 s |
| The turn after a 1,340-token answer | 17,505 | 1,342 (that answer and the new message) | 2.2 s |

The answer before a turn is still read again: its user message changed (the note left it), and what was written after it is
no longer true to the prompt. A client that leaves its history as it was sends only its new message.

On the engine's socket (`--ctx 262144` and `524288`), each against the same prompt read from token 0 with the same hints.
The logits after the prompt (a hash of all of them), the tokens written and the drafts taken were **the same in every case**:

| Case | From token 0 | Now |
|---|---|---|
| The same 13.2k prompt after 3,500 written tokens | 15.7 s | 0.03 s |
| A prompt that parts 13 tokens before its end | 16.2 s | 1.0 s |
| A 38k prompt shortened to 31k with a new tail | 40.5 s | 2.2 s (drafts taken after it: 202 of 310, the same) |
| A 13.2k conversation back after another one | 15.6 s | 0.76 s |
| A 20k conversation back | 24.5 s | 0.46 s |
| A new chat that shares its first 13k tokens | 15.9 s | 1.9 s |
| A 38k conversation parked and back (int8 rows at 512k) | 51.5 s | 0.79 s (drafts 98 of 148, the same) |

**Writing: fewer launches, the same bits.** A token is 48 passes of 16 to 29 small kernel launches. `src/ink2_fuse.cuh` runs the
three loop controllers' down projections in one launch, their gains and forgets in another, the three inputs' norms in one
and their Q8 matrices in one; every block does what it did before. Against the launches before (`INK2_FUSE2=0`), on 14
requests with a hash over the logits of every step: all 14 the same bit for bit, single tokens +9.6 to +10.2 %, draft
windows +1.0 to +4.2 %. A launch saved is the same ~1.6 ms a step either way: a step of one token takes ~17.7 ms, a draft
window of three or four rows ~30 ms (its experts ~45 % of the time, and their kernels are not touched) for ~2.7 tokens.

**Short pieces.** A piece of a prompt on the reading path costs ~250 ms whatever its length (a 128-token piece: 57 % of it in
the experts, whose smallest tile is 16 tokens wide for ~8 tokens an expert), and a piece of up to 16 tokens went as one
window of 16 rows: 239 ms, while a window of 8 rows takes 56 ms. Pieces of up to 40 tokens now go in windows of 8 rows: 16
tokens 120 ms, 17 tokens 124 ms (was 244), 32 tokens 224 ms (was 283). A draft window stays one window of up to 16 rows.

**Against the 22 Nov morning build** (p19e with its server; the same prompts, temperature 0):

| | p19e (morning) | **now** |
|---|---|---|
| Writing, no drafts (14 requests of the writing test, the same text) | 56.4 tok/s | **62.2** (+10 %) |
| Writing with drafts (the same 14 requests) | 82.1 | **83.9** (+2 %) |
| Fresh writing, the test set's 12 prompts (drafts on, as the server sends them) | 67.2 | **71.2** (+6 %) |
| Edits, Python / Markdown ¹ | 75.2 / 114.3 | **94.2 / 121.2** |
| Reading a cold 37.6k prompt / writing after it | 775.5 / 53.1 | 761.0 / **57.2** ² |
| Koda's 26 edit tasks / 45 functions with hidden tests / English 16 / Slovenian 15 | 24/26 · 43/45 · 16/16 · 12/15 | **24/26 · 45/45 · 16/16 · 12/15** |

¹ The Python file has a written `\n` in two lines, which the morning's server turned into newlines: the prompt now carries the
file as written. ² The four checkpoints of a cold read cost ~1 s of its 49 s (`INK2_CKPT=0`: 779 tok/s), most of it page-locking
their buffers the first time.

Checked from a clean clone of this branch (22 Nov, 18:09-18:19, the clone of the checks above, `build.sh` again in 61 s with
`build/ink2_tok`): `run.sh` 256k and 512k came up in 62 / 48 s at 14.76 / 15.24 GiB a card, `tools/picture_check.py` read
the title, and `tools/conversation_check.py` passed at both (a 13k chat: the tool's round read 342 tokens, the second turn 56
in 0.42 s, the turn after an answer 117 in 0.65 s). That clone was one line older: a draft window of more than 8 rows was cut
in two, so longer drafts were never taken. The line is fixed, and the fixed engine went through every test of this section
in our tree, the comparison with the morning build included.

## Short pieces read faster (p25-p27)

23 Nov. Since p21 most requests of a conversation read a few dozen to a few hundred new tokens, and such a piece had a fixed
cost of ~270 ms whatever its length. A profile of 40 pieces of 128 tokens showed where most of it went: the last 4 layers
keep their experts in Q8_0 (the release's MXFP8, converted), and on the reading path they still went the oldest way, 16
experts at a time copied to f32 and then one cuBLAS call an expert with ~8 tokens. That was ~113 ms of a 407 ms piece on each
card for 4 of the 48 passes, while a pass of NVFP4 experts took ~5 ms.

| Build | What | Output |
|---|---|---|
| p25 | the Q8 experts by the tiles of the NVFP4 expert GEMM, straight from the int8 rows and their half scales (`k_q8_gemm5`): the weights are the same floats as before, each sum one FMA chain | last row against p23b: KL 6·10⁻¹¹ / 5·10⁻¹² / 4·10⁻¹³ (4k / 14k / 40k), the same 16 tokens written after each |
| p26 | expert tiles of 8 and 4 tokens, NVFP4 and Q8 (an expert's last few tokens took a 16-token tile, half of it empty) | the same bits as p25 |
| p27 | the reading path from 31 tokens on (was 41): it now beats the writing path's 8-row windows from ~30 tokens | the same bits as p26 above 40 tokens and in writing |

A piece after a 16k conversation (back to a checkpoint, a piece of N tokens, then its last token), the same day:

| Piece | p23b (22 Nov) | **p27** |
|---|---|---|
| 32 tokens | 224 ms | **214 ms** |
| 40 tokens | 290 ms | **237 ms** |
| 64 tokens | 322 ms | **251 ms** |
| 128 tokens | 409 ms | **336 ms** |
| 256 tokens | 562 ms | **484 ms** |
| 512 tokens | 837 ms | **762 ms** |
| 2,048 tokens | 805 tok/s | **829 tok/s** |

A prompt read from token 0 at 256k: 4k 924 → 965, 14k 894 → 917, 40k 783 → 798 tok/s; the shared 37.6k read through the
server 751 → 785. Writing is the same: the same tokens and drafts as p23b on 14 requests (77.3 tok/s against 77.5). Quality
(p26, the same bits as p27 there): functions 45/45, English 16/16, Slovenian 12/15 (the same three), 0 of 18 answers not in
Slovenian, 0 of 6 with foreign words, Koda's edits 24/26 at 102 tok/s; the checkpoints and parked conversations at 256k and
512k give the same logits and tokens as reading from token 0.

Checked from a clean clone of this branch (23 Nov, 10:04-10:13, commit 1190bfc in the clone of the checks above): `build.sh`
in 61 s, `run.sh` 256k and 512k up in 54 / 46 s at 14.76 / 15.24 GiB a card, `tools/picture_check.py` read the title, and
`tools/conversation_check.py` passed at both (a 13k chat: the second turn read 56 tokens in 0.41 s, the turn after a long
answer 117 in 0.55 s, was 0.65); `tools/bench.py`: fresh 73.4 tok/s, the edit 88.1, a 34.7k read at 792 (256k) / 792 (512k).

## Reading faster (p19)

22 Nov. A profile of a 40k read at 256k showed the time going to the NVFP4 experts (40 %), the full layers' attention (18 %),
the dense matrices (19 %, already ~7.4 TFLOPS in cuBLAS) and the two cards' exchanges (13 %). Four changes, all in f32:

| Build | What | 4k / 14k / 40k at CTX 256k, tok/s | Output |
|---|---|---|---|
| p18e | before | 717 / 655 / 571 | |
| p19a | the full layers' query blocks sized by the chunk's keys, not by the longest context (at 256k they were 64 queries even for a 4k prompt, and P·V ran in calls of a few CUDA blocks), up to 2,048 | 794 / 760 / 664 | KL 1.5·10⁻⁶ against p18e |
| p19c | the shared scratch of the 512k mode in every mode: 0.43 GiB less VRAM at 256k, room for bigger blocks | 792 / 755 / 667 | 256k = 512k bit for bit below 32k |
| p19b | experts by a tile list: one CUDA block a tile of 64, 32 or 16 tokens (the smallest that holds an expert's last tokens), so a big expert's tiles run side by side instead of one after another in one block | 884 / 839 / 745 | the same bits as p19c |
| p19e | the exchanges in 4 slices: each slice goes to RAM on a copy stream as soon as it is made, while the next one is made, and comes back on another | **~940 / 887 / 776** | KL 1.9·10⁻⁶ against 2 slices |

On the expert GEMM alone (`k_fp4_gemm5`, a chunk's routing): gate_up 18.2 → 14.2 ms, down 8.4 → 7.5 ms, the same bits.
Quality of p19e at 256k, against p18e on the same sets and server: English questions 16/16 (16/16), Slovenian 12/15 with the
same three missed (12/15), 32 greedy tokens on English / Slovenian / code = llama.cpp, the last row of a 4k prompt KL 0.0107
against llama.cpp with the same first token (the reading path of 21 Nov: 0.0108, its top two swapped). Functions 43/45 (44/45)
and Koda's edits 24/26 (25/26): two tasks flipped, both to a wrong answer of an ordinary kind (`flatten({})` gave `{'': {}}`; a
rename done as one replace-all that also renamed `ZNIZAN_DDV`). The change of the sums is as big as the one p18e itself
makes between two contexts (dolg40k's last row: p18e at 64k against 256k KL 1.5·10⁻⁶, p19e against p18e 3-6·10⁻⁶). Koda's edits: the same 26 tasks at CTX 65536 gave p19e 24/26 and p18e 24/26 (both miss konstanta_en there), functions 43/45 and 44/45: the one task that stays apart is flatten.

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

## Loops in the reasoning

The engine has no sampler: every answer is greedy. On 22 Nov, after ~20 tool calls (web searches and pages, a 15k-token
prompt), Inkling's reasoning fell into repeating itself word for word: two lines 194 times over, then a list 21 times, 84k
characters in 13 minutes and no answer (the client asked it to go on each time it hit the length limit). Since then
`serve/ink2_server.py` watches the reply's reasoning as it streams. When the reasoning's tail repeats one block of 32 to
8,000 characters at least 3 times back to back and over 2,000 characters, the server stops the engine. It then keeps the
reasoning up to the end of the block's first copy and closes it with one sentence ("I am repeating the same lines. I have
enough to answer, so I stop thinking here and write the answer now."). The model then writes its answer. The engine reads
the kept part again from the prompt's snapshot, so this costs about a second. A second loop in the same reply ends it.
`timings.loop_cuts` counts the cuts, and the server's log prints each one.

On that day's job, the guard would have cut after 5,248 characters of the last round instead of 84k. Tested on 562
reasoning and answer texts from our test runs (1.74M characters), it never cut one. With a reasoning that starts by
repeating the two lines, the engine was cut after 9.3 copies, and the model wrote a short answer from what it had found.
Plain, streamed and tool-call requests give the same tokens as before. `tools/loop_check.py` checks the guard without the
model. Answers themselves are not watched, since a long table or matrix can repeat a row legitimately.

## How it works

- **Split:** query heads 12 + 12 and KV heads 4 + 4 per card; every expert's FF split in half (gate and up rows, down
  columns); the output vocabulary split in half. Two sums per pass (after attention, after the experts) through pinned
  host memory, card 0's part first on both cards, so both hold the same bits. 96 exchanges per token. No NVLink, no P2P.
- **Writing path:** NVFP4 experts stored planar per expert (all codes, then all scales) and read by row kernels with the
  input in shared memory padded against bank conflicts (`k_fp4_rows7` / `k_fp4_rows6`); fused kernels for the exchange +
  post-convolution + residual, the K/V convolutions + clips + cache write, the router + job list; attention by KV head with
  the three query heads together.
- **Reading path:** chunks of 2,048 tokens; dense matrices as Q8 → f32 + cuBLAS; the experts sorted by expert and run by a
  tile GEMM that decodes NVFP4 in shared memory, one CUDA block a tile of 64 / 32 / 16 tokens from a list (~6 TFLOPS); window
  attention for all query blocks and heads in one batched call; full-attention layers with query blocks sized by the
  chunk's keys, keys in 32k blocks and P·V split by 1,024-key pieces at long context; the cards' sums in 4 slices that
  travel while the next slice is made.
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
`INK2_T1TT` (1), `INK2_WIN_SEG` (0), `INK2_PERM` (0), `INK2_FEED_T` (8, rows of a window when a short piece is read; 16 before p23), `INK2_PF_MIN` (31, the shortest piece on the
reading path; 41 before p27, 17 before p23), `INK2_PF_CHUNK` (2048),
`INK2_PF_QSF` (1024, smaller on its own past 96k), `INK2_PF_SMB` (MB for the score matrix), `INK2_PF_ATTB` (1),
`INK2_PF_RB` (0), `INK2_PV_SPLIT` (1), `INK2_DS_T1` (1), `INK2_DS_ROWS` (1 = k + 1 rows, 0 = all 8), `INK2_DS_PMIN` (0.7),
`INK2_DS_RING` (on past 64k), `INK2_NG_MIN` (6), `INK2_SNAP` (1), `INK2_PF_QSF_MAX` (2048, the largest query block; a
set `INK2_PF_QSF` fixes it), `INK2_SHARE` (1, the shared scratch in every mode), `INK2_GEMM5` (1, the tile-list expert
GEMM), `INK2_XSPLIT` (4, slices of an exchange; 1 = one copy), `INK2_KV8` / `INK2_KV8_F16` / `INK2_KV8_PART` / `INK2_KV8_ROT`
(the 512k mode), `INK2_VRAM_LOG`, `INK2_CKPT` (1, checkpoints; 0 = the one snapshot of before), `INK2_CKPT_MB` (24576,
RAM for them), `INK2_CKPT_GAP` (4096) / `INK2_CKPT_EVERY` (16384), `INK2_PARK_MB` (32768, RAM for parked conversations; 0 =
off), `INK2_PARK_MIN` (1024, shorter conversations are read again), `INK2_FUSE2` (6, a mask of the merged launches: 2 the
controllers, 4 the norms and matrices; 1 the depth mixes and 8 the routing measured no gain), `INK2_Q8G` (1, the Q8 experts
of the reading path by `k_q8_gemm5`; 0 = f32 copies and cuBLAS), `INK2_TILE_S` (1, expert tiles of 8 and 4 tokens),
`INK2_CATM` (0, see below). Engine flags: `--model`, `--mtp`, `--draft-vocab`, `--ctx`,
`--serve SOCK`, or `--tokens FILE [--gen N] [--logits F] [--logits-all F] [--dump DIR]` for tests (`--logits-all`: every row of the last window).

## What did not help

A CUDA graph per token (no gain), reading an expert once per draft window (slower), 128×128 expert tiles on the reading path
(slower), a custom R·B kernel (same as cuBLAS), job order by expert in a window (same speed). For p19: 8 × 8 values a
thread in the expert GEMM, with 128 threads or with 128-token tiles (both slower than 8 × 4 with 64-token tiles); an
expert's tiles in one block with 32 / 16-token remainders (worse on skewed routing than one block a tile); query blocks
above 256 at 40k (no gain); the exchange buffers placed by NUMA node (2.99 instead of up to 3.40 ms an exchange, ~1 %).

For p22 (writing, all with the same bits as before): the rms norms computed inside every block of the matrices' launch
instead of a launch of their own (single tokens +1.5 %, draft windows -8 to -13 %: the blocks are bound by their
instructions, not by the launch); the routing done by the router's last block (windows -1.8 to -2.7 %); the two depth mixes
in one launch (+0.1 %); the experts' rows with a lane on several rows at once and the input in registers instead of shared
memory (20-80 % slower than `k_fp4_rows7` / `k_fp4_rows6`, which reach 369 / 366 GB/s on a window's 32 jobs).

For p25-p27: one cuBLAS call for each group of the attention's input projections that read the same input (q, gate, r and
sink; k with dk; v with dv), into a scratch and from there to each part's buffer: 128-token pieces 322 → 320 ms only, and
the changed bits in every layer move the routing's near ties (a 4k prompt got another first token, KL 4·10⁻³), so it stays
off (`INK2_CATM=1` turns it on). The small expert tiles in a kernel of their own with a smaller input buffer, 3 blocks an SM
instead of 2 (the same bits, 322 → 324 ms). One exchange instead of 4 slices on short pieces (329 against 322 ms). Our own
GEMM for the dense Q8 matrices (an idea from `iammrduncan/tensorless`): not built, the profile leaves it little, 3 % for
their f32 copies on a 128-token piece and cuBLAS at ~6.6 TFLOPS on the products.
