# engine-inkling2: `ink2`, Inkling 2 Mini on two Tesla P100

An engine of its own for Thinking Machines' Inkling 2 Mini (26.4B looped MoE, NVFP4 experts) on two Tesla P100 16 GB: both
cards on every layer at once, the whole model in VRAM, DSpark (MTP) and prompt-lookup drafts, an OpenAI-compatible server
with tool calls and pictures. Fresh writing 56.6 tok/s (78.8 with drafts), edits ~100-127, reading ~940 tok/s at 4k and
790 at 37.6k; context
**256k** (KV f16) or **512k** (KV int8 past 32k, the first 32k bit for bit as 256k, same speed); llama.cpp on the same cards:
13.0 / 143. A conversation keeps its place: checkpoints and parked conversations in RAM, so a chat's next turn reads only
its new messages. The full write-up: [docs/INKLING2.md](../docs/INKLING2.md).

```bash
bash engine-inkling2/build.sh                 # build/ink2_v1, build/ink2_vis, build/ink2_tok + llama-tokenize (CPU)
bash engine-inkling2/setup.sh /data/inkling2  # model 17.6 GB + drafter 0.4 GB + projector 0.64 GB (--no-pictures: without), a venv
bash engine-inkling2/run.sh /data/inkling2 [CTX=32768|256k|512k] [PORT=8080]
python3 engine-inkling2/tools/bench.py http://127.0.0.1:8080
python3 engine-inkling2/tools/picture_check.py http://127.0.0.1:8080
python3 engine-inkling2/tools/conversation_check.py http://127.0.0.1:8080
```

| Path | What |
|---|---|
| `src/ink2_v1.cu` | the engine: loading and splitting, the writing path, the reading path (2,048-token chunks), drafts, checkpoints and parked conversations, the socket server |
| `src/ink2_fuse.cuh` | merged launches of the writing path (the loop controllers, the norms and Q8 matrices), the same bits |
| `src/ink2_kern.cuh` | writing kernels: Q8 GEMVs, NVFP4 expert rows, attention by KV head, the fused exchange / convolution kernels |
| `src/ink2_pf.cuh` | reading kernels: NVFP4 tile GEMM, convolutions over a chunk, softmax with the relative bias and sink |
| `src/ink2_dspark.cuh` | the DSpark drafter's kernels (NeoX RoPE, block attention, the Markov head) |
| `src/ink2_vis.cpp` | the picture encoder: llama.cpp's mtmd (the vision projector) on the CPU, rows to a file |
| `src/ink2_tok.cpp` | llama.cpp's tokenizer as a process that stays up (the server's `--tok`) |
| `src/gguf_min.*` | a small GGUF reader (NVFP4 = type 40) |
| `src/bench_dec.cu`, `src/ink2_bench.cuh` | a one-card microbenchmark of the decode kernels on random weights |
| `serve/ink2_server.py` | the HTTP server (OpenAI API + llama-server endpoints, tools, pictures), greedy |
| `data/draft_vocab_32768.bin`, `tools/draft_vocab.py` | the drafter's reduced vocabulary and how it was made |
| `tools/bench.py` | fresh writing, an edit and a long read through the API |
| `tools/picture_check.py` | a picture and two questions through the API |
| `tools/conversation_check.py` | a chat-style conversation with a tool: tokens read a request, first byte; a written backslash-n |

Measured on one machine only (2× P100 16 GB, PCIe 3.0 x16 each, 99 GiB RAM, CUDA 12.4). MIT, like the rest of the repository;
the model's own license applies to its weights.
