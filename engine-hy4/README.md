# engine-hy4: Tencent Hy4 flash on two Tesla P100 (test build)

Our own engine for Hy4 flash, not a patch of `engine/`. The story, the numbers, the switches and the limits are in
[docs/HY4.md](../docs/HY4.md).

| File | What |
|---|---|
| `build.sh` | builds `build/hy4_v1` (the engine), `build/bench_dec` and `build/bench_bw` for sm_60 with nvcc |
| `setup.sh DATA` | the model from Xylolabs/Hy4-flash-GGUF (pinned revision, SHA-256), a Python venv, the GGUF's chat template, llama.cpp's `llama-tokenize` (PR #32517's commit, CPU) |
| `run.sh DATA` | the engine on a Unix socket plus the OpenAI-compatible wrapper on port 8080 |
| `bench.py [URL]` | fresh writing, two edits and a long read through the API, as in docs/HY4.md |
| `src/hy4_v1.cu` | model loading, the two-card engine (writing path, LRU, the socket server) |
| `src/hy4_kern.cuh` | the first version's kernels: IQ3_XXS and Q8 GEMVs, norms, iHC, attention, Engram, the cross-card sums |
| `src/hy4_dec.cuh` | the writing-path kernels: MLA over position ranges, grouped GEMVs, the router, the indexer's top-k |
| `src/hy4_pf.cuh`, `src/v1_pf_part.inc` | the prompt path: tiled IQ3 GEMM, cuBLAS attention, staging of a chunk's experts |
| `src/gguf_min.cpp/.hpp` | a small GGUF reader (mmap; a split model's shards by name) |
| `src/hy4_tables.h` | ggml's IQ3_XXS and IQ4_NL tables (MIT, the ggml authors) |
| `serve/hy4_openai.py` | `/v1/chat/completions` (stream, tools, reasoning_content), `/v1/models`, `/health` |
| `tools/bench_dec.cu`, `tools/bench_bw.cu`, `tools/sim_cache.py` | kernel checks, PCIe copy speed, the expert-cache simulator |

The engine's own socket protocol (what `serve/hy4_openai.py` speaks): one JSON line a connection,
`{"tokens": [...], "max": N, "stop": [...], "lookup": N, "stream": 1}` → with `stream`, one line `{"tok": id, "b": base64}` a
token, then `{"prompt_n", "cached", "read_s", "write_s", "hits", "misses", "drafted", "accepted", "tokens": [...]}`.
`{"exit": 1}` ends the server.
