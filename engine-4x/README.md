# The 4-card engine (PROOF OF CONCEPT)

> **PROOF OF CONCEPT, checked on one machine only.** It is built for a machine we do not have: four Tesla P100 16 GB, each on
> PCIe 3.0 x8 behind one PCIe switch, the switch on the chipset by PCIe 3.0 x4. KaneQuah ran it on such a machine
> ([issue #1](https://github.com/Megla123/Scream-P100/issues/1)): with the fixes below, IQ3_S's first-token logits match
> llama.cpp and our two-card engine. What it is, why, how to check that machine and what he measured:
> [docs/POC-4X.md](../docs/POC-4X.md).

```bash
bash build.sh --4x       # build-4x/split/strata-split-4; engine/ and build/ are not touched
python3 engine-4x/fill_profile_4x.py --models DATA/models --mtp DATA/prep/mtp-rt --tokenizer DATA/prep/tokenizer \
    --logs DATA/logs --out DATA/profile-4x.json
bash run.sh DATA --config DATA/profile-4x.json
```

`build.sh --4x` copies `engine/` to `build-4x/engine-4x`, applies the patches below to the copy (in the order `ff64_patch`,
`ncard4_patch`, `tail_patch`, `rows_tail_patch`, `peer_patch`, `prompt_peer_patch`; `FF4X=32` leaves `ff64_patch` out), and
builds it for four cards (`STRATA_SPLIT_CARDS=4`). Every patch checks every edit it makes (the build stops if one does not
match).

| File | What it is |
|---|---|
| `ncard4_patch.py` | `pf_gather` (`FNX_PF_SP`) for any count of cards: each card's finished part goes once to its own host rows and the others copy it from there (two cards: the same bytes in the same order as before). Edge0-145B (`FNX_LATENT`, `FNX_LAT_SPLIT`) refused on more than two cards with a plain message at the load |
| `ff64_patch.py` | KaneQuah's: four cards cut the experts' FF width in 64-column units (192, 192, 128, 128), not 160 each: the first-token logits of the 160-column build were wrong in a hardware run (issue #1); applied first, so `tail_patch.py`'s tail path is not used. The default; `FF4X=32 bash build.sh --4x` leaves it out |
| `tail_patch.py` | the prompt path's down kernel for FF slices of 160 columns (2.5 x 64): Kenzu's change for three P100 (github.com/Kenzu/qwen3.8-flash-next-2x-p100, commit `c43e4b8`), unchanged |
| `rows_tail_patch.py` | the cause of issue #1's wrong logits: the decode windows' row kernel added a sixth 32-column run past a 160-column down slice (the next row's bytes times the next token's activations). It now checks that run against the slice's end, as it did runs 2 and 3; with `ff64_patch.py`'s slices the check is always true |
| `peer_patch.py` | `FNX_PEER` on NVIDIA: the decode windows' sums and attention merge pushed into the other cards' memory when every pair passes a start probe, else the host exchange; copies `cuda_peer.hpp` into `src/` |
| `prompt_peer_patch.py` | KaneQuah's: `FNX_PF_PEER=1` (off by default; reading 338 -> 889 tok/s on his machine): the prompt path's chunk sums and attention partials go card to card into device buffers instead of page-locked host memory, flags stay in host memory; needs the peer probe to pass |
| `cuda_peer.hpp` | the IOMMU check (`FNX_PEER_IOMMU_TRUST=1` skips it, KaneQuah's switch), peer access and the start probe (from `engine/src/hip/peer.h`, the AMD version) |
| `flash-next-iq3s-4x-256k.template.json`, `flash-next-iq3s-4x-512k.template.json` | the IQ3_S profiles for four cards |
| `fill_profile_4x.py` | writes a run config from them through `profile/fill_profile.py` (`--context 256k`, the default, `512k` or `128k`; `--gpu`; `--lib-dir`, else the CUDA libraries' folders it finds) |
| `sim_gather.py` | a CPU simulation of the new gather's flags for 2 to 8 cards (no deadlock, every part right, none overwritten before it is read) |
| `logit_requests.py` | the same requests every run (short, ~2k, ~6-7k and ~20-25k tokens) and run configs with `env` entries changed, for the comparisons in docs/POC-4X.md |
| `compare_logits.py` | compares two runs' first-token logits (`FNX_DUMP_FIRST_LOGITS`), e.g. the 4-card engine against the 2-card one |

`tune/check.py --cards 4` looks at such a machine (all four cards, copies to all four at once, `nvidia-smi topo`).
