# The low-RAM engine (32 GB and 64 GB modes)

Three profiles run an engine of their own, so the engine every other profile runs (`engine/`, `build/`) is not touched by
them. On a machine with 32 GB of RAM ([docs/GUIDE.md, "32 GB of RAM"](../docs/GUIDE.md#32-gb-of-ram)) there are two: the
Coder's experts with exact routing, and the fast UD-Q4_K_XL. On a machine with 64 GB there is the exact UD-Q4_K_XL
([docs/GUIDE.md, "UD-Q4_K_XL on 64 GB"](../docs/GUIDE.md#ud-q4_k_xl-on-64-gb)):

```bash
bash build.sh --32g      # build-32g/split/strata-split-2
bash setup.sh /data/flash-next-q4-coder-32 --models /data/flash-next-q4/models --model q4_k_xl --coder --ram 32   # 32 GB, Coder
bash setup.sh /data/flash-next-q4-32 --models /data/flash-next-q4/models --model q4_k_xl --fast --ram 32          # 32 GB, fast
bash setup.sh /data/flash-next-q4-64 --models /data/flash-next-q4/models --model q4_k_xl --ram 64                 # 64 GB
```

`build.sh --32g` copies `engine/` to `build-32g/engine-32g`, applies the four patches below to the copy, in this order, and
builds it into `build-32g`. The profiles that `setup.sh --ram 32` and `--model q4_k_xl --ram 64` (or `profile/fill_profile.py`
with the same options) write name that engine.

| File | What it is |
|---|---|
| `ssd_patch.py` | `FNX_SSD_EXPERTS`: no routed expert gets a page-locked RAM copy; its host copy is its bytes in the model's files, read when a card needs it (by each card's copier while writing, `FNX_SSD_AHEAD` layers ahead while reading a prompt). Copies `ssd.hpp` from beside itself into `src/` |
| `ssd3_patch.py` | `FNX_SSD_RAM_GIB`: the RAM tier, page-locked, sized by need under that cap, exclusive with the cards' expert slots (an evicted expert is copied down into it before its slot is written), warmed at the load with the hottest experts the cards do not hold (the coldest in the slots CLOCK reaches first), read by the prompt path. Runs after `ssd_patch.py` |
| `ssd3f_patch.py` | the K/V pool's borrowed arena slots: their experts go down into the RAM tier, and slots given back are filled from it. Runs after `ssd3_patch.py` |
| `ssd3g_patch.py` | the copies down in two parts, every gate\|up part first and then every down part, each with its own event, so an expert's gate\|up part coming up waits only for its slot's gate\|up part going down. Runs after `ssd3f_patch.py` |
| `ssd.hpp` | the reads from the files, the slot fills, the switches' defaults |

Every patch checks every edit it makes (the build stops if one does not match), and with `FNX_SSD_EXPERTS` unset the
patched engine takes none of the new paths. They apply to `engine/`'s sources of 19 Nov 2026 (SHA-256 in
[docs/PROVENANCE.md](../docs/PROVENANCE.md)); our build of the result, `3d1c17ba`, is what all three profiles were measured
on (the fast 32 GB mode first on `4b5107bf`, with the first two patches as they were that afternoon; the Coder's on 21 Nov,
nothing rebuilt). The switches are in [docs/SWITCHES.md, "The 32 GB mode"](../docs/SWITCHES.md#the-32-gb-mode), which
covers all three.
