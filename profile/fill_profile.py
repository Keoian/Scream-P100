#!/usr/bin/env python3
"""Writes a run config for Strata's server (serve/server.py --config) from one of the templates in this folder.

    python3 profile/fill_profile.py --models /data/flash-next --mtp /data/flash-next/mtp/rt \
        --tokenizer /data/flash-next/tokenizer --vision-exe /opt/strata-v0.2.9/build-vision/bin/strata-vision \
        --logs /var/tmp --out my-profile.json

--models     the folder with the model's files (both IQ3_S shards, or the four UD-Q4_K_XL parts, or Edge0's four files in
             edge0-145b/ beside the IQ3_S's second shard) and mmproj-Qwen3.8-Flash-Next-BF16.gguf
--model      iq3_s (default): ISTA-DASLab's IQ3_S, flash-next-iq3s-512k.template.json; q4_k_xl: Unsloth's UD-Q4_K_XL,
             flash-next-q4_k_xl-512k.template.json (README, "Unsloth's UD-Q4_K_XL"; its 256k one with
             --template profile/flash-next-q4_k_xl-256k.template.json); edge0: Edge0's Edge0-145B-NVFP4,
             flash-next-edge0-512k.template.json (README, "Edge0-145B")
--mtp        the MTP draft layer's runtime folder (dense.bin, dense.txt, experts.bin; README, "The MTP draft layer")
--tokenizer  the tokenizer folder (vocab.json, merges.txt, chat_template.jinja ...; README, "The tokenizer folder")
--vision-exe Strata's strata-vision; leave it out to run without pictures (the "vision" block and --vision go)
--pictures-gpu the picture encoder on the first card of --gpu (vision.cuda_device), only while pictures come: the engine's --vram-elastic
             ("vram_elastic": true) and the vision block's "gpu", "elastic", "elastic_mib" 1623 (the encoder's 1,367 MiB
             peak with ISTA-DASLab's BF16 projector at 1,024 picture tokens, plus 256) and "elastic_idle_s" 120. The first
             picture lends that much of each card's expert slots to the driver and starts the encoder (a few seconds); 120 s
             after the last one the encoder stops and the slots come back (docs/GUIDE.md, "Pictures on a card"). Needs
             --vision-exe, the default 96 GB mode
--cpu-experts FNX_CPU_EXPERTS=1: a decode window's routed experts that are not in VRAM computed on the CPU from their RAM copies
             while the cards compute the ones that are, instead of copied to the cards first (docs/GUIDE.md, "Missing experts
             on the CPU"). --model q4_k_xl in the default 96 GB mode only: the CPU path decodes that file's expert types
             (Q4_K, Q5_K, Q5_1, Q8_0; IQ4_NL too), and needs every expert's copy in RAM. Each card's copies then sit on the
             memory of the CPU the card hangs from (FNX_NUMA_EXPERTS). Not bit for bit the same answers: the CPU sums in
             fp32 where the cards' kernels sum in fp16
--route-prior FNX_ROUTE_PRIOR: 0 (default) is exact; 0.15 is our production setting, faster and lossy (README)
--draft-vocab another draft vocabulary than engine/data/draft_vocab_sl.bin, e.g. engine/data/draft_vocab_en.bin
--exe        the engine, if build.sh was run with another build directory (default build/split/strata-split-2; with
             --ram 32, and --ram 64 with --model q4_k_xl, the low-RAM engine, build-32g/split/strata-split-2 from
             'bash build.sh --32g')
--gpu        the two P100 as nvidia-smi numbers them (default "0,1")
--ram        96 (default): the template as it is; 64: the 64 GB mode. With IQ3_S (README, "64 GB of RAM"): FNX_PLE_RAM=0,
             FNX_EXPERT_DYN_GIB=13, --max-context 262144 without YaRN, --conversation-cache-mib 2000. With --model q4_k_xl
             (docs/GUIDE.md, "UD-Q4_K_XL on 64 GB"; exact, not with --fast): FNX_SSD_EXPERTS=1, FNX_SSD_RAM_GIB (--ssd-ram-gib,
             default 25), FNX_SSD_AHEAD=1, no FNX_EXPERT_DYN_GIB, --conversation-cache-mib 1000, the 524,288 positions kept.
             32: the 32 GB mode (docs/GUIDE.md, "32 GB of RAM"; --model q4_k_xl with --coder or --fast): FNX_SSD_EXPERTS=1
             (no expert pinned in RAM, the Coder's experts the cards do not hold in a RAM tier filled from the model's files),
             FNX_SSD_RAM_GIB (--ssd-ram-gib), FNX_PF_TAIL_MIB=2700, no FNX_EXPERT_DYN_GIB, --conversation-cache-mib 1500.
             With --coder alone the profile's routing stays exact (λ 0): the Coder's answers of 96 GB, bit for bit
--ssd-ram-gib the low-RAM engine's RAM tier, at most GiB a card, sized by need below that (default 9.5 with --ram 32, of
             which the Coder's experts take 8.5; 25 with --ram 64 --model q4_k_xl)
--coder      FNX_EXPERT_MASK=engine/data/coder_mask.txt: the router picks only from the 256 experts a layer that
             ISTA-DASLab's Coder release keeps (README, "Coder"); faster, a different model than the full one (lossy against
             it), for code
--fast       the fast UD-Q4_K_XL (docs/GUIDE.md, "Fast UD-Q4_K_XL"; --model q4_k_xl only): the Coder mask as --coder plus
             FNX_ROUTE_PRIOR=1.0 and FNX_ONEHOP=2. LOSSY: answers are worded differently from the exact profile and
             instructions in other languages than English may be ignored; for code and English

Every path the config names is checked; a missing one is reported and nothing is written. Anything else (the context
length, other switches) is edited in the written JSON; docs/SWITCHES.md lists the switches.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
# per model: its template and its files (the first is --native; the engine finds the others beside it)
MODELS = {
    "iq3_s": ("flash-next-iq3s-512k.template.json",
              [f"Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{i}-of-00002.gguf" for i in (1, 2)]),
    "q4_k_xl": ("flash-next-q4_k_xl-512k.template.json",
                [f"Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf" for i in (1, 2, 3, 4)]),
    # Edge0-145B: the skeleton GGUF is --native, the latent experts and D/U beside it (FNX_LATENT), the lookup table from the
    # IQ3_S's second shard (FNX_LATENT_PLE, checked below)
    "edge0": ("flash-next-edge0-512k.template.json",
              ["model-skeleton.gguf", "experts.bin", "latent_dense_q8.bin", "labels.bin"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True)
    ap.add_argument("--model", choices=sorted(MODELS), default="iq3_s")
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vision-exe")
    ap.add_argument("--pictures-gpu", action="store_true",
                    help="the picture encoder on the first card of --gpu only while pictures come (--vram-elastic; docs/GUIDE.md, "
                         "\"Pictures on a card\")")
    ap.add_argument("--cpu-experts", action="store_true",
                    help="FNX_CPU_EXPERTS=1: a window's experts that are not in VRAM computed on the CPU (--model q4_k_xl, 96 GB mode)")
    ap.add_argument("--logs", default="/var/tmp")
    ap.add_argument("--route-prior", default="0")
    ap.add_argument("--draft-vocab", help="another draft vocabulary (default engine/data/draft_vocab_sl.bin)")
    ap.add_argument("--exe", help="the engine (default build/split/strata-split-2; with --ram 32, and --ram 64 with q4_k_xl, "
                                  "build-32g/split/strata-split-2)")
    ap.add_argument("--gpu", default="0,1")
    ap.add_argument("--ram", choices=["96", "64", "32"], default="96",
                    help="64: the 64 GB mode (README, \"64 GB of RAM\"; with q4_k_xl docs/GUIDE.md, \"UD-Q4_K_XL on 64 GB\"); "
                         "32: the 32 GB mode (--model q4_k_xl with --coder or --fast; docs/GUIDE.md, \"32 GB of RAM\")")
    ap.add_argument("--ssd-ram-gib", type=float,
                    help="the RAM tier of the low-RAM engine, at most GiB a card (FNX_SSD_RAM_GIB; default 9.5 with --ram 32, "
                         "25 with --ram 64 --model q4_k_xl)")
    ap.add_argument("--coder", action="store_true", help="only the Coder release's experts (README, \"Coder\")")
    ap.add_argument("--fast", action="store_true",
                    help="the fast UD-Q4_K_XL: Coder mask, FNX_ROUTE_PRIOR=1.0, FNX_ONEHOP=2 (lossy; docs/GUIDE.md, \"Fast UD-Q4_K_XL\")")
    ap.add_argument("--template", help="another template than the model's")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if float(a.route_prior) < 0:
        sys.exit("--route-prior: 0 or more")
    if a.ram == "64" and a.model == "edge0":
        sys.exit("--ram 64: not with --model edge0 (its profile needs 96 GB)")
    if a.ram == "64" and a.fast:
        sys.exit("--ram 64: the exact UD-Q4_K_XL only; the fast profile below 96 GB is --ram 32 (it runs on 64 GB too)")
    if a.ram == "32" and not (a.model == "q4_k_xl" and (a.coder or a.fast)):
        sys.exit("--ram 32: only with --model q4_k_xl and --coder or --fast (the Coder's experts fit, the full model's do not; "
                 "docs/GUIDE.md, \"32 GB of RAM\")")
    low = a.ram == "32" or (a.ram == "64" and a.model == "q4_k_xl")   # the low-RAM engine (build.sh --32g, FNX_SSD_EXPERTS)
    if a.ssd_ram_gib is None:
        a.ssd_ram_gib = 9.5 if a.ram == "32" else 25.0
    elif not low:
        sys.exit("--ssd-ram-gib: only with --ram 32, or --ram 64 --model q4_k_xl")
    if not 1 <= a.ssd_ram_gib <= 64:
        sys.exit("--ssd-ram-gib: GiB a card, 1 to 64")
    if a.coder and a.model == "edge0":
        sys.exit("--coder: not with --model edge0 (the Coder's list names Flash-Next's experts, not Edge0's latent ones)")
    if a.fast and a.model != "q4_k_xl":
        sys.exit("--fast: only with --model q4_k_xl (the settings were measured on UD-Q4_K_XL only)")
    if a.fast and a.route_prior != "0":
        sys.exit("--fast sets FNX_ROUTE_PRIOR=1.0 itself: leave --route-prior out")
    cards = a.gpu.split(",")
    if len(cards) != 2 or not all(c.strip().isdigit() for c in cards) or cards[0].strip() == cards[1].strip():
        sys.exit("--gpu: two different card numbers, e.g. 0,1")
    template, files = MODELS[a.model]
    if not a.exe:   # the low-RAM modes have an engine of their own (build.sh --32g)
        a.exe = str(RELEASE / ("build-32g" if low else "build") / "split" / "strata-split-2")
    values = {"{{RELEASE}}": str(RELEASE), "{{EXE}}": str(Path(a.exe).resolve()),
              "{{MODELS}}": str(Path(a.models).resolve()), "{{MTP}}": str(Path(a.mtp).resolve()),
              "{{TOKENIZER}}": str(Path(a.tokenizer).resolve()), "{{LOGS}}": str(Path(a.logs).resolve()),
              "{{VISION_EXE}}": str(Path(a.vision_exe).resolve()) if a.vision_exe else "", "{{ROUTE_PRIOR}}": a.route_prior}

    def fill(x):
        if isinstance(x, str):
            for k, v in values.items():
                x = x.replace(k, v)
            return x
        if isinstance(x, list):
            return [fill(v) for v in x]
        if isinstance(x, dict):
            return {k: fill(v) for k, v in x.items()}
        return x

    cfg = fill(json.loads(Path(a.template or HERE / template).read_text(encoding="utf-8")))
    cfg["gpu"] = ",".join(c.strip() for c in cards)
    if a.draft_vocab:
        cfg["args"][cfg["args"].index("--draft-vocab") + 1] = str(Path(a.draft_vocab).resolve())
    if not a.vision_exe:
        cfg.pop("vision")
        cfg["args"].remove("--vision")
    if a.pictures_gpu:
        if not a.vision_exe:
            sys.exit("--pictures-gpu needs --vision-exe")
        if a.ram != "96":
            sys.exit("--pictures-gpu: measured with the 96 GB mode only (README, docs/GUIDE.md \"Pictures on a card\")")
        # 21 Nov 2026 (docs/GUIDE.md, "Pictures on a card"): the encoder on the first of the engine's cards while pictures come;
        # the engine lends each card's coldest expert slots to the driver for it (--vram-elastic), and takes them back 120 s
        # after the last picture. cuda_device is numbered as nvidia-smi numbers the cards, as "gpu" is
        cfg["vram_elastic"] = True
        cfg["vision"].update({"gpu": True, "cuda_device": int(cards[0]), "elastic": True, "elastic_mib": 1623,
                              "elastic_idle_s": 120})
    if a.ram == "64" and a.model == "q4_k_xl":
        # the exact UD-Q4_K_XL's 64 GB mode of 19 Nov 2026 (docs/GUIDE.md, "UD-Q4_K_XL on 64 GB"): the profile's own settings
        # (lambda 0, 524,288 positions) on the low-RAM engine; no expert pinned in RAM, the experts the cards do not hold (52.1 GiB)
        # in a RAM tier of at most --ssd-ram-gib a card (sized by need) and in the model's files; a prompt ring of two layers;
        # parked conversations get 1000 MiB
        cfg["env"]["FNX_SSD_EXPERTS"] = "1"
        cfg["env"]["FNX_SSD_RAM_GIB"] = f"{a.ssd_ram_gib:g}"
        cfg["env"]["FNX_SSD_AHEAD"] = "1"
        cfg["env"].pop("FNX_EXPERT_DYN_GIB", None)
        cfg["args"][cfg["args"].index("--conversation-cache-mib") + 1] = "1000"
    elif a.ram == "64":
        # the lookup table read through the page cache, only the coldest 13 GiB of the experts in VRAM with a RAM copy; at 524,288
        # positions the K/V pool would need nearly every expert slot, so the trained 262,144 (no YaRN) and a smaller conversation cache
        cfg["env"]["FNX_PLE_RAM"] = "0"
        cfg["env"]["FNX_EXPERT_DYN_GIB"] = "13"
        cfg["args"][cfg["args"].index("--max-context") + 1] = "262144"
        cfg["args"][cfg["args"].index("--conversation-cache-mib") + 1] = "2000"
        for k in ("FNX_ROPE_SCALE", "FNX_YARN_ORIG_CTX"):
            cfg["env"].pop(k)
    if a.cpu_experts:
        # 22 Nov 2026 (docs/GUIDE.md, "Missing experts on the CPU"): the engine turns FNX_NUMA_EXPERTS on with it
        if a.model != "q4_k_xl":
            sys.exit("--cpu-experts: with --model q4_k_xl only (the CPU path has decoders for Q4_K, Q5_K, Q5_1, Q8_0 and IQ4_NL experts, "
                     "not for IQ3_S or Edge0's)")
        if low or a.ram != "96":
            sys.exit("--cpu-experts: the default 96 GB mode only (it needs every routed expert's copy in RAM)")
        cfg["env"]["FNX_CPU_EXPERTS"] = "1"
    if a.coder or a.fast:
        cfg["env"]["FNX_EXPERT_MASK"] = str(RELEASE / "engine" / "data" / "coder_mask.txt")
    if a.fast:
        # our fast profile of 19 Nov 2026 (docs/GUIDE.md, "Fast UD-Q4_K_XL"): λ 1.0 prefers the experts already on the cards
        # (higher was no faster), FNX_ONEHOP=2 sums the two cards' halves in one pass
        cfg["env"]["FNX_ROUTE_PRIOR"] = "1.0"
        cfg["env"]["FNX_ONEHOP"] = "2"
    if a.ram == "32":
        # the 32 GB mode of 19 Nov 2026 (docs/GUIDE.md, "32 GB of RAM"; with --coder alone measured on 21 Nov): no expert pinned in
        # RAM; the Coder's experts the cards do not hold sit in a RAM tier of --ssd-ram-gib a card, the rest is read from the
        # model's files; the prompt path borrows no more expert slots than it uses (2700 of 4096 MiB), and parked conversations
        # get 1500 MiB. Where an expert's bytes come from does not change what is computed: λ stays the profile's
        cfg["env"]["FNX_SSD_EXPERTS"] = "1"
        cfg["env"]["FNX_SSD_RAM_GIB"] = f"{a.ssd_ram_gib:g}"
        cfg["env"]["FNX_PF_TAIL_MIB"] = "2700"
        cfg["env"].pop("FNX_EXPERT_DYN_GIB", None)
        cfg["args"][cfg["args"].index("--conversation-cache-mib") + 1] = "1500"

    native = Path(cfg["args"][cfg["args"].index("--native") + 1])
    need = [str(native.with_name(f)) for f in files] + [
            cfg["args"][cfg["args"].index("--draft-vocab") + 1],
            cfg["env"]["FNX_EXPERT_PROFILE"], cfg["tokenizer"] + "/vocab.json", cfg["tokenizer"] + "/chat_template.jinja",
            cfg["args"][cfg["args"].index("--mtp") + 1] + "/dense.bin",
            cfg["args"][cfg["args"].index("--mtp") + 1] + ("/experts-q4_0.bin" if cfg["env"].get("FNX_MTP_EXPERTS") == "q4_0" else "/experts.bin"),
            cfg["exe"]]
    if "vision" in cfg:
        need += [cfg["vision"]["exe"], cfg["vision"]["mmproj"]]
    if "FNX_EXPERT_MASK" in cfg["env"]:
        need.append(cfg["env"]["FNX_EXPERT_MASK"])
    if "FNX_LATENT_PLE" in cfg["env"]:
        need.append(cfg["env"]["FNX_LATENT_PLE"])
    missing = [p for p in need if not Path(p).exists()]
    if missing:
        sys.exit("missing, nothing written:\n  " + "\n  ".join(missing))
    Path(a.out).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({a.model}, FNX_ROUTE_PRIOR={cfg['env']['FNX_ROUTE_PRIOR']}, pictures {('on card ' + str(cfg['vision'].get('cuda_device', 0)) + ' while they come' if cfg['vision'].get('elastic') else 'on') if 'vision' in cfg else 'off'}, "
          f"{a.ram} GB of RAM{f' (a RAM tier of at most {a.ssd_ram_gib:g} GiB a card)' if low else ''}"
          f"{', the Coder experts' if a.coder and not a.fast else ''}"
          f"{', FAST (lossy: Coder mask, λ 1.0, FNX_ONEHOP=2)' if a.fast else ''})")


if __name__ == "__main__":
    main()
