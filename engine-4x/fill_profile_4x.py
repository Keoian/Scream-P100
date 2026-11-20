#!/usr/bin/env python3
"""UNVERIFIED PROOF OF CONCEPT, its answers on four cards are not verified: writes a run config for the 4-card engine (bash build.sh --4x).

    python3 engine-4x/fill_profile_4x.py --models /data/flash-next/models --mtp /data/flash-next/prep/mtp-rt \
        --tokenizer /data/flash-next/prep/tokenizer --logs /data/flash-next/logs --out /data/flash-next/profile-4x.json
    bash run.sh /data/flash-next --config /data/flash-next/profile-4x.json

It fills one of the IQ3_S templates beside this script through profile/fill_profile.py (which fills in and checks every path,
and writes nothing when one is missing), then names the four cards. ISTA-DASLab's IQ3_S only: UD-Q4_K_XL's experts (72 GiB) would
leave too many of them in RAM behind a shared PCIe 3.0 x4 link, and Edge0-145B runs on two cards only (docs/POC-4X.md).

--context    256k (default): 262,144 positions, the model's own, no YaRN; the K/V pool, which grows into the experts' VRAM slots
             as a conversation gets longer (FNX_ARENA), takes at most half of what 512k takes from them, about 1.5 GiB a card.
             512k: 524,288 positions with YaRN x2 (our production's), up to about 3 GiB a card. 128k: 131,072, no YaRN
--gpu        the four P100 as nvidia-smi numbers them (default "0,1,2,3")
--lib-dir    a folder of the CUDA 12 libraries the engine was built with (libcudart, libcublas, libcublasLt; may be given more
             than once). Without it the config's lib_dirs are the folders where ldd finds them, else the toolkit the build used
             (build-4x's CMakeCache.txt), CUDA_HOME, CUDA_PATH, /usr/local/cuda*. KaneQuah (issue #1): an empty lib_dirs left
             his CUDA 12.9 libraries unfound
--exe        the engine (default build-4x/split/strata-split-4)
--models, --mtp, --tokenizer, --vision-exe, --logs, --route-prior, --draft-vocab: as profile/fill_profile.py
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
CUDA_LIBS = ("libcudart.so", "libcublas.so", "libcublasLt.so")


def cuda_lib_dirs(exe, given):
    """(folders for lib_dirs, libraries found nowhere): the CUDA libraries the engine needs, where ldd finds them, else in
    --lib-dir, the build's toolkit, CUDA_HOME, CUDA_PATH or /usr/local/cuda*"""
    try:
        out = subprocess.run(["ldd", exe], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    need, dirs = {}, []
    for ln in out.splitlines():
        m = re.match(r"\s*(\S+)\s+=>\s+(.*?)\s*(\(0x[0-9a-f]+\))?$", ln)
        if m and m.group(1).startswith(CUDA_LIBS):
            need[m.group(1)] = None if m.group(2) == "not found" else m.group(2)
    if not need:   # no ldd here, or it could not read the binary: the usual sonames of a CUDA 12 build
        need = {"libcudart.so.12": None, "libcublas.so.12": None, "libcublasLt.so.12": None}
    cands = list(given)
    for d in [Path(exe).resolve().parent, *Path(exe).resolve().parents][:4]:   # the build's CMakeCache.txt: the toolkit it used
        cache = d / "CMakeCache.txt"
        if cache.exists():
            for ln in cache.read_text(encoding="utf-8", errors="replace").splitlines():
                if ln.startswith(("CMAKE_CUDA_COMPILER:", "CUDAToolkit_LIBRARY_DIR:")):
                    v = os.path.realpath(ln.split("=", 1)[1].strip())
                    root = Path(v).parent.parent if ln.startswith("CMAKE_CUDA_COMPILER:") else None
                    cands += [v] if root is None else [str(root / "lib64"), str(root / "targets" / "x86_64-linux" / "lib")]
            break
    for e in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(e):
            cands += [os.path.join(os.environ[e], "lib64"), os.path.join(os.environ[e], "targets", "x86_64-linux", "lib")]
    cands += ["/usr/local/cuda/lib64"] + sorted(glob.glob("/usr/local/cuda-12*/lib64"), reverse=True)
    missing = []
    for lib, path in need.items():
        d = os.path.dirname(path) if path else next((c for c in cands if os.path.exists(os.path.join(c, lib))), None)
        if d is None:
            missing.append(lib)
        elif d not in dirs:
            dirs.append(d)
    for g in given:   # what the user named comes first, whether or not a library was looked up there
        if g in dirs:
            dirs.remove(g)
    return list(given) + dirs, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vision-exe")
    ap.add_argument("--logs", default="/var/tmp")
    ap.add_argument("--route-prior", default="0")
    ap.add_argument("--draft-vocab")
    ap.add_argument("--exe", default=str(RELEASE / "build-4x" / "split" / "strata-split-4"))
    ap.add_argument("--gpu", default="0,1,2,3")
    ap.add_argument("--lib-dir", action="append", default=[])
    ap.add_argument("--context", choices=["128k", "256k", "512k"], default="256k")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cards = [c.strip() for c in a.gpu.split(",")]
    if len(cards) != 4 or not all(c.isdigit() for c in cards) or len(set(cards)) != 4:
        sys.exit("--gpu: four different card numbers, e.g. 0,1,2,3")
    template = HERE / f"flash-next-iq3s-4x-{'512k' if a.context == '512k' else '256k'}.template.json"
    out = Path(a.out)
    tmp = out.with_name(out.name + ".fill-tmp")
    cmd = [sys.executable, str(RELEASE / "profile" / "fill_profile.py"), "--models", a.models, "--mtp", a.mtp, "--tokenizer", a.tokenizer,
           "--logs", a.logs, "--route-prior", a.route_prior, "--exe", a.exe, "--gpu", "0,1", "--template", str(template), "--out", str(tmp)]
    if a.vision_exe:
        cmd += ["--vision-exe", a.vision_exe]
    if a.draft_vocab:
        cmd += ["--draft-vocab", a.draft_vocab]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL)   # its own "wrote" line would name two cards; its errors still show
    if r.returncode != 0:
        sys.exit(r.returncode)
    cfg = json.loads(tmp.read_text(encoding="utf-8"))
    os.remove(tmp)
    cfg["gpu"] = ",".join(cards)
    cfg["lib_dirs"], missing = cuda_lib_dirs(cfg["exe"], a.lib_dir)
    if missing:
        print(f"WARNING: {', '.join(missing)} found neither by ldd nor in {', '.join(a.lib_dir) or 'the usual CUDA folders'}: "
              "name their folder with --lib-dir (the CUDA 12 toolkit's lib64)", file=sys.stderr)
    if a.context == "128k":
        cfg["args"][cfg["args"].index("--max-context") + 1] = "131072"
    out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: UNVERIFIED 4-card proof of concept, IQ3_S, cards {cfg['gpu']}, {a.context}, lib_dirs {cfg['lib_dirs']}, "
          f"FNX_ROUTE_PRIOR={cfg['env']['FNX_ROUTE_PRIOR']}, pictures {'on' if 'vision' in cfg else 'off'}, engine {cfg['exe']}")


if __name__ == "__main__":
    main()
