#!/bin/bash
# setup.sh DATA [options]: everything the engine needs besides itself, in the folder DATA (run build.sh first):
#
#   DATA/strata-v0.2.9    Strata's server and tools at v0.2.9, with their Python environment (.venv)
#   DATA/models           the model's two IQ3_S shards (or with --model q4_k_xl the four UD-Q4_K_XL parts, or with --model edge0
#                         Edge0's four files in edge0-145b/ and the IQ3_S's second shard) and the picture projector (85 GB,
#                         112 GB, or 87 GB), checked against SHA-256
#   DATA/prep/tokenizer   the tokenizer folder, from the GGUF's metadata (Strata's tools/strata_tokenizer.py)
#   DATA/prep/mtp-rt      the MTP draft layer: 5.2 GB read from Qwen's checkpoint and packed (Strata's tools/mtp_*.py)
#   DATA/strata-v0.2.9/build-vision/bin/strata-vision   the picture encoder (not with --no-pictures)
#   DATA/profile.json     the run config (profile/fill_profile.py); an existing one is kept
#
# A step whose result is already there is skipped, so after an interruption it is simply run again.
#
#   bash setup.sh /data/flash-next
#   bash setup.sh /data/flash-next --check                    # the checks and what is still to do; changes nothing
#   bash setup.sh /data/flash-next --models /mnt/models --no-pictures --route-prior 0.15
#   bash setup.sh /data/flash-next --ram 64                   # a machine with 64 GB of RAM (README, "64 GB of RAM")
#   bash setup.sh /data/flash-next-q4 --model q4_k_xl         # Unsloth's UD-Q4_K_XL (README, "Unsloth's UD-Q4_K_XL")
#   bash setup.sh /data/edge0 --model edge0                   # Edge0's Edge0-145B-NVFP4 (README, "Edge0-145B")
#   bash setup.sh /data/flash-next-coder --models /data/flash-next/models --coder   # the Coder's experts only (README, "Coder")
#   bash setup.sh /data/flash-next-q4-fast --models /data/flash-next-q4/models --model q4_k_xl --fast   # the fast UD-Q4_K_XL, lossy
#   bash setup.sh /data/flash-next-q4-32 --models /data/flash-next-q4/models --model q4_k_xl --fast --ram 32   # the same with 32 GB of RAM
#   bash setup.sh /data/flash-next-q4-coder-32 --models /data/flash-next-q4/models --model q4_k_xl --coder --ram 32   # the Coder, 32 GB
#   bash setup.sh /data/flash-next-q4-64 --models /data/flash-next-q4/models --model q4_k_xl --ram 64   # the exact UD-Q4_K_XL, 64 GB
#
# Options:
#   --model M        iq3_s (default): ISTA-DASLab's GSQ-RCO IQ3_S; q4_k_xl: Unsloth's UD-Q4_K_XL (512k, needs 96 GB of RAM, or
#                    64 GB with --ram 64); edge0: Edge0's Edge0-145B-NVFP4 (512k, needs 96 GB of RAM)
#   --models DIR     the model files somewhere else (for example already downloaded); missing ones are downloaded there
#   --strata DIR     an existing Strata checkout at v0.2.9 instead of DATA/strata-v0.2.9
#   --no-pictures    no projector download, no encoder build, a profile without pictures
#   --pictures-gpu   the picture encoder on the first of the two cards only while pictures come (the engine's --vram-elastic;
#                    docs/GUIDE.md, "Pictures on a card"): a 4K picture in seconds instead of on the CPU, each card lends
#                    1.6 GiB of expert slots while the encoder runs. The default 96 GB mode only
#   --cpu-experts    FNX_CPU_EXPERTS=1 in the profile (--model q4_k_xl, the default 96 GB mode; docs/GUIDE.md, "Missing experts
#                    on the CPU"): a decode window's experts that are not in VRAM are computed on the CPU from their RAM
#                    copies while the cards compute the ones that are. Faster writing on cards that hold a small part of
#                    the experts; needs AVX2 and free cores (16 threads a card by default)
#   --route-prior X  FNX_ROUTE_PRIOR in the profile: 0 (default) is exact; 0.15 is our production setting, faster, lossy
#   --ram 64         the 64 GB mode in the profile; default 96. IQ3_S: 262,144 positions, the lookup table from the disk.
#                    UD-Q4_K_XL (exact, not with --fast; docs/GUIDE.md, "UD-Q4_K_XL on 64 GB"): 524,288 positions, no expert
#                    pinned in RAM, the experts the cards do not hold in a RAM tier of 25 GiB a card and in the model's files
#                    (keep them on an SSD); the low-RAM engine: run 'bash build.sh --32g' first (the profile names build-32g/)
#   --ram 32         the 32 GB mode (--model q4_k_xl with --coder or --fast; docs/GUIDE.md, "32 GB of RAM"): no expert pinned in
#                    RAM, the Coder's experts the cards do not hold in a RAM tier filled from the model's files (FNX_SSD_EXPERTS);
#                    wants the files on an SSD. It has an engine of its own: run 'bash build.sh --32g' first (the profile names
#                    build-32g/). With --coder (and no --fast) it answers as the Coder profile does with 96 GB, bit for bit
#   --coder          route only to the 256 experts a layer that ISTA-DASLab's Coder release keeps (FNX_EXPERT_MASK): faster,
#                    lossy against the full model (it is a different model), for code; with either file
#   --fast           the fast UD-Q4_K_XL (--model q4_k_xl only; docs/GUIDE.md, "Fast UD-Q4_K_XL"): the Coder mask, FNX_ROUTE_PRIOR=1.0
#                    and FNX_ONEHOP=2, about 1.4x fresh writing and 2x code edits. LOSSY: answers are worded differently from
#                    the exact profile, instructions in other languages than English may be ignored; for code and English
#   --gpu A,B        the two P100 as nvidia-smi numbers them; default: the two there are, or with more the pair that hangs
#                    from two CPUs with the least VRAM in use (only when the profile is written; it is kept afterwards)
#   --check          only check and report
# Environment: HF_TOKEN (sent to Hugging Face if set), CUDA, HOSTCXX (as for build.sh), JOBS (encoder build, default 6).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
say() { echo "setup.sh: $*"; }
die() { echo "setup.sh: $*" >&2; exit 1; }

STRATA_REPO=https://github.com/Niko1221/Strata.git
STRATA_TAG=v0.2.9
STRATA_COMMIT=c62d5e3a15b40ea51c96aae9e8d36928965f53ab
ISTA=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09   # Strata's pin
UNSLOTH=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/084c631ac221016019d6e0a863afe92d90edd3b0/UD-Q4_K_XL
EDGE0=https://huggingface.co/Edge0/Edge0-145B-NVFP4/resolve/cf9b3679e7da201a658a694e1730607193f2b032
IQ3_S=(Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf)
Q4_K_XL=(Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
         Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf)
# Edge0: the skeleton GGUF (dense layers, router, tokenizer), the latent experts (NVFP4), D/U (Q8_0) and the experts' groups; its
# lookup table is the IQ3_S's second shard (the same table)
EDGE0_FILES=(edge0-145b/model-skeleton.gguf edge0-145b/experts.bin edge0-145b/latent_dense_q8.bin edge0-145b/labels.bin)
MMPROJ=mmproj-Qwen3.8-Flash-Next-BF16.gguf   # ISTA-DASLab's projector, also for UD-Q4_K_XL and Edge0 (ours runs it with all three)
declare -A SIZE=([${IQ3_S[0]}]=54817524224 [${IQ3_S[1]}]=28800138432
                 [${Q4_K_XL[0]}]=10946618 [${Q4_K_XL[1]}]=49720539552 [${Q4_K_XL[2]}]=32803030176 [${Q4_K_XL[3]}]=28800138432
                 [${EDGE0_FILES[0]}]=5991569248 [${EDGE0_FILES[1]}]=47563800576 [${EDGE0_FILES[2]}]=3743416320 [${EDGE0_FILES[3]}]=98304
                 [$MMPROJ]=907543008)
declare -A SHA=([${IQ3_S[0]}]=4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3
                [${IQ3_S[1]}]=316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
                [${Q4_K_XL[0]}]=acc44040f74d0e55fb89c3967056cce5dbda1bcc52e909c3ba6fc72b5e9c0858
                [${Q4_K_XL[1]}]=45792ae869072d8229384c51c7c73dd959e191505252b41dd71852d9746f5ce4
                [${Q4_K_XL[2]}]=8c76e79d444b16a663d9c8db54e585ed3b1da13eebd2619c37b0c355da803c82
                [${Q4_K_XL[3]}]=5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36
                [${EDGE0_FILES[0]}]=ab49fbe7590cdcda45d7a2bcae4be0bd42c6a70b951d594a00eb158d7df0b72f
                [${EDGE0_FILES[1]}]=a39b0c127983b85d07326c8bad49260d6b475293b46e4166428c91a765ee593a
                [${EDGE0_FILES[2]}]=fbb25caaa0dcf45ef62f1a4945dcf98a53a166934a3b461b166fbafa831d87ca
                [${EDGE0_FILES[3]}]=ce9b4a20b92c5c866e65d84a3884c104986522471cf11cc5f797a153affbcc06
                [$MMPROJ]=b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0)

[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || die "usage: bash setup.sh DATA [--model iq3_s|q4_k_xl|edge0] [--models DIR] [--strata DIR] [--no-pictures] [--pictures-gpu] [--cpu-experts] [--route-prior X] [--ram 96|64|32] [--coder] [--fast] [--gpu A,B] [--check]"
DATA=$1
shift
MODEL=iq3_s MODELS='' STRATA='' PICTURES=1 PICTURES_GPU=0 CPU_EXPERTS=0 PRIOR=0 RAM=96 CODER=0 FAST=0 CHECK=0 GPU=''
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=${2:?--model iq3_s, q4_k_xl or edge0}; case "$MODEL" in iq3_s|q4_k_xl|edge0) ;; *) die "--model: iq3_s, q4_k_xl or edge0" ;; esac; shift 2 ;;
    --models) MODELS=${2:?--models DIR}; shift 2 ;;
    --strata) STRATA=${2:?--strata DIR}; shift 2 ;;
    --no-pictures) PICTURES=0; shift ;;
    --pictures-gpu) PICTURES_GPU=1; shift ;;
    --cpu-experts) CPU_EXPERTS=1; shift ;;
    --route-prior) PRIOR=${2:?--route-prior X}; shift 2 ;;
    --ram) RAM=${2:?--ram 96, 64 or 32}; [ "$RAM" = 96 ] || [ "$RAM" = 64 ] || [ "$RAM" = 32 ] || die "--ram: 96, 64 or 32"; shift 2 ;;
    --coder) CODER=1; shift ;;
    --fast) FAST=1; shift ;;
    --gpu) GPU=${2:?--gpu A,B}; echo "$GPU" | grep -Eq '^[0-9]+,[0-9]+$' && [ "${GPU%,*}" != "${GPU#*,}" ] ||
             die "--gpu: two different card numbers as nvidia-smi shows them, e.g. 1,2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) die "unknown option $1" ;;
  esac
done
[ "$RAM" != 64 ] || [ "$MODEL" != edge0 ] || die "--ram 64: not with Edge0 (its profile needs 96 GB); the IQ3_S file or the UD-Q4_K_XL"
[ "$RAM" != 64 ] || [ "$FAST" = 0 ] || die "--ram 64: the exact UD-Q4_K_XL only; the fast profile below 96 GB is --ram 32 (it runs on 64 GB too)"
[ "$RAM" != 32 ] || { [ "$MODEL" = q4_k_xl ] && [ "$CODER$FAST" != 00 ]; } ||
  die "--ram 32: only with --model q4_k_xl and --coder or --fast (docs/GUIDE.md, \"32 GB of RAM\")"
[ "$CODER" = 0 ] || [ "$MODEL" != edge0 ] || die "--coder: not with --model edge0 (the Coder's list names Flash-Next's experts, not Edge0's)"
[ "$PICTURES_GPU" = 0 ] || [ "$PICTURES" = 1 ] || die "--pictures-gpu: not with --no-pictures"
[ "$CPU_EXPERTS" = 0 ] || { [ "$MODEL" = q4_k_xl ] && [ "$RAM" = 96 ]; } ||
  die "--cpu-experts: with --model q4_k_xl in the default 96 GB mode only (docs/GUIDE.md, \"Missing experts on the CPU\")"
[ "$PICTURES_GPU" = 0 ] || [ "$RAM" = 96 ] || die "--pictures-gpu: the default 96 GB mode only (docs/GUIDE.md, \"Pictures on a card\")"
[ "$FAST" = 0 ] || [ "$MODEL" = q4_k_xl ] || die "--fast: only with --model q4_k_xl (the settings were measured on UD-Q4_K_XL only)"
[ "$FAST" = 0 ] || [ "$PRIOR" = 0 ] || die "--fast sets FNX_ROUTE_PRIOR=1.0 itself: leave --route-prior out"
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }
DATA=$(abspath "$DATA")
MODELS=$(abspath "${MODELS:-$DATA/models}")
STRATA=$(abspath "${STRATA:-$DATA/strata-v0.2.9}")
PREP=$DATA/prep
EXE=$HERE/build/split/strata-split-2
BUILD_CMD="bash build.sh"
if [ "$RAM" = 32 ] || { [ "$RAM" = 64 ] && [ "$MODEL" = q4_k_xl ]; }; then   # the low-RAM engine
  EXE=$HERE/build-32g/split/strata-split-2; BUILD_CMD="bash build.sh --32g"
fi
GG=$HERE/third_party/llama.cpp
declare -A URL=([$MMPROJ]=$ISTA/$MMPROJ)
for f in "${IQ3_S[@]}"; do URL[$f]=$ISTA/IQ3_S/$f; done   # the shards sit in the IQ3_S folder at this revision (found by Kenzu)
for f in "${Q4_K_XL[@]}"; do URL[$f]=$UNSLOTH/$f; done
for f in "${EDGE0_FILES[@]}"; do URL[$f]=$EDGE0/${f#edge0-145b/}; done
case $MODEL in
  iq3_s) NEEDED=("${IQ3_S[@]}") ;;
  q4_k_xl) NEEDED=("${Q4_K_XL[@]}") ;;
  edge0) NEEDED=("${EDGE0_FILES[@]}" "${IQ3_S[1]}") ;;   # the skeleton first: the tokenizer comes from it
esac
SHARD1=${NEEDED[0]}
[ "$PICTURES" = 1 ] && NEEDED+=("$MMPROJ")
TODO=0
todo() { TODO=$((TODO + 1)); say "to do: $*"; }
ok() { say "ok: $*"; }
gb() { awk -v b="$1" 'BEGIN {printf "%.1f GB", b / 1e9}'; }

# ------------------------------------------------------------------ checks
[ -x "$EXE" ] || die "no $EXE: run '$BUILD_CMD' first"
[ -d "$GG/gguf-py/gguf" ] || die "no $GG/gguf-py: run 'bash build.sh' first (it fetches llama.cpp)"
for t in git curl sha256sum python3; do command -v $t >/dev/null || die "$t is needed"; done
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10 or newer is needed"
if grep -qs '^STRATA_SPLIT_PF_JOB:[A-Z]*=64$' "$HERE/build/CMakeCache.txt"; then
  say "note: this engine was built with prompt-path jobs of 64 rows; one built with the current 'bash build.sh' (96 rows) reads long prompts faster (README, \"Build\")"
fi

# the two P100: --gpu, else with more than two the pair that hangs from two CPUs (each card its own link to memory) with the
# least VRAM in use; a user's 4-card machine wrote 45 tok/s on two cards behind one PCIe switch and 58 on one card per CPU
pick_pair() {   # lines "index used numa" -> "i,j"
  awk 'BEGIN {n = 0}
       {ix[n] = $1; used[n] = $2; node[n] = $3; n++}
       END {best = -1
            for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) {
              s = (node[i] == node[j] || node[i] < 0 ? 1000000 : 0) + (used[i] > used[j] ? used[i] : used[j])
              if (best < 0 || s < best) {best = s; pair = ix[i] "," ix[j]}
            }
            print pair}'
}
GPU_LIST=0,1
if [ -n "$GPU" ]; then
  GPU_LIST=$GPU
  ok "two Tesla P100: cards $GPU_LIST (--gpu)"
elif command -v nvidia-smi >/dev/null; then
  p100=$( (nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used --format=csv,noheader,nounits || true) | awk -F', ' '/P100/ {print $1, $3, $4}' |
    while read -r i bus used; do
      bdf=$(echo "$bus" | tr 'A-F' 'a-f'); bdf=${bdf: -12}
      echo "$i $used $(cat "${SYSFS:-/sys}/bus/pci/devices/$bdf/numa_node" 2>/dev/null || echo -1)"
    done)
  n=$(echo "$p100" | grep -c . || true)
  if [ "$n" = 2 ]; then
    GPU_LIST=$(echo "$p100" | awk '{print $1}' | paste -sd, -)
    ok "two Tesla P100: cards $GPU_LIST"
  elif [ "$n" -gt 2 ]; then
    GPU_LIST=$(echo "$p100" | pick_pair)
    ok "two Tesla P100: cards $GPU_LIST of $n ($(echo "$p100" | awk '{printf "%s%s: NUMA node %s, %s MiB in use", (NR > 1 ? "; " : ""), $1, $3, $2}'))"
    say "note: the engine runs on two cards; from two CPUs (or at least two PCIe switches) they copy faster; --gpu A,B picks them (docs/GUIDE.md, \"Other machines\")"
  else
    found=$(echo "$p100" | awk '{print $1}' | paste -sd, -)
    say "WARNING: nvidia-smi does not show two Tesla P100 (found: ${found:-none}); the profile names cards 0,1"
  fi
else
  say "WARNING: no nvidia-smi; the profile names cards 0,1"
fi
mem_gib=$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)
if [ "$RAM" = 96 ] && [ "$MODEL" = q4_k_xl ] && [ "$mem_gib" -lt 88 ]; then
  say "WARNING: $mem_gib GiB of RAM; this profile needs 96 GB; with 64 GB use --ram 64 (with --coder or --fast: --ram 32)"
elif [ "$RAM" = 96 ] && [ "$MODEL" != iq3_s ] && [ "$mem_gib" -lt 88 ]; then
  say "WARNING: $mem_gib GiB of RAM; this model needs 96 GB; with 64 GB use the IQ3_S file and --ram 64"
elif [ "$RAM" = 96 ] && [ "$mem_gib" -lt 88 ]; then
  say "WARNING: $mem_gib GiB of RAM; this profile needs 96 GB (about 85 GiB once loaded, more while loading); with 64 GB use --ram 64"
elif [ "$RAM" = 64 ] && [ "$mem_gib" -lt 60 ]; then
  say "WARNING: $mem_gib GiB of RAM; the 64 GB mode needs 64 GB"
elif [ "$RAM" = 32 ] && [ "$mem_gib" -lt 30 ]; then
  say "WARNING: $mem_gib GiB of RAM; the 32 GB mode needs 32 GB (and FNX_SSD_RAM_GIB lower than its 9.5 with less)"
else
  ok "$mem_gib GiB of RAM"
fi

# disk: what the missing model files and the MTP preparation still need
need=0
for f in "${NEEDED[@]}"; do
  have=$(stat -L -c %s "$MODELS/$f" 2>/dev/null || stat -L -c %s "$MODELS/$f.part" 2>/dev/null || echo 0)
  need=$((need + ${SIZE[$f]} - have))
done
[ -f "$PREP/mtp-rt/experts.bin" ] || need=$((need + 7 * 1024 * 1024 * 1024))
[ -f "$PREP/mtp-rt/experts-q4_0.bin" ] || need=$((need + 2 * 1024 * 1024 * 1024))
avail_on() { local d=$1; while [ ! -d "$d" ]; do d=$(dirname "$d"); done; df -Pk "$d" | awk 'NR == 2 {print $4 * 1024}'; }
if [ "$need" -gt 0 ]; then
  for d in "$MODELS" "$PREP"; do
    free=$(avail_on "$d")
    [ "$free" -ge "$need" ] || say "WARNING: about $(gb "$need") still to write, $(gb "$free") free at $d"
  done
fi

# ------------------------------------------------------------------ Strata v0.2.9 and its Python environment
strata_ok() { git -C "$STRATA" rev-parse -q --verify HEAD >/dev/null 2>&1; }
if strata_ok && [ -f "$STRATA/.venv/.setup-ok" ]; then
  ok "Strata at $STRATA"
elif [ "$CHECK" = 1 ]; then
  todo "Strata $STRATA_TAG into $STRATA, and its .venv"
else
  if ! strata_ok; then
    [ ! -e "$STRATA" ] || [ -z "$(ls -A "$STRATA")" ] || die "$STRATA exists and is not a git checkout"
    say "cloning Strata $STRATA_TAG into $STRATA"
    git clone -q --depth 1 --branch "$STRATA_TAG" "$STRATA_REPO" "$STRATA"
  fi
  say "Strata's Python environment"
  python3 -m venv "$STRATA/.venv"
  "$STRATA/.venv/bin/pip" install -q -r "$STRATA/requirements.txt"
  touch "$STRATA/.venv/.setup-ok"
fi
if strata_ok; then
  head=$(git -C "$STRATA" rev-parse HEAD)
  [ "$head" = "$STRATA_COMMIT" ] || die "$STRATA is at $head, not Strata $STRATA_TAG ($STRATA_COMMIT)"
fi

# ------------------------------------------------------------------ the model files
verified() {   # name: already checked at this size and time
  local f=$MODELS/$1
  grep -qxF "$1 $(stat -L -c '%s %Y' "$f")" "$MODELS/.verified" 2>/dev/null
}
check_sha() {   # name path: the file's SHA-256 against the table (a few minutes for a shard)
  say "checking $1 (SHA-256)"
  [ "$(sha256sum "$2" | cut -d' ' -f1)" = "${SHA[$1]}" ]
}
for f in "${NEEDED[@]}"; do
  p=$MODELS/$f
  if [ -f "$p" ] && [ "$(stat -L -c %s "$p")" = "${SIZE[$f]}" ] && verified "$f"; then
    ok "$f"
    continue
  fi
  if [ "$CHECK" = 1 ]; then
    if [ -f "$p" ]; then todo "check $p"; else todo "download $f ($(gb "${SIZE[$f]}"))"; fi
    continue
  fi
  mkdir -p "$(dirname "$p")"
  if [ ! -f "$p" ]; then
    part=$p.part
    if [ "$(stat -L -c %s "$part" 2>/dev/null || echo 0)" != "${SIZE[$f]}" ]; then
      say "downloading $f ($(gb "${SIZE[$f]}")) from ${URL[$f]%/*}"
      auth=()
      [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
      curl -L --fail --retry 5 --retry-delay 20 -C - ${auth[@]+"${auth[@]}"} -o "$part" \
        "${URL[$f]}" || die "download of $f failed (run again to resume)"
    fi
    [ "$(stat -L -c %s "$part")" = "${SIZE[$f]}" ] || die "$part has $(stat -L -c %s "$part") bytes, not ${SIZE[$f]}"
    check_sha "$f" "$part" || die "$part does not match its SHA-256; delete it and run again"
    mv "$part" "$p"
  else
    [ "$(stat -L -c %s "$p")" = "${SIZE[$f]}" ] || die "$p has $(stat -L -c %s "$p") bytes, not ${SIZE[$f]}"
    check_sha "$f" "$p" || die "$p does not match its SHA-256 (delete it to download it again)"
  fi
  echo "$f $(stat -L -c '%s %Y' "$p")" >> "$MODELS/.verified"
  ok "$f"
done

# ------------------------------------------------------------------ the tokenizer folder and the MTP draft layer
run_tool() { (cd "$STRATA" && STRATA_GGUF_PY="$GG/gguf-py" .venv/bin/python "$@"); }
if [ -f "$PREP/tokenizer/vocab.json" ] && [ -f "$PREP/tokenizer/chat_template.jinja" ]; then
  ok "tokenizer at $PREP/tokenizer"
elif [ "$CHECK" = 1 ]; then
  todo "the tokenizer folder $PREP/tokenizer"
else
  say "the tokenizer folder"
  mkdir -p "$PREP"
  run_tool tools/strata_tokenizer.py --gguf "$MODELS/$SHARD1" --out "$PREP"
fi

if [ -f "$PREP/mtp-rt/dense.bin" ] && [ -f "$PREP/mtp-rt/dense.txt" ] && [ -f "$PREP/mtp-rt/experts.bin" ]; then
  ok "MTP draft layer at $PREP/mtp-rt"
elif [ "$CHECK" = 1 ]; then
  todo "the MTP draft layer $PREP/mtp-rt (5.2 GB from Qwen's checkpoint)"
else
  say "the MTP draft layer: its tensors from Qwen's checkpoint (5.2 GB, resumable)"
  mkdir -p "$PREP/mtp-bf16"
  run_tool tools/mtp_fetch.py inventory --out "$PREP/mtp-bf16"
  run_tool tools/mtp_fetch.py fetch --out "$PREP/mtp-bf16"
  run_tool tools/mtp_fetch.py verify --out "$PREP/mtp-bf16"
  if [ ! -f "$PREP/mtp-q2_0.gguf" ]; then
    run_tool tools/mtp_pack.py --src "$PREP/mtp-bf16" --experts q2_0 --out "$PREP/mtp-q2_0.tmp.gguf"
    mv "$PREP/mtp-q2_0.tmp.gguf" "$PREP/mtp-q2_0.gguf"
  fi
  rm -rf "$PREP/mtp-rt.tmp"
  run_tool tools/mtp_rt.py --gguf "$PREP/mtp-q2_0.gguf" --out "$PREP/mtp-rt.tmp"
  mv "$PREP/mtp-rt.tmp" "$PREP/mtp-rt"
  say "the MTP draft layer is in $PREP/mtp-rt"
fi
# its experts as Q4_0 from the BF16 tensors (scripts/mtp_q4.py; the UD-Q4_K_XL profiles use them: writing +10 %, the same
# answers): the expert tensors are fetched again if mtp-bf16 is gone
if [ -f "$PREP/mtp-rt/experts-q4_0.bin" ]; then
  ok "Q4_0 draft experts at $PREP/mtp-rt/experts-q4_0.bin"
elif [ "$CHECK" = 1 ]; then
  todo "the Q4_0 draft experts $PREP/mtp-rt/experts-q4_0.bin (5 GB from Qwen's checkpoint unless $PREP/mtp-bf16 is there)"
else
  say "the MTP draft layer's experts as Q4_0"
  if [ ! -f "$PREP/mtp-bf16/tensors/mtp.layers.0.mlp.experts.down_proj.bin" ] || [ ! -f "$PREP/mtp-bf16/tensors/mtp.layers.0.mlp.experts.gate_up_proj.bin" ]; then
    mkdir -p "$PREP/mtp-bf16"
    run_tool tools/mtp_fetch.py inventory --out "$PREP/mtp-bf16"
    run_tool tools/mtp_fetch.py fetch --out "$PREP/mtp-bf16" --only mlp.experts
  fi
  (cd "$STRATA" && .venv/bin/python "$HERE/scripts/mtp_q4.py" "$PREP/mtp-bf16" "$PREP/mtp-rt/experts-q4_0.bin")
  say "$PREP/mtp-bf16 and $PREP/mtp-q2_0.gguf (6 GB) can be deleted now"
fi

# ------------------------------------------------------------------ the picture encoder (as ours: CUDA build, run on the CPU)
VISION=$STRATA/build-vision/bin/strata-vision
if [ "$PICTURES" = 0 ]; then
  ok "no pictures"
elif [ -x "$VISION" ]; then
  ok "picture encoder at $VISION"
elif [ "$CHECK" = 1 ]; then
  todo "build the picture encoder $VISION"
else
  . "$HERE/scripts/toolchain.sh"
  pick_toolchain
  HOSTCC=$(dirname "$HOSTCXX")/$(basename "$HOSTCXX" | sed 's/g++/gcc/')
  [ -x "$HOSTCC" ] || HOSTCC=$(command -v gcc || true)
  [ -n "$HOSTCC" ] || die "no C compiler for the picture encoder (install gcc-13)"
  say "building the picture encoder (nvcc $CUDA_VER, $HOSTCXX)"
  cmake -S "$STRATA/tools/vision" -B "$STRATA/build-vision" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$GG" \
    -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX" -DCMAKE_C_COMPILER="$HOSTCC" -DCMAKE_CXX_COMPILER="$HOSTCXX"
  nice -n 10 cmake --build "$STRATA/build-vision" --target strata-vision -j "${JOBS:-6}"
  [ -x "$VISION" ] || die "the build gave no $VISION"
fi

# ------------------------------------------------------------------ the run config
CFG=$DATA/profile.json
if [ -f "$CFG" ]; then
  ok "run config $CFG (kept; delete it to write it again)"
  had=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("gpu", ""))' "$CFG" 2>/dev/null || true)
  [ -z "$GPU" ] || [ "$GPU" = "$had" ] || say "WARNING: the kept run config names cards $had, not $GPU (--gpu); set \"gpu\" in it or delete it"
elif [ "$CHECK" = 1 ]; then
  todo "the run config $CFG"
else
  mkdir -p "$DATA/logs"
  vis=()
  [ "$PICTURES" = 1 ] && vis=(--vision-exe "$VISION")
  [ "$PICTURES_GPU" = 1 ] && vis+=(--pictures-gpu)
  [ "$CPU_EXPERTS" = 1 ] && vis+=(--cpu-experts)
  coder=()
  [ "$CODER" = 1 ] && coder=(--coder)
  [ "$FAST" = 1 ] && coder+=(--fast)
  python3 "$HERE/profile/fill_profile.py" --models "$MODELS" --mtp "$PREP/mtp-rt" --tokenizer "$PREP/tokenizer" \
    ${vis[@]+"${vis[@]}"} ${coder[@]+"${coder[@]}"} --logs "$DATA/logs" --model "$MODEL" --route-prior "$PRIOR" --ram "$RAM" \
    --gpu "$GPU_LIST" --exe "$EXE" --out "$CFG"
fi

if [ "$CHECK" = 1 ]; then
  say "$TODO step(s) to do"
else
  say "done. Start it with: bash $HERE/run.sh $DATA"
fi
