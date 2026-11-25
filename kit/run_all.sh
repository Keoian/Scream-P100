#!/bin/bash
# run_all.sh REL NEW: the battery on two builds of ink2_v1 (REL: branch inkling2 at 1aa0abe, NEW: this branch), each arm a
# freshly loaded engine. Needs both cards free for about 45 minutes and INK2_DATA. Prints the comparison at the end.
set -u
K=$(cd "$(dirname "$0")" && pwd); cd "$K"; REL=$1; NEW=$2
arm() { echo "=== $(date -u +%T) arm $1"; python3 battery.py "$@" | tail -n 2 | cut -c1-150; }
arm rel   $REL 262144 full+long36
arm off   $NEW 262144 full
arm on2   $NEW 262144 full+long36 INK2_PF_ZIP=2
arm on2b  $NEW 262144 full INK2_PF_ZIP=2
arm on3   $NEW 262144 short INK2_PF_ZIP=3
arm n0    $NEW 262144 short INK2_PF_ZIP=2 INK2_NUMA=0
arm n2    $NEW 262144 short INK2_PF_ZIP=2 INK2_NUMA=2
arm r512  $REL 524288 full+long60
arm o512  $NEW 524288 full
arm z512  $NEW 524288 full+long60 INK2_PF_ZIP=2
{ python3 batcmp.py rel off on2 on2b on3 n0 n2; python3 batcmp.py r512 o512 z512
  grep -h -E 'INK2_PF_ZIP|medpomnilnika|v RAM-u|VRAM [0-9]' out/bat-on2.err out/bat-n0.err out/bat-n2.err out/bat-z512.err out/bat-r512.err | cut -c1-160; } 2>&1 | tee result.txt
