#!/bin/bash
# make_ids.sh ENGINE_DIR DATA: kit/all.ids, the tokens of kit/text.txt (86,596 with the files of 21 Nov), by the engine's own llama-tokenize
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
"$1/build/llama.cpp/build/bin/llama-tokenize" -m "$2/Inkling-2-Mini-NVFP4-Q8_0.gguf" -f "$K/text.txt" --ids --log-disable 2>/dev/null | tail -1 > "$K/all.ids"
python3 -c "import ast,sys; n=len(ast.literal_eval(open('$K/all.ids').read().strip())); print(n, 'tokens'); sys.exit(0 if n==86596 else 1)"
