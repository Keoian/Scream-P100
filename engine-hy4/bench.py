#!/usr/bin/env python3
"""engine-hy4/bench.py [URL] (2026-11-21): the speeds docs/HY4.md gives, through the OpenAI-compatible API (default
http://127.0.0.1:8080/v1), on a freshly started engine: fresh writing (12 short prompts, 300 tokens each, the same ones as our runs),
two edits that give back a whole file (Python and Markdown from this repo; the second of two runs counts, as in ours; the
answers go to bench-edit-*.txt in the current folder) and one long read (~14k tokens). No thinking. Speeds are the wrapper's
timings (llama-server's names). Standard library only."""
import json
import sys
import time
import urllib.request
from pathlib import Path

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent

FRESH = [
    "Razloži v približno 200 besedah, kako deluje hranilnik toplote v hiši in kako ga dimenzioniramo.",
    "Napiši funkcijo v Pythonu, ki prebere CSV z datumi in vrne mesečne vsote po kategorijah. Dodaj primer.",
    "Explain in about 200 words how a B-tree index speeds up database lookups and when it does not help.",
    "Write a small Dart class for a retrying HTTP client with exponential backoff and a cancel method.",
    "Opiši v približno 200 besedah zgodovino železnice na Slovenskem in njen pomen za gospodarstvo.",
    "Napiši bash skripto, ki v mapi poišče datoteke, večje od 1 GB, in jih razvrsti po datumu spremembe.",
    "Write a SQL schema for a small library with loans and fines, then a query listing overdue books per member.",
    "Razloži v približno 200 besedah razliko med TCP in UDP ter navedi, kdaj uporabimo katerega.",
    "Napiši v JavaScriptu funkcijo debounce in throttle z testi, brez zunanjih knjižnic.",
    "Pojasni v približno 200 besedah, kako pravilno obrezujemo jablane in zakaj je čas pomemben.",
    "Write a C function that parses a key=value config file into a linked list, with careful error handling.",
    "Popravi in razloži: def povprecje(xs): return sum(xs) / len(xs) ko je seznam prazen, in dodaj teste pytest.",
]


def chat(content, max_tokens, answer=None):
    body = {"model": "hy4-flash-iq3xxs", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.0, "reasoning_effort": "none", "stream": False}
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    if answer is not None:
        answer.append(out["choices"][0]["message"].get("content") or "")
    return out["timings"]


def main():
    sys.stdout.reconfigure(line_buffering=True)   # each line as it comes, also into a file
    print(f"bench.py: {URL}, {time.strftime('%F %T')}")
    gen = ms = 0.0
    for i, p in enumerate(FRESH):
        t = chat(p, 300)
        gen, ms = gen + t["predicted_n"], ms + t["predicted_ms"]
        print(f"  fresh {i + 1:2d}: {t['predicted_per_second']:.1f} tok/s ({t['predicted_n']} tokens)")
    fresh = gen / ms * 1000
    print(f"fresh writing: {fresh:.1f} tok/s")

    py = (ROOT / "engine-hy4/tools/sim_cache.py").read_text(encoding="utf-8")
    md = (ROOT / "docs/GUIDE.md").read_text(encoding="utf-8")[:5000].rsplit("\n", 1)[0] + "\n"   # whole lines only
    edits = {
        "python": "Here is a Python file:\n\n```python\n" + py + "\n```\n\nRename the variable `acc` to `accesses` and give back "
                  "the whole file, otherwise unchanged. Code only.",
        "markdown": "Here is a document:\n\n" + md + "\n\nReplace the word `engine` with `runtime` everywhere and give back the whole "
                    "document, otherwise unchanged. The document only.",
    }
    res = {}
    for name, prompt in edits.items():
        for run in (1, 2):
            got = []
            t = chat(prompt, 4000, got)
            print(f"  edit {name} {run}: {t['predicted_per_second']:.1f} tok/s ({t['predicted_n']} tokens), "
                  f"read {t['prompt_n']} tokens at {t['prompt_per_second']:.0f} tok/s")
        res[name] = t["predicted_per_second"]
        Path(f"bench-edit-{name}.txt").write_text(got[0], encoding="utf-8")   # the answer, to check by eye
    print(f"edits: python {res['python']:.1f}, markdown {res['markdown']:.1f} tok/s")

    text = ""
    for f in sorted((ROOT / "docs").glob("*.md")):
        text += f"\n\n=== {f.name} ===\n" + f.read_text(encoding="utf-8")
    t = chat(text[:44000] + "\n\nSummarise in five bullet points what this project is.", 200)
    print(f"long read: {t['prompt_n']} tokens at {t['prompt_per_second']:.0f} tok/s, then writing {t['predicted_per_second']:.1f} tok/s")


if __name__ == "__main__":
    main()
