#!/usr/bin/env python3
"""UNVERIFIED PROOF OF CONCEPT (engine-4x/README.md): the same requests every time, for comparing first-token logits between runs
(docs/POC-4X.md, "Checking this version"). Two commands:

    python3 engine-4x/logit_requests.py profile IN.json OUT.json FNX_DUMP_FIRST_LOGITS=/tmp/run-a [KEY=VALUE ...]
        a copy of a run config with these "env" entries set (KEY= with nothing after it removes one)
    python3 engine-4x/logit_requests.py send [--url http://127.0.0.1:8080]
        sends the requests below, one at a time, temperature 0, one token each; the engine writes each one's first-token logits to
        PREFIX.0, PREFIX.1, ... when its config has FNX_DUMP_FIRST_LOGITS=PREFIX. Start the engine fresh for each run: the
        conversation cache would serve a repeated prompt without reading it

The requests (sizes estimated; the replies print the real ones): 0 a short one (read through the decode windows,
FNX_READ_WINDOWS=144), 1 a repetitive one of about 2k tokens (like request 4 of issue #1: one prompt chunk), 2 about 6-7k tokens
(two chunks of 4,096 rows), 3 about 20-25k tokens (a pass of four chunks, then the rest, the experts in RAM staged). Then compare
two runs with engine-4x/compare_logits.py PREFIX-A PREFIX-B.
"""
import json
import sys
import time
import urllib.request


def texts():
    words = ("river stone lantern copper meadow signal harbor willow ember quartz falcon orchard tunnel violet anchor summit "
             "cobalt thistle beacon marble").split()
    out = ["Write one short sentence about the sea."]
    rep = "".join(f"Line {i}: the {words[i % 20]} and the {words[(3 * i) % 20]} are listed again as item {i}.\n" for i in range(110))
    out.append("Here is a repetitive list.\n\n" + rep + "\nWhich word appears in line 57? Answer with one word.")

    def prose(n_lines, salt):
        lines = []
        for i in range(n_lines):
            a, b, c = words[(i * 7 + salt) % 20], words[(i * 11 + 3 * salt) % 20], words[(i * 13 + 5 * salt) % 20]
            lines.append(f"{i + 1}. In section {i // 10 + 1}, the {a} report notes that {b} levels near the {c} rose by {(i * 37 + salt) % 97} "
                         f"percent, while the count of {words[(i + salt) % 20]} items stayed at {(i * 53 + 7 * salt) % 1000}.")
        return "\n".join(lines)
    out.append("Read these notes.\n\n" + prose(190, 1) + "\n\nWhat did section 4 say about the harbor? One sentence.")
    out.append("Read this log.\n\n" + prose(640, 2) + "\n\nWhich section mentions the falcon first? One sentence.")
    return out


def send(url):
    model = "strata"
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=10) as r:
            ids = [m.get("id", "") for m in json.load(r).get("data", [])]
            model = ids[0] if ids else model
    except Exception:
        pass
    for i, t in enumerate(texts()):
        body = {"model": model, "messages": [{"role": "user", "content": t}], "max_tokens": 1, "temperature": 0.0,
                "reasoning_effort": "none", "stream": False}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.load(r)
        u = out.get("usage", {})
        print(f"request {i}: {u.get('prompt_tokens', '?')} prompt tokens, {time.time() - t0:.1f} s, "
              f"reply {json.dumps(out['choices'][0]['message'].get('content', ''))}")


def profile(src, dst, sets):
    cfg = json.load(open(src, encoding="utf-8"))
    for kv in sets:
        if "=" not in kv:
            sys.exit(f"{kv}: want KEY=VALUE")
        k, v = kv.split("=", 1)
        if v:
            cfg["env"][k] = v
        else:
            cfg["env"].pop(k, None)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=1) + "\n")
    print(f"wrote {dst}: " + ", ".join(sets))


def main():
    a = sys.argv[1:]
    if a[:1] == ["send"]:
        url = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:8080"
        send(url.rstrip("/"))
    elif a[:1] == ["profile"] and len(a) >= 4:
        profile(a[1], a[2], a[3:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
