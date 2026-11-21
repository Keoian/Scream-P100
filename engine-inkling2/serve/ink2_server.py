#!/usr/bin/env python3
"""ink2_server.py --port N [--host H] [--alias A] --sock S --ctx C --model M.gguf --tokenizer llama-tokenize --gguf-py DIR

An OpenAI-compatible HTTP server in front of the engine (ink2_v1 --serve S), also answering the endpoints a llama-server
client expects: /health, /props, /slots, /tokenize, /apply-template, /metrics, /v1/models and /v1/chat/completions with and
without "stream". The prompt is Inkling 2's chat template as text (tools and pictures are not supported), tokens come from
llama.cpp's llama-tokenize (vocabulary only), text back from the GGUF vocabulary (GPT-2 byte level). Greedy only: the engine
has no sampler, so temperature and the like are ignored. Drafts: DSpark (MTP) 3 + prompt lookup 15 unless a request says
"mtp" / "lookup". The reply's <|content_thinking|> part streams as reasoning_content, its <|content_text|> part as content.
reasoning_effort (none / minimal / low / medium / high / xhigh) or chat_template_kwargs.reasoning_effort sets the template's
"Thinking effort level". One generation at a time; the other endpoints answer meanwhile. A client that leaves stops the
generation. SIGTERM stops the engine too when INK2_ENGINE_PID is set (run.sh does)."""
import argparse
import codecs
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--alias", default="inkling2-mini")
ap.add_argument("--sock", required=True)
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True, help="llama.cpp's llama-tokenize")
ap.add_argument("--gguf-py", required=True, help="llama.cpp's gguf-py folder (knows NVFP4)")
A, _ = ap.parse_known_args()
TOKENIZE = A.tokenizer
sys.path.insert(0, A.gguf_py)
import gguf  # noqa: E402

EFFORT = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
ROLE = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>",
        "developer": "<|message_system|>"}
EOS = 200006
MTP = int(os.environ.get("INK2_OA_MTP", "3"))
LOOKUP = int(os.environ.get("INK2_OA_LOOKUP", "15"))

r = gguf.GGUFReader(A.model)
f = r.fields["tokenizer.ggml.tokens"]
TOKS = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
tf = r.fields.get("tokenizer.chat_template")
TEMPLATE = bytes(tf.parts[tf.data[0]]).decode("utf-8") if tf else ""
del r
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs, n_ = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n_)
        n_ += 1
BYTEDEC = {chr(c): b for b, c in zip(bs, cs)}
SPECIAL = {i: t for i, t in enumerate(TOKS) if t.startswith("<|") and t.endswith("|>")}
T_THINK = next(i for i, t in SPECIAL.items() if t == "<|content_thinking|>")
T_TEXT = next(i for i, t in SPECIAL.items() if t == "<|content_text|>")

LOCK = threading.Lock()
STATE = {"busy": False, "n_prompt": 0, "n_decoded": 0, "n_ctx": A.ctx, "t_start": 0.0, "requests": 0, "tokens_out": 0}


def token_bytes(i):
    t = TOKS[i]
    return t.encode() if i in SPECIAL else bytes(BYTEDEC.get(ch, ord("?")) for ch in t)


def render(messages, effort):
    num = EFFORT[effort] if isinstance(effort, str) and effort in EFFORT else float(effort if effort is not None else 0.9)
    num = max(0.0, min(0.99, num))
    eff = "0" if num == 0.0 else f"{num}"
    s, emitted = "", False
    for m in messages:
        role = m.get("role", "user")
        if not emitted and role not in ("system", "developer"):
            s += f"<|message_system|><|content_text|>Thinking effort level: {eff}<|end_message|>"
            emitted = True
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if role == "assistant" and m.get("reasoning_content"):
            s += f"{ROLE['assistant']}<|content_thinking|>{m['reasoning_content']}<|end_message|>"
        s += f"{ROLE.get(role, ROLE['user'])}<|content_text|>{content}<|end_message|>"
        if role == "assistant":
            s += "<|content_model_end_sampling|>"
    if not emitted:
        s += f"<|message_system|><|content_text|>Thinking effort level: {eff}<|end_message|>"
    return s


def tokenize(text):
    p = subprocess.run([TOKENIZE, "-m", A.model, "--ids", "--log-disable", "--stdin"], input=text, capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": ""}, timeout=600)
    line = [x for x in p.stdout.splitlines() if x.startswith("[")]
    if not line:
        raise RuntimeError("llama-tokenize: " + p.stderr[-300:])
    return json.loads(line[-1])


def effort_of(body):
    kw = body.get("chat_template_kwargs") or {}
    e = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if e is None and kw.get("enable_thinking") is False:
        e = "none"
    return e if e is not None else 0.9


class Parts:
    """Turns the reply's tokens into (kind, text) pieces: kind "r" = reasoning, "c" = content; template tokens are dropped."""

    def __init__(self):
        self.kind = "c"
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, ids):
        out = []
        for i in ids:
            if i == T_THINK:
                out.append((self.kind, self.dec.decode(b"", final=True)))
                self.kind = "r"
                continue
            if i == T_TEXT:
                out.append((self.kind, self.dec.decode(b"", final=True)))
                self.kind = "c"
                continue
            if i in SPECIAL or i == EOS:
                continue
            out.append((self.kind, self.dec.decode(token_bytes(i))))
        return [(k, t) for k, t in out if t]


def engine_lines(req):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(7200)
    s.connect(A.sock)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    try:
        while True:
            c = s.recv(1 << 16)
            if not c:
                break
            buf += c
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield json.loads(line)
    finally:
        s.close()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        b = obj.encode() if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            return self._send(200, {"status": "ok"})
        if path == "/props":
            return self._send(200, {"default_generation_settings": {"n_ctx": A.ctx, "params": {"temperature": 0.0}}, "total_slots": 1,
                                    "model_path": A.model, "model_alias": A.alias, "chat_template": TEMPLATE, "build_info": "ink2_v1",
                                    "modalities": {"vision": False, "audio": False}})
        if path == "/slots":
            st = STATE
            return self._send(200, [{"id": 0, "n_ctx": A.ctx, "is_processing": st["busy"], "n_prompt_tokens": st["n_prompt"],
                                     "next_token": [{"n_decoded": st["n_decoded"], "has_next_token": st["busy"]}]}])
        if path == "/metrics":
            return self._send(200, f"# ink2_v1\nllamacpp:requests_processing {1 if STATE['busy'] else 0}\n"
                                   f"llamacpp:tokens_predicted_total {STATE['tokens_out']}\n", "text/plain")
        if path in ("/v1/models", "/models"):
            return self._send(200, {"object": "list", "data": [{"id": A.alias, "object": "model", "owned_by": "ink2_v1",
                                                                "meta": {"n_ctx_train": 1048576, "n_ctx": A.ctx}}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._body()
        except Exception as ex:
            return self._send(400, {"error": f"bad json: {ex}"})
        if path == "/tokenize":
            try:
                ids = tokenize(body.get("content") or "")
            except Exception as ex:
                return self._send(500, {"error": str(ex)})
            return self._send(200, {"tokens": ids})
        if path == "/apply-template":
            return self._send(200, {"prompt": render(body.get("messages") or [], effort_of(body))})
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self.chat(body)
        return self._send(404, {"error": "not found"})

    def chat(self, body):
        t0 = time.time()
        try:
            ids = tokenize(render(body.get("messages") or [], effort_of(body)))
        except Exception as ex:
            return self._send(500, {"error": str(ex)})
        if len(ids) >= A.ctx - 1:
            return self._send(400, {"error": {"message": f"prompt of {len(ids)} tokens exceeds the context of {A.ctx}", "type": "exceed_context_size_error"}})
        mx = int(body.get("max_tokens") or body.get("max_completion_tokens") or 0) or (A.ctx - len(ids) - 1)
        mx = max(1, min(mx, A.ctx - len(ids) - 1))
        stream = bool(body.get("stream"))
        req = {"tokens": ids, "max": mx, "stop": [EOS], "mtp": int(body.get("mtp", MTP)), "lookup": int(body.get("lookup", LOOKUP)),
               "stream": 1 if stream else 0}
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(t0)
        if not LOCK.acquire(timeout=3600):
            return self._send(503, {"error": "busy"})
        try:
            STATE.update(busy=True, n_prompt=len(ids), n_decoded=0, t_start=t0)
            STATE["requests"] += 1
            parts = Parts()
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def sse(obj):
                    self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()

                sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": A.alias,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]})
            out, final, t_first = [], None, None
            try:
                for msg in engine_lines(req):
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    if "t" in msg:
                        if t_first is None:
                            t_first = time.time()
                        out += msg["t"]
                        STATE["n_decoded"] = len(out)
                        if stream:
                            for kind, text in parts.feed(msg["t"]):
                                sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": A.alias,
                                     "choices": [{"index": 0, "delta": {"reasoning_content" if kind == "r" else "content": text},
                                                  "finish_reason": None}]})
                    elif "tokens" in msg:
                        final = msg
            except (BrokenPipeError, ConnectionResetError):
                return   # the client left: closing the engine's socket stops it at its next write
            if final is None:
                raise RuntimeError("the engine closed without an answer")
            if not stream:
                out = final["tokens"]
            n = len(out)
            STATE["tokens_out"] += n
            w = final.get("write_s") or 1e-9
            finish = "stop" if out and out[-1] == EOS else "length"
            timings = {"prompt_n": len(ids) - final.get("cached", 0), "cache_n": final.get("cached", 0),
                       "prompt_ms": 1000 * final.get("read_s", 0),
                       "prompt_per_second": (len(ids) - final.get("cached", 0)) / max(final.get("read_s", 0), 1e-9),
                       "predicted_n": n, "predicted_ms": 1000 * w, "predicted_per_second": max(n - 1, 0) / w,
                       "draft_n": final.get("drafted", 0), "draft_n_accepted": final.get("accepted", 0)}
            usage = {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n,
                     "prompt_tokens_details": {"cached_tokens": final.get("cached", 0)}}
            if stream:
                sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": A.alias,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage, "timings": timings})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            think, content = [], []
            for kind, text in Parts().feed(out):
                (think if kind == "r" else content).append(text)
            return self._send(200, {"id": cid, "object": "chat.completion", "created": created, "model": A.alias,
                                    "choices": [{"index": 0, "finish_reason": finish,
                                                 "message": {"role": "assistant", "content": "".join(content).strip(),
                                                             "reasoning_content": "".join(think).strip()}}],
                                    "usage": usage, "timings": timings})
        except Exception as ex:
            if stream:
                try:
                    self.wfile.write(b"data: " + json.dumps({"error": {"message": str(ex)}}).encode() + b"\n\n")
                except Exception:
                    pass
                return
            return self._send(500, {"error": {"message": str(ex)}})
        finally:
            STATE.update(busy=False)
            LOCK.release()


def _stop(*_):   # the node stops a profile with SIGTERM to this process: the engine (started by ink2-ovoj.sh) goes with it
    pid = int(os.environ.get("INK2_ENGINE_PID") or 0)
    if pid:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    os._exit(0)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    srv = ThreadingHTTPServer((A.host, A.port), H)
    srv.daemon_threads = True
    print(f"ink2_server: {A.alias} na {A.host}:{A.port}, ctx {A.ctx}, pogon {A.sock}", file=sys.stderr, flush=True)
    srv.serve_forever()
