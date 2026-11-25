#!/usr/bin/env python3
"""battery.py LABEL BIN CTX SET [VAR=value ...]: starts BIN as the server does (--serve SOCK, the drafter loaded) and sends a
battery of requests over its socket with "lhall": 1, so every reply carries lh (a hash of the logits after the prompt) and
lh2 (a hash over every later step's logits): the maintainers' own way of comparing two builds bit for bit. Drafts on
(mtp 3, lookup 15), a fixed number of written tokens. SET: full | short, optionally +long36 / +long60 / +long200.
Needs INK2_DATA (setup.sh's folder) and kit/all.ids (make_ids.sh). INK2_ENG = engine-inkling2/ when BIN is not in its build/.
Saves out/bat-LABEL.json; compare two with batcmp.py."""
import ast, json, os, socket, subprocess, sys, time
label, binp, ctx, sets = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split('+')
env = dict(os.environ); env.update(a.split('=', 1) for a in sys.argv[5:])
D = os.environ['INK2_DATA']                                   # setup.sh's DATA
H = os.path.dirname(os.path.dirname(os.path.abspath(binp))) if os.environ.get('INK2_ENG') is None else os.environ['INK2_ENG']   # engine-inkling2/
X = os.path.dirname(os.path.abspath(__file__))
os.makedirs(X + '/out', exist_ok=True)
ids = ast.literal_eval(open(X + '/all.ids').read().strip())
sock = '%s/out/bat-%s.sock' % (X, label)
if os.path.exists(sock): os.remove(sock)
err = open('%s/out/bat-%s.err' % (X, label), 'w')
eng = subprocess.Popen([binp, '--model', D + '/Inkling-2-Mini-NVFP4-Q8_0.gguf', '--mtp', D + '/dspark-Inkling-2-Mini-Q8_0.gguf',
                        '--draft-vocab', H + '/data/draft_vocab_32768.bin', '--serve', sock, '--ctx', ctx], env=env, stderr=err)
for _ in range(300):
    if os.path.exists(sock) or eng.poll() is not None: break
    time.sleep(0.5)
if not os.path.exists(sock): sys.exit('engine did not start: see out/bat-%s.err' % label)
def ask(tokens, hints=(), forget=0, mx=40):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(1800); s.connect(sock)
    s.sendall((json.dumps({'tokens': tokens, 'max': mx, 'stop': [], 'noeos': 1, 'mtp': 3, 'lookup': 15, 'lhall': 1, 'forget': forget,
                           'ckpt': [h for h in hints if h < len(tokens) - 8]}) + '\n').encode())
    buf = b''
    while not buf.endswith(b'\n'):
        c = s.recv(1 << 16)
        if not c: break
        buf += c
    s.close()
    r = json.loads(buf.decode().strip().split('\n')[-1])
    if 'error' in r: sys.exit('engine error: ' + r['error'])
    return r
out = []
def rec(name, r):
    read = r['prompt_n'] - max(r['cached'], 0)
    out.append({'name': name, 'lh': r['lh'], 'lh2': r['lh2'], 'tokens': r['tokens'], 'drafted': r['drafted'], 'accepted': r['accepted'],
                'cached': r['cached'], 'back': r['back'], 'read': read, 'read_s': r['read_s'], 'write_s': r['write_s']})
    print('%-22s lh %s lh2 %s read %6d in %6.2f s (%6.1f tok/s) cached %6d back %6d drafts %d/%d' % (
        name, r['lh'], r['lh2'], read, r['read_s'], read / max(r['read_s'], 1e-9), r['cached'], r['back'], r['accepted'], r['drafted']), flush=True)
    return r
# A: one read from token 0, lengths around the chunk's edges (2,048; a second chunk needs 31 more tokens)
ALL = [31, 500, 2047, 2048, 2078, 2079, 2100, 2304, 4095, 4096, 4097, 4126, 4127, 6399, 8192, 10000]
edges = ALL if 'full' in sets else [2079, 4127, 6399, 10000]
for L in edges:
    o = 1000 + 3571 * ALL.index(L)   # the same text for a length in every set
    rec('edge-%d' % L, ask(ids[o:o + L], forget=1))
# B: a conversation with checkpoints, its next turn, a prompt that parts at a checkpoint, another conversation, and back
x = ids[0:20000]
r1 = rec('conv-20k', ask(x, hints=[3000, 9000, 15000, 19990], forget=1, mx=60))
t2 = x + r1['tokens'] + ids[30000:36000]
rec('next-turn-6k', ask(t2, hints=[3000, 9000, 15000, 19990, 20060], mx=60))
rec('part-at-9k', ask(x[:9000] + ids[40000:47000], hints=[3000, 9000], mx=60))
rec('other-conv-13k', ask(ids[50000:63000], hints=[6000, 12990], mx=60))
rec('back-to-first', ask(t2 + ids[64000:66100], hints=[3000, 9000, 15000, 19990, 20060, 26060], mx=60))
if 'long36' in sets: rec('long-36k', ask(ids[0:36000], hints=[35990], forget=1, mx=60))
if 'long200' in sets: rec('long-200k', ask((ids + ids + ids)[:200000], hints=[199990], forget=1, mx=60))
if 'long60' in sets: rec('long-60k', ask(ids[2000:62000], hints=[59990], forget=1, mx=60))
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(sock); s.sendall(b'{"exit": 1}\n'); s.close()
try: eng.wait(timeout=60)
except subprocess.TimeoutExpired: eng.kill()
json.dump(out, open('%s/out/bat-%s.json' % (X, label), 'w'), indent=1)
