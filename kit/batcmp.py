#!/usr/bin/env python3
"""batcmp.py REF LABEL...: each LABEL's battery against REF's: lh, lh2, the written tokens, the drafts and where it went back to."""
import json, sys
import os
X = os.path.dirname(os.path.abspath(__file__)) + '/out/bat-%s.json'
ref = {r['name']: r for r in json.load(open(X % sys.argv[1]))}
bad = 0
for lab in sys.argv[2:]:
    cur = {r['name']: r for r in json.load(open(X % lab))}
    n = same = 0
    for name, a in ref.items():
        if name not in cur: continue
        b = cur[name]; n += 1
        ok = all(a[k] == b[k] for k in ('lh', 'lh2', 'tokens', 'drafted', 'accepted', 'cached', 'back'))
        same += ok
        if not ok: print('  %s %s DIFFERENT: %s' % (lab, name, [k for k in ('lh', 'lh2', 'tokens', 'drafted', 'accepted', 'cached', 'back') if a[k] != b[k]]))
    tr = sum(r['read'] for r in cur.values() if r['read'] > 4000); ts = sum(r['read_s'] for r in cur.values() if r['read'] > 4000)
    print('%s against %s: %d of %d the same (lh, lh2, tokens, drafts, checkpoint); reads over 4k tokens: %.1f tok/s' % (lab, sys.argv[1], same, n, tr / max(ts, 1e-9)))
    bad += n - same
sys.exit(1 if bad else 0)
