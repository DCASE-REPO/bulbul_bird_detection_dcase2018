#!/usr/bin/env python

import random
import sys

fnin = sys.argv[1]
thr = float(sys.argv[2])
fnout = sys.argv[3:]

ok = lambda r: r <= thr or r >= 1.-thr

with open(fnin, 'r') as fin:
    ids = [id for id,rt in (ln.split(',') for ln in fin) if ok(float(rt))]

random.shuffle(ids)

nout = len(fnout)
for i, fn in enumerate(fnout):
    with open(fn, 'w') as fout:
        for id in ids[i::nout]:
            print >>fout, id

