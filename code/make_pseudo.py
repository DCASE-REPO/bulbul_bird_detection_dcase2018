#!/usr/bin/env python

import random
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filelist", type=str, help="filelist file")
parser.add_argument("--filelist-header", action='store_true', help="filelist file has header")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold (default=%(default)s)")
parser.add_argument("--folds", type=int, default=2, help="Number of folds (default=%(default)s)")
parser.add_argument("--out", type=str, help="Out file template (default='%(default)s')")
parser.add_argument("--out-prefix", type=str, default='', help="out item prefix (default='%(default)s')")
parser.add_argument("--out-suffix", type=str, default='', help="out item suffix (default='%(default)s')")
parser.add_argument("--out-header", action='store_true', help="write eventual filelist header")
args = parser.parse_args()

fnin = args.filelist
thr = args.threshold
folds = args.folds
fnout = args.out

ok = lambda r: r <= thr or r >= 1.-thr

hdr = None
with open(fnin, 'r') as fin:
    if args.filelist_header:
        hdr = fin.next()
    ids = [(id,rt) for id,rt in (ln.strip().split(',') for ln in fin) if ok(float(rt))]

random.shuffle(ids)

for fold in range(folds):
    fn = fnout%(dict(fold=fold+1))
    with open(fn, 'w') as fout:
        if args.out_header and hdr is not None:
            fout.write(hdr)
        for id, rt in ids[fold::folds]:
            print >>fout, "%s%s%s,%s" % (args.out_prefix, id, args.out_suffix, rt)
