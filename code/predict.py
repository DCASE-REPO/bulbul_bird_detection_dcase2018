#!/usr/bin/env python
import numpy as np
import h5py
import os
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs='+', type=str, help="Model file(s), using wildcards")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold (default=%(default)s)")
parser.add_argument("--acc", choices=('mean','median'), default='mean', help="Accumulation (default='%(default)s')")
parser.add_argument("--acc-id", choices=('mean','median','min','max'), default='max', help="Per-id accumulation (default='%(default)s')")
parser.add_argument("--filelist", type=str, help="filelist file")
parser.add_argument("--filelist-header", action='store_true', help="filelist file has header")
parser.add_argument("--out", type=str, help="out file (default=stdout)")
parser.add_argument("--out-prefix", type=str, default='', help="out item prefix (default='%(default)s')")
parser.add_argument("--out-suffix", type=str, default='', help="out item suffix (default='%(default)s')")
parser.add_argument("--out-header", action='store_true', help="write eventual filelist header")
args = parser.parse_args()
    
facc = np.__dict__[args.acc]
facc_id = np.__dict__[args.acc_id]

resids = None
res = []
for fn in args.filenames:
    with h5py.File(fn, 'r') as f5:
        print >>sys.stderr, "Reading", fn
        ids = f5['ids']['id'].value
        if resids is None:
            resids = list(set(ids))
        else:
            assert resids == list(set(ids))

        results = f5['results'].value.flatten()
            
        r = np.empty(len(resids), dtype=float)
        for i,id in enumerate(resids):
            ididxs = np.where(ids == id)[0]
            if len(ididxs) != 1:
                print >>sys.stderr, "%s: id=%s, %i times"%(fn,id,len(ididxs))
            r[i] = facc_id(results[ididxs])
        res.append(r)
        
res = np.asarray(res)
fns = [os.path.splitext(os.path.split(fn)[-1])[0] for fn in args.filenames]

mns = facc(res, axis=0)

results = dict(zip((os.path.splitext(os.path.split(r)[-1])[0] for r in resids), mns))

if args.filelist:
    if args.out:
        fout = open(args.out, 'w')
    else:
        fout = sys.stdout
    
    with open(args.filelist, 'r') as flist:
        for lni,ln in enumerate(flist):
            if lni == 0 and args.filelist_header:
                if args.out_header:
                    print >>fout, ln.strip() # replicate header line
            else:
                fid = ln.split(',')[0] # first column only
                try:
                    pred = results[fid]
                except KeyError:
                    print >>sys.stderr, "Prediction not found for %s" % fid
                    exit(-1)
                if pred <= args.threshold or pred >= 1.-args.threshold:
                    print >>fout, "%s%s%s,%.6f" % (args.out_prefix, fid, args.out_suffix, pred)

    if args.out:
        fout.close()