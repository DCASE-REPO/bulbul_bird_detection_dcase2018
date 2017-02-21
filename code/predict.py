#!/usr/bin/env python
import numpy as np
import h5py
import os
import sys
from collections import defaultdict
from itertools import izip

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs='+', type=str, help="Model file(s), using wildcards")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold (default=%(default)s)")
parser.add_argument("--acc", choices=('mean','median'), default='mean', help="Accumulation (default='%(default)s')")
parser.add_argument("--acc-id", choices=('mean','median','min','max'), default='max', help="Per-id accumulation (default='%(default)s')")
parser.add_argument("--filelist", type=str, help="filelist file(s) (multiple files comma-separated)")
parser.add_argument("--filelist-header", action='store_true', help="filelist files have header")
parser.add_argument("--out", type=str, help="out file (default=stdout)")
parser.add_argument("--keep-prefix", action='store_true', help="keep eventual item prefix")
parser.add_argument("--keep-suffix", action='store_true', help="keep eventual item suffix")
parser.add_argument("--out-prefix", type=str, default='', help="out item prefix (default='%(default)s')")
parser.add_argument("--out-suffix", type=str, default='', help="out item suffix (default='%(default)s')")
parser.add_argument("--out-header", action='store_true', help="write eventual filelist header")
parser.add_argument("--skip-missing", action='store_true', help="Skip files with missing predictions")
args = parser.parse_args()
    
facc = np.__dict__[args.acc]
facc_id = np.__dict__[args.acc_id]

res = defaultdict(list) # total results
for fn in args.filenames:
    resf = defaultdict(list) # per file
    with h5py.File(fn, 'r') as f5:
        print >>sys.stderr, "Reading", fn
        ids = f5['ids']['id'].value
        results = f5['results'][:,-1] # either scalar probability or two-element softmax output
        assert len(ids) == len(results)
        for i, r in izip(ids, results):
            resf[i].append(r)
    # accumulate over file and add to total
    for i,r in resf.iteritems():
        if len(r) != 1:
            print >>sys.stderr, "%s: id=%s, %i times"%(fn,i,len(r))
        res[i].append(facc_id(r))

# sort ids        
resids = sorted(res.keys())
# calculate bagged results for each id
mns = np.asarray([facc(res[i], axis=0) for i in resids])

prefun = (lambda fn:fn) if args.keep_prefix else (lambda fn: os.path.split(fn)[-1])
suffun = (lambda fn:fn) if args.keep_suffix else (lambda fn: os.path.splitext(fn)[0])

results = dict(zip((suffun(prefun(r)) for r in resids), mns))

if args.filelist:
    if args.out:
        fout = open(args.out, 'w')
    else:
        fout = sys.stdout
    
    for fn in args.filelist.split(','):
        with open(fn, 'r') as flist:
            for lni,ln in enumerate(flist):
                if lni == 0 and args.filelist_header:
                    if args.out_header:
                        print >>fout, ln.strip() # replicate header line
                else:
                    fid = ln.strip().split(',')[0].strip() # first column only
                    try:
                        pred = results[fid]
                    except KeyError:
                        print >>sys.stderr, "Prediction missing for %s," % fid
                        if args.skip_missing:
                            print >>sys.stderr, "skipping."
                            continue
                        else:
                            print >>sys.stderr, "exiting."
                            exit(-1)
                    if pred <= args.threshold or pred >= 1.-args.threshold:
                        print >>fout, "%s%s%s,%.6f" % (args.out_prefix, fid, args.out_suffix, pred)

    if args.out:
        fout.close()
