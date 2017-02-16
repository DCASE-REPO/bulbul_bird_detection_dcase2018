#!/usr/bin/env python

from sklearn.metrics import roc_auc_score
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("pred", type=str, help="prediction file")
parser.add_argument("gt", nargs='+', type=str, help="ground truth file(s)")
parser.add_argument("--pred-header", action='store_true', help="header line present in prediction file")
parser.add_argument("--pred-suffix", type=str, default='', help="suffix for items in prediction file")
parser.add_argument("--gt-header", action='store_true', help="header line present in ground truth file(s)")
parser.add_argument("--gt-suffix", type=str, default='', help="suffix for items in ground-truth file(s)")
parser.add_argument("--splits", type=str, default='', help="split file lists for individual scores (comma separated)")
parser.add_argument("--split-header", action='store_true', help="header line present in split file(s)")
parser.add_argument("--split-suffix", type=str, default='', help="suffix for items in split file(s)")
args = parser.parse_args()

pred_probs = {}
with open(args.pred, 'r') as f:
    if args.pred_header:
        f.next()
    for ln in f:
        k,v = ln.strip().split(',')
        pred_probs[k+args.pred_suffix] = float(v)

gt_labels = {}
for gtfn in args.gt:
    subpath = os.path.splitext(os.path.split(gtfn)[-1])[0]
    with open(gtfn, 'r') as f:
        if args.gt_header:
            f.next()
        for ln in f:
            k,v = ln.strip().split(',')
            gt_labels[os.path.join(subpath,k)+args.gt_suffix] = float(v)

gt_items = set(pred_probs.iterkeys())
pred_items = set(gt_labels.iterkeys())

both = gt_items&pred_items
missing = gt_items^pred_items
if len(missing):
    print >>sys.stderr, "Items %s missing in either set"%missing

auc_total = roc_auc_score([gt_labels[k] for k in both], [pred_probs[k] for k in both])
print "%.6f"%auc_total,

if args.splits:
    splaucs = []
    for splfn in args.splits.split(','):
        with open(splfn, 'r') as f:
            if args.split_header:
                f.next()
            split = set(ln.strip().split(',', 1)[0]+args.split_suffix for ln in f)
        both = gt_items&split
        splauc = roc_auc_score([gt_labels[k] for k in both], [pred_probs[k] for k in both])
        splaucs.append(splauc)
    print "("+",".join("%.6f"%r for r in splaucs)+")",
print
