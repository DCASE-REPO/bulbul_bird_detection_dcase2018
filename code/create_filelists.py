#!/usr/bin/env python
# -*- coding: utf-8

import sys

import argparse
parser = argparse.ArgumentParser(description='Create filelists')

parser.add_argument('path', type=str, help='path to filelists')
parser.add_argument('filelists', type=str, nargs='+', help='filelists')
parser.add_argument('--out', default="./%(fold)s_%(num)i", help='output file name (template parameter: fold, num)')
parser.add_argument('--log', action='store_true', help='Log to console')
parser.add_argument('--mode', default="train", help='"train" or "test"')
args = parser.parse_args()

fileids = []
datasetlists = {} # a dict holding 'datasetid'=>[filelist]
for filelist in args.filelists:
    with open("%s/%s.csv"%(args.path, filelist), 'r') as f:
        for ln in f:
            if not ln.startswith("itemid"):
                item, datasetid = ln.strip().split(",")[0:2]
                if datasetid not in datasetlists:
                    datasetlists[datasetid] = []
                datasetlists[datasetid].append("%s/%s.wav"%(filelist, item))

# create folds - same as number of datasetids (during train/val), since each one is treated as its own validation set
if args.mode=='train':
    nfolds = len(datasetlists)
else:
    nfolds = 1   # for "testing" we currently pool all together
for n in range(nfolds):

    if args.mode=='train':
        items_trn = []
        items_val = []
        for othern, (datasetid, itemlist) in enumerate(datasetlists.items()):
            if othern==n:
                items_val.extend(itemlist)
            else:
                items_trn.extend(itemlist)
        foldcollection = [("train", items_trn), ("val", items_val)]
    elif args.mode=='test':
        items_tst = []
        for (datasetid, itemlist) in datasetlists.items():
            items_tst.extend(itemlist)
        foldcollection = [("test", items_tst)]

    # write fold files
    for name, folditems in foldcollection:

        outname = args.out%dict(fold=name, num=n+1)
        with open(outname, 'w') as f:
            f.writelines("%s\n"%f for f in folditems)

        if args.log:
            print >>sys.stderr, "Wrote %s_%i with %i files"%(name, n+1, len(folditems))

