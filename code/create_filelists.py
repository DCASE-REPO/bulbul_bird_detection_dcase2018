#!/usr/bin/env python
# -*- coding: utf-8

import sys
labelpath = sys.argv[1]
filesets = sys.argv[2:]
for fileset in filesets:
    with open("%s/%s.csv"%(labelpath, fileset),"r") as f:
        for ln in f:
            if not ln.startswith("itemid"):
                item = ln.strip().split(",")[0]
                print "%s/%s.wav"%(fileset, item)
