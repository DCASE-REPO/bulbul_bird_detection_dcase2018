import numpy as np
import logging
import glob
import os
import random
import itertools
import urllib
import sys
# local module


def loopspec(spec, width, offs=0):
    if not width:
        yield spec
    else:
        assert width > 0

        # produce multiple outputs from input spectrum by looping it within the output window
        # TODO: don't just concatenate but do a little crossfade to make it more natural
        for o in xrange(0, len(spec), width):
            if offs+o+width > len(spec):
                idxs = np.arange(width)
                idxs += offs+o
                idxs %= len(spec)
                yield spec[idxs]
            else:
                yield spec[offs+o:offs+o+width]


def process_cut(spect, stddevs=3, ignore=2):
    from scipy.ndimage.filters import maximum_filter1d
    # "loudness" curve
    loud = np.log(np.sum(np.exp(spect), axis=1))
    # normieren
    nloud = (loud-np.mean(loud))/np.std(loud)
    lowloud = nloud < -stddevs
    # schauen, ob innerhalb von 3 stddevs, Klicks darin (bis zu 2 Frames lang) ignorieren.
    fltloud = ~maximum_filter1d(lowloud, ignore+1)
    where_ok = np.where(fltloud)[0]
    cut_front = np.min(where_ok)
    cut_back = np.max(where_ok)
    return cut_front,cut_back+1


def process_denoise(spect, mode='mean'):
    if mode == 'mean':
        corr = np.mean(spect, axis=0)
    elif mode == 'median':
        corr = np.median(spect, axis=0)
    else:
        raise ValueError('Mode unknown')
    return spect-corr


try:
    import util
except ImportError:
    pass
else:
    def process(data, args={}, label=None, column=None):
        assert column == -1
    
        data_type = util.getarg(args, 'type', label=label)
        if data_type not in ('audio','spect'):
            raise ValueError("load_data needs data_type option")
        
        labelfiles = util.getarg(args, 'labels', '', label=label, dtype=str) # wildcards and/or comma-separated
        targets_needed = util.getarg(args, 'targets_needed', True, label=label, dtype=bool)
        data_path = util.getarg(args, 'data', '', label=label, dtype=str)
        data_vars = util.getarg(args, 'data_vars', '', label=label, dtype=str)
        downmix = util.getarg(args, 'downmix', True, label=label, dtype=bool)
        pad_front = util.getarg(args, 'pad_front', 0, label=label, dtype=int)
        pad_back = util.getarg(args, 'pad_back', 0, label=label, dtype=int)
        pad_mode = util.getarg(args, 'pad_mode', 'zero', label=label, dtype=str)
        multiple = util.getarg(args, 'multiple', 1, label=label, dtype=int)
        seed = util.getarg(args, 'seed', -1, label=label, dtype=int)
        cycle = util.getarg(args, 'cycle', 0, label=label, dtype=int)
        cache = util.getarg(args, 'cache', False, label=label, dtype=bool)
        eqgain = util.getarg(args, 'eqgain', 0., label=label, dtype=float)
        width = util.getarg(args, 'width', 0, label=label, dtype=int)
        offset = util.getarg(args, 'offset', 0, label=label, dtype=int)
        useweights = util.getarg(args, 'weights', False, label=label, dtype=bool)
        lmbda = util.getarg(args, 'lambda', 1., label=label, dtype=float)

        useclasses = util.getarg(args, 'useclasses', False, label=label, dtype=bool)
        classes = util.getarg(args, 'classes', '', label=label, dtype=str)
    
        cut_stddevs = util.getarg(args, 'cut_stddevs', 0, label=label, dtype=float)
        cut_ignore = util.getarg(args, 'cut_ignore', 4, label=label, dtype=int)
        
        denoise = util.getarg(args, 'denoise', False, label=label, dtype=bool)
        denoise_mode = util.getarg(args, 'denoise_mode', 'mean', label=label, dtype=str)

        rng = random.Random(seed if seed >= 0 else None)
        classes = classes.split(',')

        # read all available labels
        labels = {}
        for fns in labelfiles.split(','):
            if fns.strip():
                for fn in glob.glob(fns):
                    fid = os.path.splitext(os.path.split(fn)[-1])[0]
                    with open(fn) as f:
                        for ln in f:
                            i,l = ln.strip().split(',')
                            i = os.path.splitext(os.path.split(i)[-1])[0] # no path or extension
                            ifull = os.path.join(fid,i)
                            try:
                                lval = float(l)
                            except ValueError:
                                # not a float
                                continue
                            if ifull in labels and labels[ifull] != lval:
                                logging.warning("Label ID %s already present with different value (%f != %f)"%(ifull,labels[ifull],lval))
                            labels[ifull] = lval # true/false

        # data variations
        data_vars = data_vars.split(',')

        cachemem = {}        
        for item in data:
            info = item[-1]
            fileid = info['id']
            fileid_noext = os.path.splitext(fileid)[0]       
            fileid_class = os.path.split(fileid_noext)[0]

            # fileid has subpaths
            fns = [data_path%dict(id=fileid, id_noext=fileid_noext, var=v) for v in data_vars]

            samplerate = None
            inps = []
            cut_low = []
            cut_high = []
            for fn in fns:
                if not os.path.exists(fn):
                    raise ValueError("No file found for input path '%s'"%fn)

                try:
                    inp_data, meta = cachemem[fn]
                except KeyError:
                    try:
                        inp_data, meta = util.load(fn, args=args, metadata=True, label=label)
                    except IOError:
                        print >>sys.stderr, "Input file %s is broken"%fn
                        raise
                if cache:
                    cachemem[fn] = (inp_data, meta)
                logging.debug("Loaded input file '%s': %s'"%(fileid, fn))

                # target processing
                if data_type == 'audio':
                    samplerate = meta['samplerate']
                    inp = inp_data
                else:
                    samplerate = 1./np.diff(inp_data['times']).mean()
                    inp = inp_data['features']
            
                if cut_stddevs > 0:
                    low,high = process_cut(inp, stddevs=cut_stddevs, ignore=cut_ignore)
                    inp = inp[low:high]
                    cut_low.append(low)
                    cut_high.append(high)
            
                if denoise:
                    # 'denoise' by subtracting the average over time
                    inp = process_denoise(inp, mode=denoise_mode)
                
                inps.append(inp)

            if cut_high:
                low_max = max(cut_low)
                high_min = min(cut_high)
                inps = [inp[low_max-low:high_min-high or None] for inp,low,high in zip(inps,cut_low,cut_high)]
       
            # time must be first axis
            inps = np.asarray(inps).swapaxes(0,1)

            if downmix:
                # mix down channels but keep dimensionality
                inps = inps.mean(axis=-1)[...,np.newaxis]

            
            samples = len(inps)
            
            # pad by given pad lengths 
            inps_len = samples+pad_front+pad_back
            # and additionally by rounding up length to 'multiple' param
            pad_multiple = int(np.ceil(float(inps_len)/multiple))*multiple-inps_len
    
            if meta is None:
                meta = dict()
    
            meta['pad_front'] = pad_front
            meta['pad_back'] = pad_back+pad_multiple
            meta['frames'] = samples
            meta['framerate'] = samplerate
    
            if pad_front+pad_back+pad_multiple:
                if pad_mode == 'zero':
                    pad_data_front = np.zeros((pad_front,)+inps.shape[1:], dtype=inps.dtype)
                    pad_data_back = np.zeros((pad_back+pad_multiple,)+inps.shape[1:], dtype=inps.dtype)
                elif pad_mode == 'copy':
                    pad_data_front = np.repeat(inps[:1], repeats=pad_front, axis=0)
                    pad_data_back = np.repeat(inps[-1:], repeats=pad_back+pad_multiple, axis=0)
                else:
                    raise ValueError("Pad mode '%s' unknown"%pad_mode)
        
                inps = np.concatenate((pad_data_front, inps, pad_data_back), axis=0)

            try:
                tgt = labels[fileid_noext]
            except KeyError:
                if targets_needed:
                    logging.error("File ID '%s' not found in labels"%fileid_noext)
                    raise
                else:
                    tgt = 0.

            if useclasses:
                clss = classes.index(fileid_class)
                outp = np.asarray((tgt,clss), dtype=bool)
            else:
                outp = np.asarray((tgt,), dtype=np.float32)

            if useweights:
                w = np.ones(outp.shape, dtype=np.float32)
                weights = [w]
            else:
                weights = []

            # update meta information
            info.update(dict(fns=fns))

            for variation in xrange(cycle or 1):
                offs = offset+(rng.randint(1, len(inps)-1) if variation else 0)
                for vinps in loopspec(inps, width, offs):
                    # augment using equalization and colored noise
                    if eqgain:
                        # use a sine curve with random phase and eqgain amplitude to modulate the spectrum
                        eq = np.sin((np.arange(vinps.shape[1],dtype=np.float32)/vinps.shape[1]+rng.random())*np.pi*2)*(eqgain*0.5)
                        vinps = vinps+eq

                    yield tuple([inp for inp in vinps.swapaxes(0,1)]+[outp] + weights + [info])
