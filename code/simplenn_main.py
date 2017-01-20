#! /usr/bin/env python
# -*- coding: utf-8

"""
Thomas Grill, 2016

Austrian Research Institute for Artificial Intelligence (OFAI)
SALSA project, supported by Vienna Science and Technology Fund (WWTF)

covered by the Artistic License 2.0
http://www.perlfoundation.org/artistic_license_2_0
"""

import simplenn
import logging


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=('train', 'train', 'test', 'evaluate', 'introspect', 'salience'), default='train', help="Mode of operation (default='%(default)s')")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-I", "--inputs", type=str, help="Define processes on data in '[key:]module.callable' form, separated by |")
    group.add_argument("--input", action='append', dest='inputs', help="Add process on data in '[key:]module.callable' form")

    parser.add_argument("-f", "--cv-folds", type=str, help="Cross-validation fold(s) to compute, colon-delimited")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-P", "--processes", type=str, help="Define processes on data in '[key:][column:]module.callable' form, separated by |")
    group.add_argument("--process", action='append', dest='processes', help="Add process on data in '[key:][column:]module.callable' form")

    # Network definition
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-L", "--layers", type=str, help="Network architecture, excluding input and output layer")
    group.add_argument("--layer", action='append', dest='layers', help="Add layer to network architecture")

    parser.add_argument("-p", "--problem", type=str, choices=('binary', 'soft_binary', 'categorical', 'regression', 'auroc'), default='binary', help="Problem class (default='%(default)s')")
    
    group.add_argument("--load", type=str, help="Model load file")
    parser.add_argument("--save", type=str, help="Model save file")

    parser.add_argument("-V", "--var", action='append', help="Additional arguments in '[key:]option=value' form")
    
    parser.add_argument("-v","--verbose", action='count', help="Increase logging verbosity")

    parser.add_argument("--pastalog", type=str, help="pastalog logger to be used (default='%(default)s')")
    args = parser.parse_args()
    
    # set up logging
    if args.verbose >= 2:
        loglvl = logging.DEBUG
    elif args.verbose == 1:
        loglvl = logging.INFO
    else:
        loglvl = logging.WARN
    logging.basicConfig(format='%(levelname)s:%(message)s',level=loglvl)

    options = vars(args)
    
    if isinstance(options['inputs'], (str, unicode)):
        options['inputs'] = [l for l in options['inputs'].split('|') if len(l.strip())]

    if isinstance(options['processes'], (str, unicode)):
        options['processes'] = [l for l in options['processes'].split('|') if len(l.strip())]

    if isinstance(options['layers'], (str, unicode)):
        options['layers'] = [l for l in options['layers'].split('|') if len(l.strip())]

    # Process variable arguments
    options['varargs'] = dict(v.split('=',1) for v in options['var']) if options['var'] else {}

    if args.pastalog:
        import pastalog
        host, model = args.pastalog.rsplit('/', 1)
        log = pastalog.Log(host, model)
        def logger(step, cat, val):
            log.post(cat, value=val, step=step)
    else:
        def logger(step, cat, val):
            print "Epoch %3i: %s = %.3g"%(step, cat, val)


    if options['cv_folds'] is None:
        folds = (None,)
    else:
        folds = map(int, options['cv_folds'].split(','))


    for fold in folds:
        if fold is not None:
            logging.info("Cross-validation fold: %i in %s"%(fold,folds))

        # stick cv_fold into varargs, so that it gets saved into the model file
        options['varargs'].update(dict(cv_fold=fold))
        experiment = simplenn.Experiment(**options)
        
        savefile = options['save']%dict(fold=fold)
        if args.mode == 'train':
            experiment.train(savefile=savefile, logger=logger)
        elif args.mode == 'test':
            experiment.test(logger=logger)
        elif args.mode == 'evaluate':
            experiment.evaluate(savefile=savefile, introspection=False)
        elif args.mode == 'introspect':
            experiment.evaluate(savefile=savefile, introspection=True)
        elif args.mode == 'salience':
            experiment.salience(savefile=savefile)
        else:
            raise ValueError("Unknown mode '%s'"%args.mode)
        
    logging.debug("Finished.")

