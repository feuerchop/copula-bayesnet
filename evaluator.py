#!/usr/bin/env python
# coding: utf-8

'''
Evaluation functions for gaussian copula structure learning

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import numpy as np
import itertools, logging, pickle
from termcolor import colored
from datetime import datetime
from gcdag import H3GCDiGraph
from joblib import Parallel, delayed, dump, load
from multiprocessing import cpu_count


def run_param_job_(clf, xtr, ytr=None, xtt=None, ytt=None, outputs=np.array([]), output_idx=tuple([]), scoring='shd'):
   start = datetime.now()
   clf.fit(xtr)
   delta = datetime.now() - start
   if scoring == 'shd':
      errors, _, _, _ = shd(ytr, clf.conditional_independences_)
   else:
      scores_tt = clf.score_function(xtt)
      errors = locals()[scoring](ytt, scores_tt)

   outputs[output_idx, 0] = errors
   outputs[output_idx, 1] = delta.total_seconds()


def eval_params(clf, Xtr, ytr,
                scoring = 'shd',
                folds = [],
                params=dict(),
                save_path='Results/untitled.p',
                logger=logging.getLogger('gcdag.experiments.log'),
                overwrite=False,
                n_jobs=-1):
   '''
   We use this generic evaluation function to run experiments with respect to certain parameters.
   :param clf: classifier
   :param net: network
   :param sizes: sample sizes
   :param folds: cross validation folds
   :param params: parameters for classifer
   :param save_path: where to save the results
   :param logger: logger object
   :param overwrite: if we should overwrite the results
   :param n_jobs: how many workers in parallel runinng, default -1 means no parallel.
   '''


   # parsing params
   eval_params_index = dict()
   params_lists = list()
   idx = 0
   for key in params.keys():
      if type(params[key]) is list:
         # param to be evaluated
         eval_params_index[idx] = key
         params_lists.append(params[key])
         idx += 1
      else:
         clf.set_params[key] = params[key]

   # [param1, param2, ... ]
   params_grid = itertools.product(*params_lists)
   result_shape = [len(l) for l in params_lists]

   n_folds = len(folds)
   result_shape.extend(n_folds)

   # last two idx is for score and runtime
   # [param1, param2, ... , nfold, 2]
   result_shape.extend([2])

   if n_jobs < 0:
      results = np.zeros(shape=tuple(result_shape))
      logger.debug('[Evaluate parameters ({:s}) vs. {:s}]'.format(' '.join(params.keys())), scoring)
      # No parallel
      for params in params_grid:
         result_idx = []
         # setup parameters for this run
         for i, p in enumerate(params):
            clf.set_params({eval_params_index[i]: p})
            result_idx.extend(params_lists[i].index(p))
            for fid, fold in enumerate(folds):
               result_idx.extend(fid)
               if type(fold) is tuple:
                  train_idx = fold[0]
                  test_idx = fold[1]
                  bootstrap_xtr = Xtr[train_idx]
                  bootstrap_xtt = Xtr[test_idx]
               else:
                  train_idx = fold
                  bootstrap_xtr = Xtr[train_idx]
                  run_param_job_(clf, xtr=bootstrap_xtr, ytr=ytr,
                                 outputs=results, output_idx=result_idx, scoring=scoring)
   else:
      # run jobs in parallel
      results = np.memmap('tmp/joblib_temp.mmap',
                          dtype=float,
                          mode='w+',
                          shape=result_shape)
      dump(Xtr, 'tmp/input_data')
      input_data = load('tmp/input_data', mmap_mode='r')

      if n_jobs == 0:
         n_jobs = cpu_count()

      # Parallel(n_jobs=n_jobs)(delayed(run_param_job_)(inputs=input_data[np.random.choice(10000, samples_sizes[s])],
      #                                           outputs=results,
      #                                           output_idx=(f, s, m),
      #                                           clf=H3GCDiGraph(verbose=False)dag_true=dag_true)
      #                         for f, s, m in product(range(cv),
      #                                                range(len(samples_sizes)),
      #                                                range(len(methods))))

      #TODO: setup parallel jobs

   # save to pickle file
   output = dict()
   output['data'] = results
   output['folds'] = folds
   output['scoring'] = scoring
   output['parameters'] = params
   if overwrite:
      with open(save_path, 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)
   else:
      time_str = str(datetime.now())
      with open(''.join(save_path, time_str), 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)



def shd(g_origin, g_dest):
   '''
   Compute structural hamming distance of two directed grapahs
   :param g_origin: adjacent matrix for original graph
   :param g_dest: adjacent matrix for destinate graph
   :return: structural hamming distance, how many false links in destinate graph
   '''

   # ensure ndarray
   g_origin = np.array(g_origin)
   g_dest = np.array(g_dest)
   node_size = g_origin.shape[0]
   dist = 0
   missings = 0
   extras = 0
   inverses = 0

   if g_origin.shape[0] != g_dest.shape[0]:
      print 'Graph nodes are different: can not compare, return -1!'
      return -1

   if g_origin.shape[0] == 1:
      dist = 0
      missings = 0
      extras = 0
      inverses = 0
   else:
      dist, missings, extras, inverses = shd(g_origin[np.ix_(range(1, node_size), range(1, node_size))],
                                             g_dest[np.ix_(range(1, node_size), range(1, node_size))])
      for j in range(1, node_size):
         if g_origin[0, j] == 0 and g_origin[j, 0] == 0:
            # no edge in original graph
            if g_dest[0, j] == 1 or g_dest[0, 1] == 1:
               extras += 1
         else:
            # there is edge in original graph
            if g_dest[0, j] == 0 and g_dest[j, 0] == 0:
               # no such edge in destinate graph
               missings += 1
            else:
               if g_origin[0, j] != g_dest[0, j] and g_origin[j, 0] != g_dest[j, 0]:
                  # direction is wrong
                  inverses += 1

      dist = missings + extras + inverses

   return dist, missings, extras, inverses


def cost_rule(g_origin, g_dest, node_size):
    tp_edge = 0
    fn_edge = 0
    tn_edge = 0
    fp_edge = 0
    for j in xrange(1,node_size):
        if g_origin[0,j] == 0 and g_origin[j,0] == 0:
            if g_dest[0,j] == 1 or g_dest[j,0] == 1:
                tn_edge += 1
            elif g_dest[0,j] == 0 and g_dest[j,0] == 0:
                fp_edge += 1
        else:
            if g_dest[0,j] == 0 and g_dest[j,0] == 0:
                fn_edge += 1
            elif g_dest[0,j] == 1 or g_dest[j,0] == 1:
                tp_edge = 0
    return tp_edge, fn_edge, tn_edge, fp_edge
               
            
       
    
def information(g_origin, g_dest):
    g_origin = np.array(g_origin)
    g_dest = np.array(g_dest)
    node_size = g_origin.shape[0]
    tp, fn, tn, fp = cost_rule(g_origin, g_dest, node_size)
    precision = tp /float(tp+fp)
    recall = tp/float(tp+fn)
    error_rate = (fp+fn)/float(tp+fn+tn+fp)
    return precision, recall, error_rate
    
if __name__ == '__main__':
   # unit test find_path

   clf = H3GCDiGraph()
