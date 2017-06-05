#!/usr/bin/env python
# coding: utf-8

'''
Experiments for ECML17

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import argparse
import json
from datetime import datetime
from itertools import product
from multiprocessing import cpu_count
from joblib import Parallel, delayed, dump, load
from evaluator import *
from gcdag import H3GCDiGraph
from helper import *
from drawing import *
from bokeh.plotting import show, gridplot
from Plottings.DataViz import DataViz
from sklearn.model_selection import KFold

# globals

datapath = 'Datasets/'
jsonpath = 'jsons/'
figpath = 'Figures/'
output_path = 'Results/'


# run joblib
def run_job_(inputs, outputs, output_idx, clf, Xtt=None, dag_true=None):
   start = datetime.now()
   clf.fit(inputs)
   clf.fit_structure(clf.precision_)
   delta = datetime.now() - start
   if dag_true is not None:
      errors, missings, extras, inverses = shd(dag_true, clf.conditional_independences_)
      outputs[output_idx[0], output_idx[1], output_idx[2], 0] = errors
      outputs[output_idx[0], output_idx[1], output_idx[2], 1] = delta.total_seconds()
      outputs[output_idx[0], output_idx[1], output_idx[2], 2] = missings
      outputs[output_idx[0], output_idx[1], output_idx[2], 3] = extras
      outputs[output_idx[0], output_idx[1], output_idx[2], 4] = inverses
   elif Xtt is not None:
      errors = clf.score_function(Xtt)          # log-likelihood per instance
      outputs[output_idx[0], output_idx[1], output_idx[2], 0] = errors
      outputs[output_idx[0], output_idx[1], output_idx[2], 1] = delta.total_seconds()


# EVALUATE REAL SAMPLES
def eval_realsamples(net=None,
                     csv_file='',
                     sizes = [100, 200, 500, 1000, 2000, 4000],
                     n_sizes=3,
                     folds=5,
                     penalty=0.01,
                     alpha=0.1,
                     methods=['mle', 'glasso', 'ledoit_wolf'],
                     save_path=output_path,
                     logger=logging.getLogger('gcdag.experiments.realsamples.log'),
                     overwrite=False,
                     n_jobs=-1):
   input_data, node_names = read_csv(csv_file)
   N, d = input_data.shape
   if sizes:
      sample_sizes = sizes
   else:
      sample_sizes = np.linspace(100, N, n_sizes, endpoint=True, dtype=int).tolist()

   if n_jobs < 0:
      results = np.zeros(shape=(folds, len(methods), len(sample_sizes), 5))
      logger.debug('[Experiment sample sizes vs. per-instance log-likelihood]')
      # No parallel
      logger.debug(
         '{:10s} {:10s} {:10s} {:10s} {:10s}'.format('Method', 'Fold', 'Size', '(perInst) log-ll', 'Time elapsed'))
      for i, m in enumerate(methods):
         clf = H3GCDiGraph(penalty=penalty, method=m, verbose=False, alpha=alpha, vnames=node_names)
         for j, size in enumerate(sample_sizes):
            bootstrap = input_data[np.random.choice(N, size)]
            kfold = KFold(shuffle=True, n_splits=folds)
            splits = kfold.split(bootstrap)
            for fid, split in enumerate(splits):
               train_idx, test_idx = split[0], split[1]
               run_job_(bootstrap[train_idx], results, (fid, i, j), clf, dag_true=None, Xtt=bootstrap[test_idx])
               logger.debug('{:10s} {:10s} {:10s} {:10s} {:10s}'.format(m, str(fid), str(size),
                                                                        str(results[fid, i, j, 0]),
                                                                        str(results[fid, i, j, 1])))
   else:
      results = np.memmap('tmp/joblib_temp.mmap',
                          dtype=float,
                          mode='w+',
                          shape=(folds, len(methods), len(sample_sizes), 5))
      dump(input_data, 'tmp/input_data')
      input_data = load('tmp/input_data', mmap_mode='r')

      if n_jobs == 0:
         n_jobs = cpu_count()

      splits_set = []
      bootstraps = []
      for s in sample_sizes:
         bootstraps.append(np.random.choice(N, s))
         kfold = KFold(shuffle=True, n_splits=folds)
         splits = kfold.split(input_data[bootstraps[-1]])
         split_folds = []
         for tr_id, tt_id in splits:
            split_folds.append((tr_id, tt_id))
         splits_set.append(split_folds)

      Parallel(n_jobs=n_jobs)(delayed(run_job_)(inputs=input_data[bootstraps[s]][splits_set[s][f][0]],
                                                outputs=results,
                                                output_idx=(f, m, s),
                                                clf=H3GCDiGraph(penalty=penalty,
                                                                method=methods[m],
                                                                verbose=False,
                                                                alpha=alpha,
                                                                vnames=node_names),
                                                dag_true=None,
                                                Xtt=input_data[bootstraps[s]][splits_set[s][f][1]])
                              for f, m, s in product(range(folds),
                                                     range(len(methods)),
                                                     range(len(sample_sizes))))

   # save to pickle file
   output = dict()
   output['data'] = results
   output['folds'] = folds
   output['samplesizes'] = sample_sizes
   output['methods'] = methods
   output['nodes'] = node_names
   output['parameters'] = {'penalty': penalty,
                           'alpha': alpha}

   if overwrite:
      with open(''.join([save_path, csv_file.split('/')[-1].split('_')[0], '_samplesizes', '.p']), 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)
   else:
      time_str = str(datetime.now())
      with open(''.join([save_path, csv_file.split('/')[-1].split('_')[0], '_samplesizes', '_', time_str, '.p']),
                'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)

   return fd.name


# EVALUATE SAMPLESIZES
def eval_samplesizes(net=None,
                     csv_file='',
                     sizes=range(100, 2100, 100),
                     folds=5,
                     penalty=0.01,
                     alpha=0.1,
                     methods=['mle', 'glasso', 'ledoit_wolf', 'ic'],
                     save_path=output_path,
                     logger=logging.getLogger('gcdag.experiments.samplesizes.log'),
                     overwrite=False,
                     n_jobs=-1):
   # initialization
   network_name = net
   network = network_name + '.json'
   dataset_name = network_name + '_' + str(10000)
   dataset = dataset_name + '.csv'
   input_data, node_names = read_csv(datapath + dataset)
   dag_true = json2adj(json.load(open(jsonpath + network)), node_names=node_names)

   samples_sizes = sizes
   cv = folds
   methods = methods

   # init results array
   # :,:,0,: for shd erro, :,:,1,: for runtime

   if n_jobs < 0:
      results = np.zeros(shape=(cv, len(methods), len(samples_sizes), 5))
      logger.debug('[Experiment sample sizes vs. SHD error]')
      # No parallel
      logger.debug('{:10s} {:10s} {:10s} {:10s} {:10s}'.format('Method', 'Fold', 'Size', 'SHD', 'Time elapsed'))
      for i, m in enumerate(methods):
         clf = H3GCDiGraph(penalty=penalty, method=m, verbose=False, alpha=alpha, vnames=node_names)
         for j, size in enumerate(samples_sizes):
            bootstrap = input_data[np.random.choice(10000, size)]
            for fold in range(cv):
               run_job_(bootstrap, results, (fold, i, j), clf, dag_true=dag_true)
               logger.debug('{:10s} {:10s} {:10s} {:10s} {:10s}'.format(m, str(fold), str(size),
                                                                        str(results[fold, i, j, 0]),
                                                                        str(results[fold, i, j, 1])))
   else:
      results = np.memmap('tmp/joblib_temp.mmap',
                          dtype=float,
                          mode='w+',
                          shape=(cv, len(methods), len(samples_sizes), 5))
      dump(input_data, 'tmp/input_data')
      input_data = load('tmp/input_data', mmap_mode='r')

      if n_jobs == 0:
         n_jobs = cpu_count()

      Parallel(n_jobs=n_jobs)(delayed(run_job_)(inputs=input_data[np.random.choice(10000, samples_sizes[s])],
                                                outputs=results,
                                                output_idx=(f, m, s),
                                                clf=H3GCDiGraph(penalty=penalty,
                                                                method=methods[m],
                                                                verbose=False,
                                                                alpha=alpha,
                                                                vnames=node_names),
                                                dag_true=dag_true)
                              for f, m, s in product(range(cv),
                                                     range(len(methods)),
                                                     range(len(samples_sizes))))

   # save to pickle file
   output = dict()
   output['data'] = results
   output['folds'] = cv
   output['samplesizes'] = samples_sizes
   output['methods'] = methods
   output['nodes'] = node_names
   output['parameters'] = {'penalty': penalty,
                           'alpha': alpha}

   if overwrite:
      with open(''.join([save_path, net, '_samplesizes', '.p']), 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)
   else:
      time_str = str(datetime.now())
      with open(''.join([save_path, net, '_samplesizes', '_', time_str, '.p']), 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)

   return fd.name


# EVALUATE ALPHAS
def eval_alphas(net=None,
                csv_file='',
                size=1000,
                folds=5,
                penalty=0.01,
                alphas=np.linspace(0, 0.5, 10, endpoint=True),
                methods=['mle', 'glasso', 'ledoit_wolf'],
                save_path=output_path,
                logger=logging.getLogger('gcdag.experiments.alphas.log'),
                overwrite=False,
                n_jobs=-1):
   # initialization
   network_name = net
   network = network_name + '.json'
   dataset_name = network_name + '_' + str(10000)
   dataset = dataset_name + '.csv'
   input_data, node_names = read_csv(datapath + dataset)
   dag_true = json2adj(json.load(open(jsonpath + network)), node_names=node_names)

   cv = folds
   methods = methods

   # init results array
   # :,:,0,: for shd erro, :,:,1,: for runtime

   if n_jobs < 0:
      results = np.zeros(shape=(cv, len(methods), len(alphas), 5))
      logger.debug('[Experiment alphas vs. SHD error]')
      # No parallel
      logger.debug('{:10s} {:10s} {:10s} {:10s} {:10s}'.format('Method', 'Fold', 'alpha', 'SHD', 'Time elapsed'))
      for i, m in enumerate(methods):
         for j, alpha in enumerate(alphas):
            clf = H3GCDiGraph(penalty=penalty, method=m, verbose=False, alpha=alpha, vnames=node_names)
            for fold in range(cv):
               bootstrap = input_data[np.random.choice(10000, size)]
               run_job_(bootstrap, results, (fold, i, j), clf, dag_true=dag_true)
               logger.debug('{:10s} {:10s} {:10s} {:10s} {:10s}'.format(m, str(fold), str(alpha),
                                                                        str(results[fold, i, j, 0]),
                                                                        str(results[fold, i, j, 1])))
   else:
      results = np.memmap('tmp/joblib_temp.mmap',
                          dtype=float,
                          mode='w+',
                          shape=(cv, len(methods), len(alphas), 5))
      dump(input_data, 'tmp/input_data')
      input_data = load('tmp/input_data', mmap_mode='r')

      if n_jobs == 0:
         n_jobs = cpu_count()

      Parallel(n_jobs=n_jobs)(delayed(run_job_)(inputs=input_data[np.random.choice(10000, size)],
                                                outputs=results,
                                                output_idx=(f, m, a),
                                                clf=H3GCDiGraph(penalty=penalty,
                                                                method=methods[m],
                                                                verbose=False,
                                                                alpha=alphas[a],
                                                                vnames=node_names),
                                                dag_true=dag_true)
                              for f, m, a in product(range(cv),
                                                     range(len(methods)),
                                                     range(len(alphas))))

   # save to pickle file
   output = dict()
   output['data'] = results
   output['folds'] = cv
   output['alphas'] = alphas
   output['methods'] = methods
   output['nodes'] = node_names
   output['parameters'] = {'penalty': penalty,
                           'size': size}

   if overwrite:
      with open(''.join([save_path, net, '_alphas', '.p']), 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)
   else:
      time_str = str(datetime.now())
      with open(''.join([save_path, net, '_alphas', '_', time_str, '.p']), 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)
   # return the filename
   return fd.name


# EVALUATE PENALTY
def eval_penalty(net=None,
                 csv_file='',
                 size=1000,
                 folds=5,
                 penalty=np.linspace(0, 0.2, 10, endpoint=True),
                 alpha=0.1,
                 methods=['mle', 'glasso'],
                 save_path=output_path,
                 logger=logging.getLogger('gcdag.experiments.penalty.log'),
                 overwrite=False,
                 n_jobs=-1):
   # initialization
   network_name = net
   network = network_name + '.json'
   dataset_name = network_name + '_' + str(10000)
   dataset = dataset_name + '.csv'
   input_data, node_names = read_csv(datapath + dataset)
   dag_true = json2adj(json.load(open(jsonpath + network)), node_names=node_names)

   cv = folds
   methods = methods

   # init results array
   # :,:,0,: for shd erro, :,:,1,: for runtime

   if n_jobs < 0:
      results = np.zeros(shape=(cv, len(methods), len(penalty), 5))
      logger.debug('[Experiment penalty vs. SHD error]')
      # No parallel
      logger.debug('{:10s} {:10s} {:10s} {:10s} {:10s}'.format('Method', 'Fold', 'penalty', 'SHD', 'Time elapsed'))
      for i, m in enumerate(methods):
         for j, pen in enumerate(penalty):
            clf = H3GCDiGraph(penalty=pen, method=m, verbose=False, alpha=alpha, vnames=node_names)
            for fold in range(cv):
               bootstrap = input_data[np.random.choice(10000, size)]
               run_job_(bootstrap, results, (fold, i, j), clf, dag_true=dag_true)
               logger.debug('{:10s} {:10s} {:10s} {:10s} {:10s}'.format(m, str(fold), str(pen),
                                                                        str(results[fold, i, j, 0]),
                                                                        str(results[fold, i, j, 1])))
   else:
      results = np.memmap('tmp/joblib_temp.mmap',
                          dtype=float,
                          mode='w+',
                          shape=(cv, len(methods), len(penalty), 5))
      dump(input_data, 'tmp/input_data')
      input_data = load('tmp/input_data', mmap_mode='r')

      if n_jobs == 0:
         n_jobs = cpu_count()

      Parallel(n_jobs=n_jobs)(delayed(run_job_)(inputs=input_data[np.random.choice(10000, size)],
                                                outputs=results,
                                                output_idx=(f, m, p),
                                                clf=H3GCDiGraph(penalty=penalty[p],
                                                                method=methods[m],
                                                                verbose=False,
                                                                alpha=alpha,
                                                                vnames=node_names),
                                                dag_true=dag_true)
                              for f, m, p in product(range(cv),
                                                     range(len(methods)),
                                                     range(len(penalty))))

   # save to pickle file
   output = dict()
   output['data'] = results
   output['folds'] = cv
   output['penalty'] = penalty
   output['methods'] = methods
   output['nodes'] = node_names
   output['parameters'] = {'alpha': alpha,
                           'size': size}

   if overwrite:
      with open(''.join([save_path, net, '_penalty', '.p']), 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)
   else:
      time_str = str(datetime.now())
      with open(''.join([save_path, net, '_penalty', '_', time_str, '.p']), 'w') as fd:
         print colored('Done! writting to ' + fd.name, 'red')
         pickle.dump(output, fd)
   # return the filename
   return fd.name


# main ---

argparser = argparse.ArgumentParser(prog='experiments',
                                    usage="this program conducts experiments for ECML17 paper.",
                                    description='''
                                       experiments code for ECML'17
                                    ''',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--net", help="network JSON file name ,e.g., credit_loan", default='credit_loan')
argparser.add_argument("--csv", help="use csv dataset for real dataset evaluation", default='')
argparser.add_argument("--eval_func", help="Evaluation function name")
argparser.add_argument("--folds", help="How many folds to conduct.. ", default=3, type=int)
argparser.add_argument("--save_path", help="Where to save experiments results.", default=output_path)
argparser.add_argument("--ncpus", help="How many cpus we use. defaults use all available CPUs.", type=int, default=0)
argparser.add_argument("--view", help="show the results in plotting.", action='store_true')
argparser.add_argument("--view_only", help="No run, only view.", action='store_true')
args = argparser.parse_args()

net = args.net
csv_file = args.csv
func = args.eval_func
save_path = args.save_path
view = args.view
view_only = args.view_only
ncpus = args.ncpus
folds = args.folds
logger = logging.getLogger('gcdag.experiments.log')
logging.basicConfig(format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s', )
logger.setLevel(logging.DEBUG)

xticks_key = {'eval_samplesizes': 'samplesizes',
              'eval_alphas': 'alphas',
              'eval_penalty': 'penalty',
              'eval_realsamples': 'samplesizes'}

if csv_file:
   output_html = ''.join([csv_file.split('/')[-1].split('.')[0], xticks_key[func], '.html'])
   ylabel = 'log-likelihood per Instance'
   saved_file = ''.join([save_path, csv_file.split('/')[-1].split('_')[0], '_', xticks_key[func], '.p'])
else:
   output_html = ''.join([net, xticks_key[func], '.html'])
   ylabel = 'SHD'
   saved_file = ''.join([save_path, net, '_', xticks_key[func], '.p'])

conf = {'colormap': 'Set2_',
        'width': 600,
        'height': 400,
        'output_file': output_html}

if not view_only:
   saved_file = locals()[func](net=net,
                               csv_file=csv_file,
                               # sizes=[100, 500, 1000, 2000],
                               folds=folds,
                               # penalty=0.1,
                               # alpha=0.15,
                               # methods=['mle', 'glasso', 'ledoit_wolf', 'pc'],
                               save_path=save_path,
                               logger=logger,
                               overwrite=True,
                               n_jobs=ncpus)
   if view:
      with open(saved_file, 'r') as fd:
         plotting_data = pickle.load(fd)
         plotting = DataViz(config=conf)
         f1 = plotting.fill_between(xticks=plotting_data[xticks_key[func]],
                                    mean=plotting_data['data'][:, :, :, 0].mean(axis=0),
                                    std=plotting_data['data'][:, :, :, 0].std(axis=0), xlabel=xticks_key[func],
                                    ylabel=ylabel,
                                    legend=plotting_data['methods'],
                                    legend_orientation='horizontal',
                                    legend_loc='top_right')
         f2 = plotting.fill_between(xticks=plotting_data[xticks_key[func]],
                                    mean=plotting_data['data'][:, :, :, 1].mean(axis=0),
                                    std=plotting_data['data'][:, :, :, 1].std(axis=0), xlabel=xticks_key[func],
                                    ylabel='Runtime',
                                    legend=plotting_data['methods'],
                                    legend_orientation='horizontal',
                                    legend_loc='top_right')
         show(gridplot([f1, f2], ncols=2))
         plotting.send_to_server()

else:
   with open(saved_file, 'r') as fd:
      plotting_data = pickle.load(fd)
      plotting = DataViz(config=conf)
      f1 = plotting.fill_between(xticks=plotting_data[xticks_key[func]],
                                 mean=plotting_data['data'][:, :, :, 0].mean(axis=0),
                                 std=plotting_data['data'][:, :, :, 0].std(axis=0), xlabel=xticks_key[func],
                                 ylabel=ylabel,
                                 legend=plotting_data['methods'],
                                 legend_orientation='horizontal',
                                 legend_loc='top_right')
      f2 = plotting.fill_between(xticks=plotting_data[xticks_key[func]],
                                 mean=plotting_data['data'][:, :, :, 1].mean(axis=0),
                                 std=plotting_data['data'][:, :, :, 1].std(axis=0), xlabel=xticks_key[func],
                                 ylabel='Runtime',
                                 legend=plotting_data['methods'],
                                 legend_orientation='horizontal',
                                 legend_loc='top_right')
      show(gridplot([f1, f2], ncols=2))
      plotting.send_to_server()
