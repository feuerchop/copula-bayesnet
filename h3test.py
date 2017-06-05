from evaluator import shd
from gcdag import H3GCDiGraph
from helper import *
from pgm2dot import *
import json, pickle
from datetime import datetime as dt
import matplotlib.pylab as plt

datapath = 'Datasets/'
jsonpath = 'jsons/'
figpath = 'Plottings/'

ground_truth = True

if ground_truth:
   network_name = 'credit_loan'
   network = network_name + '.json'
   dataset_name = network_name + '_' + str(10000)
   dataset = dataset_name + '.csv'
   input_data, node_names = read_csv(datapath + dataset)
   dag_true = json2adj(json.load(open(jsonpath + network)), node_names=node_names)
   dot_true = adj2dot(dag_true, node_names=node_names)
   dot_true.render(filename=network_name + '_true', directory=figpath, cleanup=True, view=True)
else:
   dataset_name = 'abalone_4176'
   # dataset_name = 'housing_506'
   dataset = dataset_name + '.csv'
   input_data, node_names = read_csv(datapath + dataset)

sample_size = input_data.shape[0]
# batch = sample_size

# methods:
# methods = ['mle', 'glasso', 'ledoit_wolf', 'spectral', 'pc']
# methods = ['mle', 'glasso', 'ledoit_wolf']
methods = ['ic']
folds = 3
sizes = [100, 500, 1000, 2000]
alphas = np.linspace(0.01, 1, 10, endpoint=True)
res_mat = np.zeros(shape=(folds, len(methods), len(sizes), len(alphas), 2))
for mid, m in enumerate(methods):
   for aid, a in enumerate(alphas):
      clf = H3GCDiGraph(penalty=0.01, method=m, verbose=False, alpha=0.1, vnames=node_names, pval=a)
      for sid, size in enumerate(sizes):
         for f in range(folds):
            bootstrap = input_data[np.random.choice(sample_size, size)]
            start = dt.now()
            clf.fit(bootstrap)
            delta = dt.now() - start
            # dot_raw = adj2dot(clf.dag_raw, node_names=clf.vertexes)
            # dot_moral = adj2dot(clf.dag_moral, node_names=clf.vertexes)
            # render all figs
            # clf.show(output_pdf=dataset_name + '_' + m, dir=figpath)
            # dot_raw.render(filename=dataset_name + '_' + m + '_raw', directory=figpath, cleanup=True, view=False)
            # dot_moral.render(filename=dataset_name + '_' + m + '_moral', directory=figpath, cleanup=True, view=False)
            if ground_truth:
               error, missings, extras, inverses = shd(dag_true, clf.conditional_independences_)
               print '{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format(m, 'fold', 'size', 'alpha', 'SHD',
                                                                                      'missings', 'extras', 'inverses')
               print '{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('', str(f), str(size), str(a),
                                                                                      str(error), str(missings),
                                                                                      str(extras), str(inverses))
               res_mat[f, mid, sid, aid, 0] = error
               res_mat[f, mid, sid, aid, 1] = delta.total_seconds()
# plotting
from Plottings.DataViz import DataViz
from bokeh.plotting import show, gridplot
conf = {'colormap': 'Set2_',
        'width': 600,
        'height': 400,
        'output_file': 'pc_test.html'}

with open('tmp/ic_test.p', 'w') as fd:
   pickle.dump(res_mat, fd)
with open('tmp/ic_test.p', 'r') as fd:
   plotting_data = pickle.load(fd)
   plotting = DataViz(config=conf)
   f1 = plotting.fill_between(xticks=sizes,
                              mean=plotting_data[:, :, :, 1, 0].mean(axis=0),
                              std=plotting_data[:, :, :, 1, 0].std(axis=0),
                              legend=plotting_data['methods'],
                              legend_orientation='horizontal',
                              xlabel='sizes',
                              legend_loc='top_right')
   f2 = plotting.fill_between(xticks=alphas,
                              mean=plotting_data[:, :, 3, :, 0].mean(axis=0),
                              std=plotting_data[:, :, 3, :, 0].std(axis=0),
                              xlabel='alphas',
                              legend=plotting_data['methods'],
                              legend_orientation='horizontal',
                              legend_loc='top_right')
   show(gridplot([f1, f2], ncols=2))
   plotting.send_to_server()