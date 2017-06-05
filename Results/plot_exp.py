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

import matplotlib as mpl
import matplotlib.pylab as plt
import pickle, json
import numpy as np

# set global styles
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['font.size'] = 12
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelsize'] = 'large'

nets = ['example_network', 'credit_loan']
net_names = ['Students', 'Credit Loan']
evals = ['alphas', 'samplesizes']
markers = ['o', 's', 'v', '^']
colors = ['b', 'g', 'r', 'c']

#### plot experiment 1: alphas vs. shd
f1 = plt.figure(figsize=(10, 3))
for i, net in enumerate(nets):
   ax = f1.add_subplot(1, 2, i + 1)
   res_file = ''.join([net, '_', evals[0], '.p'])
   with open(res_file, 'r') as fd:
      results = pickle.load(fd)
      methods = results['methods']
      alphas = results['alphas']
      plot_data = results['data']
      for m, method in enumerate(methods):
         ax.errorbar(alphas[:6], plot_data[:, m, :6, 0].mean(axis=0),
                     yerr=plot_data[:, m, :6, 0].std(axis=0),
                     fmt=colors[m] + markers[m] + '-',
                     mec=colors[m],
                     label=method,
                     mfc='none')
         # ax.fill_between(alphas[:7], y1=plot_data[:, m, :7, 0].mean(axis=0) - plot_data[:, m, :7, 0].std(axis=0),
         #                 y2=plot_data[:, m, :7, 0].mean(axis=0) + plot_data[:, m, :7, 0].std(axis=0),
         #                 facecolor=colors[m], alpha=0.05)
      if i == 0:
         lgd = ax.legend(loc='upper left', ncol=len(methods), bbox_to_anchor=(0.4, 1.21))
         ax.set(ylabel='SHD Total Errors')
      ax.set(xlim=[-0.02, 0.3], ylim=[plot_data[:, m, :6, 0].min() - 1, 1.1 * plot_data[:, m, :6, 0].max()],
             xticks=[0, 0.1, 0.2, 0.3], xlabel=net_names[i])
f1.savefig('../Plottings/exp1.eps', dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artist=[lgd])

#### plot experiment 2: alphas vs. missing/extras/inverses
f2 = plt.figure(figsize=(10, 4))
axes_names = ['Missing edges', 'Extra edges', 'Reversed edges']
for i, net in enumerate(nets):
   for j in range(3):
      ax = plt.subplot(2, 3, i * 3 + j + 1)
      res_file = ''.join([net, '_', evals[0], '.p'])
      with open(res_file, 'r') as fd:
         results = pickle.load(fd)
         methods = results['methods']
         alphas = results['alphas']
         plot_data = results['data']
         for m, method in enumerate(methods):
            ax.plot(alphas[:6], plot_data[:, m, :6, j + 2].mean(axis=0),
                    colors[m] + markers[m],
                    mec=colors[m],
                    label=method,
                    mfc='none')
            # ax.fill_between(alphas[:7], y1=plot_data[:, m, :7, 0].mean(axis=0) - plot_data[:, m, :7, 0].std(axis=0),
            #                 y2=plot_data[:, m, :7, 0].mean(axis=0) + plot_data[:, m, :7, 0].std(axis=0),
            #                 facecolor=colors[m], alpha=0.05)
         if i == 0 and j == 1:
            lgd = ax.legend(loc='upper left', ncol=len(methods), bbox_to_anchor=(-0.6, 1.36))
         if j == 0:
            ax.set(ylabel=net_names[i])
         if i == 1:
            ax.set(xlabel=axes_names[j])
         ax.set(xlim=[-0.02, 0.3], ylim=[plot_data[:, m, :6, j + 2].min() - 1, 1.1 * plot_data[:, m, :6, j + 2].max()],
                xticks=[0, 0.1, 0.2, 0.3],
                yticks=np.linspace(plot_data[:, m, :6, j + 2].min(), plot_data[:, m, :6, j + 2].max(), 5, endpoint=True,
                                   dtype=int))
f2.savefig('../Plottings/exp2.eps', dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artist=[lgd])

#### plot experiment 3: examplesizes vs. shd
f3 = plt.figure(figsize=(8, 4))
axes_names = ['SHD Total Erros', 'Runtime (in seconds)']
plot_idx = np.array([0, 2, 5, 10, 19], dtype=int)
for i, net in enumerate(nets):
   for j in range(2):
      ax = plt.subplot(2, 2, i * 2 + j + 1)
      res_file = ''.join([net, '_', evals[1], '.p'])
      with open(res_file, 'r') as fd:
         results = pickle.load(fd)
         methods = results['methods']
         sizes = np.array(results['samplesizes'])
         plot_data = results['data']
         for m, method in enumerate(methods):
            ax.errorbar(sizes[plot_idx], plot_data[:, m, plot_idx, j].mean(axis=0),
                        yerr=plot_data[:, m, plot_idx, j].std(axis=0),
                        fmt=colors[m] + markers[m] + '-',
                        mec=colors[m],
                        label=method,
                        mfc='none')
            # ax.fill_between(alphas[:7], y1=plot_data[:, m, :7, 0].mean(axis=0) - plot_data[:, m, :7, 0].std(axis=0),
            #                 y2=plot_data[:, m, :7, 0].mean(axis=0) + plot_data[:, m, :7, 0].std(axis=0),
            #                 facecolor=colors[m], alpha=0.05)
         if i == 0 and j == 1:
            lgd = ax.legend(loc='upper left', ncol=len(methods), bbox_to_anchor=(-1.15, 1.38))
         if j == 0:
            ax.set(ylabel=net_names[i])
         if i == 1:
            ax.set(xlabel=axes_names[j])
         ax.set(ylim=[plot_data[:, :, plot_idx, j].min()-1.5, 1.1*plot_data[:, :, plot_idx, j].max()],
                xlim=[50,2050])
f3.savefig('../Plottings/exp3.eps', dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artist=[lgd])

#### plot experiment 4: real exp on abalone
f4 = plt.figure(figsize=(10, 3))
res_file = ''.join(['abalone', '_', evals[1], '.p'])
with open(res_file, 'r') as fd:
   results = pickle.load(fd)
   methods = results['methods']
   sizes = np.array(results['samplesizes'])
   plot_data = results['data']
   ax = plt.subplot(1,2,1)
   bx = plt.subplot(1,2,2)
   for m, method in enumerate(methods):
      ax.errorbar(sizes[:4], plot_data[:, m, :4, 0].mean(axis=0),
                   yerr=plot_data[:, m, :4, 0].std(axis=0),
                   fmt=colors[m] + markers[m] + '-',
                   mec=colors[m],
                   label=method,
                   mfc='none')
      bx.errorbar(sizes[:4], plot_data[:, m, :4, 1].mean(axis=0),
                   yerr=plot_data[:, m, :4, 1].std(axis=0),
                   fmt=colors[m] + markers[m] + '-',
                   mec=colors[m],
                   label=method,
                   mfc='none')
   lgd = ax.legend(loc='upper left', ncol=len(methods), bbox_to_anchor=(0.4, 1.24))
   ax.set(ylabel='Log-likelihood per Instance',
                 xlim=[50,1050], xticks=[100,200,500,1000],
          yticks=[-3,-10,-15])
   bx.set(ylabel='Runtime (in seconds)',
          ylim=[0,8],
                 xlim=[50,1050], xticks=[100,200,500,1000])
f4.savefig('../Plottings/exp4.eps', dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artist=[lgd])
plt.show()
