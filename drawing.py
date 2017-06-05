#!/usr/bin/env python
# coding: utf-8

'''
Convert JSON input file in libpgm format to Graphviz dot format

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import numpy as np
from graphviz import Digraph


def json2dot(json_obj, comment='default graphviz object'):
   '''
   Convert dict to Graphviz
   :param file: Input JSON file in libpgm format. see:
                http://pythonhosted.org/libpgm/unittestlgdict.html
                for an example, which defines a linear Gaussian bayes network
   :return: a dot object containing the graph
   '''

   vdata = json_obj['Vdata']
   nodes = vdata.keys()

   # parse the JSON and construct the dot
   dot = Digraph(comment=comment)
   dot.node_attr.update(color='lightblue2', style='filled')
   node_list = dict()
   node_id = 0
   for node in nodes:
      node_list[node_id] = node
      node_id += 1
      dot.node(node)
      if vdata[node]['parents'] is not None:
         for parent_id, parent in enumerate(vdata[node]['parents']):
            if not node_list.has_key(parent):
               node_list[node_id] = parent
               node_id += 1
               dot.node(parent)
            if len(vdata[node]['mean_scal']) == len(vdata[node]['parents']):
               dot.edge(parent, node, label=str(vdata[node]['mean_scal'][parent_id]))
            else:
               dot.edge(parent, node)
      else:
         pass

   return dot


def adj2dot(adj_mat, node_names=None):
   '''
   from adjacent matrix to dot
   :param adj_mat: adjacent matrix
   :param node_names: a list of node names
   :return: dot object
   '''

   import numpy as np
   dot = Digraph()
   # dot.node_attr.update(color='lightblue2',)
   adj_mat = np.array(adj_mat)
   node_size = adj_mat.shape[0]
   if node_names is None or len(node_names) != node_size:
      node_names = [str(id) for id in range(node_size)]

   for i in range(node_size):
      dot.node(name=node_names[i])

   for i in range(node_size):
      for j in range(i + 1, node_size):
         if adj_mat[i, j] == 1 and adj_mat[j, i] == 1:
            # undirected edge
            dot.edge(node_names[i], node_names[j], arrowhead='none', arrowtail='none')
         elif adj_mat[i, j] == 1 and adj_mat[j, i] == 0:
            dot.edge(node_names[i], node_names[j])
         elif adj_mat[i, j] == 0 and adj_mat[j, i] == 1:
            dot.edge(node_names[j], node_names[i])

   return dot


def json2adj(json_obj, node_names):
   '''
   Convert graph json object to adjacent matrix
   :param json_obj: Input JSON in libpgm format. see:
                http://pythonhosted.org/libpgm/unittestlgdict.html
                for an example, which defines a linear Gaussian bayes network
          node_names: a list of nodes, make sure node order is correct
   :return: a dxd ndarry for adjacent matrix
   '''

   vdata = json_obj['Vdata']
   if vdata is None:
      print 'No vdata found in JSON, exit!'
      return None

   nodes = node_names
   node_size = len(nodes)
   adjmat = np.zeros(shape=(node_size, node_size))

   for i in range(node_size):
      for j in range(i + 1, node_size):
         if vdata[nodes[i]]['children'] is not None:
            if nodes[j] in vdata[nodes[i]]['children']:
               adjmat[i, j] = 1  # i->j
            else:
               adjmat[i, j] = 0
         else:
            adjmat[i, j] = 0
         if vdata[nodes[j]]['children'] is not None:
            if nodes[i] in vdata[nodes[j]]['children']:
               adjmat[j, i] = 1  # j->i
            else:
               adjmat[j, i] = 0  # j->i
         else:
            adjmat[j, i] = 0

   return adjmat


def adj2json(adjmat, node_names=None):
   '''
   convert adjacent matrix to json object
   we only support create linear gaussian node for now
   :param adjmat: dxd adjacent matrix
   :param node_names: a list of node names
   :return: JSON object
   '''

   if adjmat is None:
      print 'Conditional independence matrix is none.. exit!'
      return None
   else:
      adjmat = np.array(adjmat)
      node_size = adjmat.shape[0]

   if node_names is None:
      node_names = [str(id) for id in range(node_size)]

   graph = dict()
   graph['V'] = node_names
   graph['E'] = []
   graph['Vdata'] = dict()

   for i in range(node_size):
      for j in range(node_size):
         if i != j and adjmat[i, j] == 1:
            graph['E'].append([node_names[i], node_names[j]])
            if not graph['Vdata'].has_key(node_names[i]):
               graph['Vdata'][node_names[i]] = dict({
                  "mean_base": 0,
                  "mean_scal": [],
                  "parents": [],
                  "variance": 0,
                  "type": "lg",
                  "children": []
               })
            if not graph['Vdata'].has_key(node_names[j]):
               graph['Vdata'][node_names[j]] = dict({
                  "mean_base": 0,
                  "mean_scal": [],
                  "parents": [],
                  "variance": 0,
                  "type": "lg",
                  "children": []
               })
            graph['Vdata'][node_names[i]]['children'].append(node_names[j])
            graph['Vdata'][node_names[j]]['parents'].append(node_names[i])

   return graph


if __name__ == '__main__':
   import json

   # unit test adj2dot
   m1 = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0]]
   print 'test adjmat'
   print m1
   # dot = adj2dot(m1)
   # dot.render('untitled', view=True)
   # unit test adj2json
   graph_json = adj2json(m1, node_names=['A', 'B', 'C', 'D'])
   # unit test json2adj
   adj = json2adj(graph_json, node_names=graph_json['V'])
   print adj
