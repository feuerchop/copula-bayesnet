#!/usr/bin/env python
# coding: utf-8

'''
Helper functions for gaussian copula structure learning

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import numpy as np
from numpy.linalg import inv
import itertools
from termcolor import colored
import pandas as pd
import os
import scipy.io

# TODO: clean code
base = os.getcwd()
def get_neighbors(dag, node):
   '''
   Get neighbours of a node
   :param dag: adjacent matrix
   :param node: node index
   :return: unknown, incoming, outgoing
   '''

   unknown = list([])
   incoming = list([])
   outgoing = list([])

   dag = np.array(dag)
   # complexity O(d)
   for j in range(dag.shape[0]):
      if j != node and dag[j, node] == 1 and dag[node, j] == 1:
         # unknown edge
         unknown.append(j)
      if dag[j, node] == 1 and dag[node, j] == 0:
         # incoming edge
         incoming.append(j)
      if dag[j, node] == 0 and dag[node, j] == 1:
         # incoming edge
         outgoing.append(j)

   return unknown, incoming, outgoing


def find_path(pdag, p, q):
   '''
   Find a path from node p to q using dijisktra algorithm
   :param pdag: partial DAG
   :param p: start node p
   :param q: end node q
   :return: flag if we can find a path, 1 yes, 0 no
   '''

   flag = 0
   path = []
   pdag = np.array(pdag)
   node_size = pdag.shape[0]
   dist = np.inf * np.ones(shape=(1, node_size)).ravel()
   previous = -1 * np.ones(shape=(1, node_size)).ravel()
   nodes = range(node_size)
   dist[p] = 0  # dist to itself is zero
   while len(nodes) > 0:
      nearest_node = nodes[np.argmin(dist[nodes])]
      if dist[nearest_node] == np.inf:
         break
      else:
         nodes.remove(nearest_node)
         undirected, incoming, outgoing = get_neighbors(pdag, nearest_node)
         for v in outgoing:
            if dist[nearest_node] + 1 < dist[v]:
               dist[v] = dist[nearest_node] + 1
               previous[v] = nearest_node

   if dist[q] != np.inf:
      flag = 1
      path.extend([q])
      while previous[q] != -1:
         path.extend([int(previous[q])])
         q = int(previous[q])
      return flag, path


def get_dependencies(dag, p, q):
   '''
   return markov blanket of both nodes
   :param dag: DAG adjecant matrix
   :param p: node index
   :param q: node index
   :return: list of dependent node indices
   '''

   p_unknown, p_incoming, p_outgoing = get_neighbors(dag, p)
   q_unknown, q_incoming, q_outgoing = get_neighbors(dag, q)
   set_p = set(p_unknown).union(set(p_incoming)).union(set(p_outgoing))
   set_q = set(q_unknown).union(set(q_incoming)).union(set(q_outgoing))
   markov_blanket = set_p.union(set_q)
   markov_blanket.remove(p)
   markov_blanket.remove(q)
   return list(markov_blanket)


def is_marriged_edge(graph, corr, p, q, alpha, verbose=False):
   '''
   find out if an edge is a married edge brought by colliders, p and q are the nodes of the edge
   :param graph: undirected graph
   :param corr: correlation matrix
   :param p: node index p
   :param q: node index q
   :param alpha: threshold alpha
   :return: flag and colliders
   '''

   # TODO: for convention, use pythonic style: carmel_function_call, NOT CarmelFunctionCall
   # TOOD: @chingyu unittest on 3-nodes

   flag = -1
   collider_candidates = find_collider_candiadtes(graph, p, q)
   if collider_candidates is None:
      # if nodes are not bridged, return None
      print 'Nodes are not bridged, exit!'
      return None

   dependencies = get_dependencies(graph, p, q)
   for i in xrange(len(collider_candidates)):
      if verbose:
         print colored('Evaluate {:d}-collider(s) ...'.format(i+1), color='yellow')
      # max. so many colliders for one edge, loop breaks util we find the real collider
      collider_comb = nchoosek(collider_candidates, i + 1)  # find combination of colliders of size i+1
      for j in xrange(collider_comb.shape[0]):
         # travel all collider combinations
         colliders = collider_comb[j, :]
         if verbose:
            print colored('for collider(s) {:s} ...'.format(str(colliders)), color='yellow')
         set_without_collider = [p] + [q] + [elem for elem in dependencies if elem not in colliders]
         set_with_collider = [p] + [q] + dependencies
         part_subcorr = corr[np.ix_(set_without_collider, set_without_collider)]
         full_subcorr = corr[np.ix_(set_with_collider, set_with_collider)]
         inv_part_subcorr = scale_matrix(inv(
            part_subcorr + np.identity(len(set_without_collider)) * 1e-5))  # times 1e-5 to ensure diagonal is not zero
         inv_full_subcorr = scale_matrix(inv(
            full_subcorr + np.identity(len(set_with_collider)) * 1e-5))
         if verbose:
            print colored('inverse correlation without colliders ...', color='yellow')
            print colored('{:s}'.format(inv_part_subcorr), color='yellow')
            print colored('inverse correlation with colliders ...', color='yellow')
            print colored('{:s}'.format(inv_full_subcorr), color='yellow')
         if abs(inv_part_subcorr[0, 1]) < alpha and abs(inv_full_subcorr[0, 1]) > alpha:
            # by removing the colliders, p and q become independent
            if verbose:
               print colored('+ Found collider {:s} for edge [%d, %d]'.format(str(colliders), p, q), color='yellow')
            flag = 1
            return flag, colliders

   # otherwise return -1 and empty colliders
   return flag, []


def scale_matrix(m):
   '''
   scale matrix by set diagonals as ones, and p(i,j) = p(i,j)/sqrt(p(i,i)*p(j, j))
   :param mat:
   :return: scaled ndarray
   '''

   d = m.shape[0]
   mat = m.copy()
   for i in range(d):
      for j in range(i + 1, d):
         mat[i, j] = mat[i, j] / np.sqrt(mat[i, i] * mat[j, j])
         mat[j, i] = mat[j, i] / np.sqrt(mat[j, j] * mat[i, i])
   for i in range(d):
      mat[i, i] = 1
   return mat


def set_marridged_edge_free(dag, i, j, colliders):
   '''
   SETMARRIEDEDGEFREE Summary of this function goes here
   suppose the edge (i, j) is a married edge brought by colliders, then
   set it free and configure the correct direction to colliders

   :param dag: DAG adjacent matrix
   :param i: node i
   :param j: node j
   :param colliders: list of collider indices
   :return: untangled adjacent matrix
   '''

   dag[i, j] = dag[j, i] = 0
   for c in colliders:
      dag[i, c] = 1
      dag[c, i] = 0
      dag[j, c] = 1
      dag[c, j] = 0
   return dag


def nchoosek(v, k):
   '''
   This method gives all possible combinations
   v: vector
   k: integer
   return: numpy array
   '''
   import itertools
   # find all combinations
   result = list(itertools.combinations(v, k))
   return np.array(result)


def find_collider_candiadtes(graph, p, q):
   '''
   return: list
   '''

   # make sure graph is an array
   graph = np.array(graph)

   candidates = []
   if graph[p, q] != 1:
      print 'Given nodes are not bridged, exit!'
      return None
   else:
      for i in xrange(graph.shape[1]):
         # make sure they are not themselves
         if p != i and q != i and graph[p, i] == 1 and graph[q, i] == 1:
            # candidates = np.concatenate(candidates, i, axis = 0) # in matlab is [candidates, i]
            candidates.append(i)
   return candidates


def is_diamond(graph, p, q):
   '''
   check if there is a diamond structure between p, q which means
   there are two paths from p to q, p--w-->q and p--z-->q, p and q are
   connected as well. Note that p--w and p--z are not directed. w and z are
   not connected. In this case, p should point to q unless it will introduce
   new V-structure.
   :param graph: adjacent matrix
   :param p: node index p
   :param q: node index q
   :return: boolean indicate if p-q forms a diamond
   '''

   undirected_p, _, _ = get_neighbors(graph, p)
   _, incoming_q, _ = get_neighbors(graph, q)

   if len(undirected_p) < 3 or q not in undirected_p:
      return False

   middle_nodes = [node for node in undirected_p if node in incoming_q]
   if len(middle_nodes) < 2:
      return False

   for w, z in nchoosek(middle_nodes, 2):
      if graph[w, z] == 1 or graph[z, w] == 1:
         return False
      else:
         graph[q, p] = 0
         return True


def read_csv(filename, header_line=True):
   '''
   load dataset from csv file
   :param filename: file path to csv
   :return: training features, a list of node names
   '''
   import csv

   csv_file = open(filename, 'r')
   reader = csv.reader(csv_file)
   cnt = 0
   input_data = []
   nodes = []
   for row in reader:
      if cnt == 0 and header_line:
         nodes = row
      else:
         input_data.append([float(elem) for elem in row])
      cnt += 1
   csv_file.close()
   return np.array(input_data), nodes


def to_adjacent_matrix(m, threshold=5e-3):
   '''
   given a numeric dxd matrix, convert it adjcent matrix
   :param m: dxd ndarray
   :return: dxd binary adjacent matrix
   '''

   m[np.where(abs(m) > threshold)] = 1
   # otherwise make it zero
   m[np.where(abs(m) <= threshold)] = 0
   return m


def get_unknown_edges(graph):
   '''
   get undirected connected edges from adjacent graph
   :param graph: dxd binary ndarray
   :return: a list of 2-dimension lists, each for one edge
   '''

   # TODO: can be determined dynamically

   unknown_edges = []
   graph = np.array(graph)
   node_size = graph.shape[0]
   for i in range(node_size):
      for j in range(i + 1, node_size):
         if graph[i, j] == 1 and graph[j, i] == 1:
            unknown_edges.append([i, j])

   return unknown_edges

def load_abalone_data():
    '''
    return data which contains 8 attributes
    '''

    
    data_path = os.path.join(base, "Datasets/abalone/abalone.data")
    print data_path
    df=pd.read_csv(data_path, sep=',')
    df.columns=['Sex', 'Length', 'Diameter', 'Height','Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Ring']
    # delete attribute 'Sex'
    del df['Sex']
    return df.values, list(df)

def load_boston_housing_data():

    data_path = os.path.join(base, "Datasets/BostonHousing/housing.data")
    print data_path
    df = pd.read_csv(data_path, sep='\s+',header = None)
    df.columns = ['CRIM', 'ZN','INDUS','CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX','PTRATIO', 'B', 'LSTAT','MEDV']
    return df.values, list(df) 
    
def load_fake_7_nodes():

    
    data_path = os.path.join(base,"Datasets/fake/node7.mat")
    mat = scipy.io.loadmat(data_path)
    nodes = ['A','B','C','D','E','F','G']
    return mat['bnData7'],nodes

def csv2dict(csv):
   '''
   convert csv data to a list of dicts
   :return: dict
   '''

   data, nodes = read_csv(csv)
   res = list([])
   for row_id in range(data.shape[0]):
      instance = dict()
      for i, n in enumerate(nodes):
         instance[n] = data[row_id, i]
      res.append(instance)
   return res


if __name__ == '__main__':
   # unit test find_path
   pdag = [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 1, 1, 0], [1, 0, 0, 1, 1], [0, 0, 0, 0, 1]]
   # f, p = find_path(pdag, 2, 4)
   print get_dependencies(pdag, 2, 3)
   print find_collider_candiadtes(pdag, 4, 3)
   print nchoosek([1, 2, 3, 4, 5], 4)

   # unit test read_csv
   data, names = read_csv('example_network_100.csv')
