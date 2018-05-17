import networkx as nx 
import os 
import numpy as np


def read_test_graphs():
	graphs = []
	for graph in os.listdir('./test_graphs/')[1:]:
		graphs.append(nx.read_graphml('./test_graphs/' + graph))
    return graphs

def build_dataset(graphs, nodes_number=6):
	for graph in graphs:
	    graph.remove_nodes_from(list(graph.nodes)[nodes_number:])
    return graphs

def laplacian_tensor(x):
    return FloatTensor(np.squeeze(np.asarray(nx.laplacian_matrix(x).todense().astype('float64'))))
