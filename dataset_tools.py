import networkx as nx 
import os 
import numpy as np
from torch import FloatTensor
import matplotlib.pyplot as plt


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

def adjacency_tensor(x):
	return FloatTensor(np.squeeze(np.asarray(nx.adjacency_matrix(x).todense().astype('float64'))))

def draw_graph_from_laplacian(x, label, ax):
	temp = x.data.numpy()
	for i in range(len(temp)):
		temp[i, i] = 0 
	nx.draw_networkx(nx.from_numpy_matrix(-temp), label=label, ax=ax)

def draw_graph(x, label=None, ax=None):
	temp = x.data.numpy().squeeze()
	nx.draw_circular(nx.from_numpy_matrix(temp), label=label, ax=ax)

def draw_graphs(graph_number, dataset, coder):
	fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(5, 7))

	draw_graph(dataset[graph_number], label="original Graph", ax=ax[0])
	ax[0].set_title('original Graph')

	draw_graph(coder(dataset)[graph_number].round(), label="Restored Graph", ax=ax[1])
	ax[1].set_title('restored Graph')