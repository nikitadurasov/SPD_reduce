import torch
from torch import nn
from torch import FloatTensor
from torch.autograd import Variable
from torch import optim
from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F

from tqdm import tqdm
from numpy.linalg import matrix_rank
from sklearn.datasets import make_spd_matrix
import os

import dataset_tools as dt
from random import shuffle
import numpy as np

# TO DO
# test function on larger datasets
# try to restore graph from its laplacian matrix
# write comments

class Linear2D(nn.Module):

    def __init__(self, in_features, out_features):
        super(Linear2D, self).__init__()
        self.W = nn.Parameter(torch.rand((out_features, in_features)).normal_())

    def forward(self, X):
        XW = self._matmul(X, self.W.transpose(0 ,1))
        WXW = self._matmul(XW, self.W, kind='left')
        return WXW

    def _matmul(self, X, Y, kind='right'):
        results = []
        for i in range(X.size(0)):
            if kind == 'right':
                result = torch.mm(X[i], Y)
            elif kind == 'left':
                result = torch.mm(Y, X[i])
            results.append(result.unsqueeze(0))
        return torch.cat(results, 0)

    def check_rank(self):
        return min(self.W.data.size()) == matrix_rank(self.W.data.numpy())


class SymmetricallyCleanLayer(nn.Module):
    def __init__(self):
        super(SymmetricallyCleanLayer, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.relu(X)


class NonLinearityBlock(nn.Module):
    def __init__(self, activation):
        super(NonLinearityBlock, self).__init__()
        self.activation = activation

    def forward(self, X):
        return self.activation(X)


class ShrinkBlock(nn.Module):
    def __init__(self, features):
        super(ShrinkBlock, self).__init__()
        self.weights_matrix = nn.Parameter(torch.zeros((features, features)).normal_())
        self.weights_matrix_square = torch.mm(self.weights_matrix.transpose(0, 1), self.weights_matrix)
        self.inverse_eye = Variable(torch.ones((features, features)) - torch.eye(features))

    def forward(self, X):
        return X - self.weights_matrix_square * self.inverse_eye


class MatrixEncoder(nn.Module):
    def __init__(self, n_features):
        super(MatrixEncoder, self).__init__()
        self.n_features = n_features
        
        self.encoder = nn.Sequential(
            Linear2D(n_features, 5),
            SymmetricallyCleanLayer()
        )
        
        self.decoder = nn.Sequential(
        	Linear2D(5, n_features)
        )
        
    def forward(self, X):
        return self.decoder(self.encoder(X))


def run_test(n_iterations=100):

	coder = MatrixEncoder(6)
	dataset = [FloatTensor(make_spd_matrix(6)).unsqueeze(0) for _ in range(100)]

	indexes = list(range(len(dataset)))
	shuffle(indexes)

	dataset = np.array(dataset)
	dataset = dataset[indexes]
	dataset = Variable(torch.cat(dataset, 0))

	test_size = 30

	train_dataset = dataset[:-test_size]
	test_dataset = dataset[-test_size:]

	optimizer = optim.Adam(coder.parameters(), lr=0.1)
	criterion = MSELoss()

	for epoch in tqdm(range(1, n_iterations)):
		optimizer.zero_grad()

		outputs_train = coder(train_dataset)
		outputs_test = coder(test_dataset)

		loss_train = criterion(outputs_train, train_dataset)
		loss_test = criterion(outputs_test, test_dataset)

		loss_train.backward(retain_graph=True)
		if epoch % 10 == 0:
			print("EPOCH: {0}, TRAIN LOSS: {1}, TEST LOSS".format(epoch, loss_train.data[0]), loss_test.data[0])

		if epoch == 200:
			optimizer.state_dict()['param_groups'][0]['lr'] == optimizer.state_dict()['param_groups'][0]['lr'] * 0.1
		optimizer.step()

	return coder, dataset

def run_test_mutag(n_iterations=2000):
	coder = MatrixEncoder(6)
	dataset = dt.build_dataset(dt.read_test_graphs())
	dataset = [dt.adjacency_tensor(x).unsqueeze(0) for x in dataset]

	indexes = list(range(len(dataset)))
	shuffle(indexes)

	dataset = np.array(dataset)
	dataset = dataset[indexes]
	dataset = Variable(torch.cat(dataset, 0))

	test_size = 30

	train_dataset = dataset[:-test_size]
	test_dataset = dataset[-test_size:]

	optimizer = optim.Adam(coder.parameters(), lr=0.1)
	criterion = MSELoss()

	for epoch in tqdm(range(1, n_iterations)):
		optimizer.zero_grad()

		outputs_train = coder(train_dataset)
		outputs_test = coder(test_dataset)

		loss_train = criterion(outputs_train, train_dataset)
		loss_test = criterion(outputs_test, test_dataset)

		loss_train.backward(retain_graph=True)
		if epoch % 10 == 0:
			print("EPOCH: {0}, TRAIN LOSS: {1}, TEST LOSS".format(epoch, loss_train.data[0]), loss_test.data[0])
		optimizer.step()


	return coder, dataset

class DetNet(nn.Module):
    def __init__(self, n_features):
        super(DetNet, self).__init__()
        self.n_features = n_features
        self.fc1 = nn.Linear(n_features**2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, X):
        x = x.view(-1, n_features**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def run_detetmenant(coder):
	dataset = dt.build_dataset(dt.read_test_graphs())
	dataset = [dt.adjacency_tensor(x).unsqueeze(0) for x in dataset]

	indexes = list(range(len(dataset)))
	shuffle(indexes)

	dataset = np.array(dataset)
	dataset = dataset[indexes]
	dataset = Variable(torch.cat(dataset, 0))

	test_size = 30

	train_dataset = dataset[:-test_size]
	test_dataset = dataset[-test_size:]

	optimizer = optim.Adam(coder.parameters(), lr=0.1)
	criterion = MSELoss()

	for epoch in tqdm(range(1, n_iterations)):
		optimizer.zero_grad()

		outputs_train = coder(train_dataset)
		outputs_test = coder(test_dataset)

		loss_train = criterion(outputs_train, train_dataset)
		loss_test = criterion(outputs_test, test_dataset)

		loss_train.backward(retain_graph=True)
		if epoch % 10 == 0:
			print("EPOCH: {0}, TRAIN LOSS: {1}, TEST LOSS".format(epoch, loss_train.data[0]), loss_test.data[0])
		optimizer.step()


	return coder, dataset

if __name__ == "__main__":
    run_test()
