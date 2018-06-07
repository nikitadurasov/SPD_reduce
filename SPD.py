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

import matplotlib.pyplot as plt

class Linear2D(nn.Module):
    """
    Linear block for matrixes. Forward is equivalent to matrix multiplication W^T * X * W.

    Block works with squared SPD matrixes: after forward applied to 
    (in_features) x (in_features) matrix X, returns new SPD matrix with 
    size (out_features, out_features). Can be applied to none SPD matrixes too.

    Parameters:
    -----------
    in_features: int
        size of input square matrixes

    out_features: int
        size of output square matrixes

    Returns
    -------
    torch.autograd.Variable object 
        Output matrix - result of multiplication of input matrix 
        and mtrix of parameters: W^T * X * W

    """

    def __init__(self, in_features, out_features):
        super(Linear2D, self).__init__()
        # initializing weights with standart normal distribution
        self.W = nn.Parameter(torch.rand((out_features, in_features)).normal_()) 

    def forward(self, X):
        # apply W^T * X * W multiplication to input
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
        # check if weights matrix is degenerate
        return min(self.W.data.size()) == matrix_rank(self.W.data.numpy())


class SymmetricallyCleanLayer(nn.Module):
    """
    Symmetrically Clean Layer for matrixes. Forward is equivalent to application 
    of ReLU block for every element of matrix.

    After application of this block to SPD matrix it returns SPD matrix of the 
    same size. This block is analogy for common non-linear block for matrixes. 

    Parameters:
    -----------

    Returns
    -------
    torch.autograd.Variable object 
        Output matrix - result of nullification of negative elements 
        in input matrix.

    """
    def __init__(self):
        super(SymmetricallyCleanLayer, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.relu(X)


class NonLinearityBlock(nn.Module):
    """
    NonLinearity Layer for matrixes. This block is generalization 
    of SymmetricallyCleanLayer: NonLinearityBlock works with arbitrary
    activation block, whereas SymmetricallyCleanLayer works with ReLU.

    Parameters:
    -----------
    activation: torch.nn.Module
        type of activation function, which will be applied to
        every element of input matrix.

    Returns
    -------
    torch.autograd.Variable object 
        Output matrix - result of application of activation block 
        to every element of matrix.

    """
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
            SymmetricallyCleanLayer(),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            Linear2D(5, n_features),
            nn.ReLU()
        )
        
    def forward(self, X):
        return self.decoder(self.encoder(X))


class VAE(nn.Module):
    def __init__(self, in_features):
        super(VAE, self).__init__()
        self.in_features = in_features

        self.fc1 = nn.Linear(self.in_features, 2*self.in_features//3)
        self.fc21 = nn.Linear(2*self.in_features//3, self.in_features//4)
        self.fc22 = nn.Linear(2*self.in_features//3, self.in_features//4)
        self.fc3 = nn.Linear(self.in_features//4, 2*self.in_features//3)
        self.fc4 = nn.Linear(2*self.in_features//3, self.in_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_features))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, in_features):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, in_features), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

class Convolution_Autoencoder(nn.Module):
    def __init__(self, in_features):
        super(Convolution_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 2, stride=0, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=0),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=0, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=0)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=0, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=0, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def run_test(in_features, n_iterations=100, size=500):

    coder = MatrixEncoder(in_features)
    dataset = [FloatTensor(10*make_spd_matrix(in_features)).unsqueeze(0) for _ in range(size)]

    dataset = Variable(torch.cat(dataset, 0))

    test_size = 10

    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    optimizer = optim.Adam(coder.parameters(), lr=0.1)
    criterion = MSELoss()

    loss_data = []

    for epoch in tqdm(range(1, n_iterations)):
        optimizer.zero_grad()

        outputs_train = coder(train_dataset)
        outputs_test = coder(test_dataset)

        loss_train = criterion(outputs_train, train_dataset)
        loss_test = criterion(outputs_test, test_dataset)

        loss_train.backward(retain_graph=True)
        if epoch % 10 == 0:
            print("EPOCH: {0}, TRAIN LOSS: {1}, TEST LOSS".format(epoch, loss_train.data[0]), float(loss_test.data[0]))

        loss_data.append((loss_train.data[0], float(loss_test.data[0])))

        optimizer.step()

    loss_data = np.array(loss_data)
    plt.plot(range(len(loss_data[20:])), loss_data[20:, 0], label='train')
    plt.plot(range(len(loss_data[20:])), loss_data[20:, 1], label='test')
    plt.legend()
    plt.show()

    return coder, dataset

def run_test_vae(in_features, n_iterations=1000, size=1000, printing_step=10):

    coder = VAE(in_features**2)
    dataset = [FloatTensor(10*make_spd_matrix(in_features)).unsqueeze(0) for _ in range(size)]

    dataset = Variable(torch.cat(dataset, 0).reshape(size, -1))

    test_size = 3

    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    optimizer = optim.Adam(coder.parameters(), lr=1e-4)

    for epoch in tqdm(range(1, n_iterations)):
        optimizer.zero_grad()

        outputs_train, mu_train, logvar_train = coder(train_dataset)
        outputs_test, mu_test, logvar_test = coder(test_dataset)

        loss_train = loss_function(outputs_train, train_dataset, mu_train, logvar_train, in_features**2)
        loss_test = loss_function(outputs_test, test_dataset, mu_test, logvar_test, in_features**2)

        loss_train.backward(retain_graph=True)
        if epoch % printing_step == 0:
            print("EPOCH: {0}, TRAIN LOSS: {1}, TEST LOSS".format(epoch, loss_train.data[0]), float(loss_test.data[0]))

        if epoch == 1000:
            optimizer.state_dict()['param_groups'][0]['lr'] == optimizer.state_dict()['param_groups'][0]['lr'] * 0.1
        optimizer.step()

    return coder, dataset

def run_test_conv(in_features, n_iterations=1000, size=1000, printing_step=10):

    coder = Convolution_Autoencoder(in_features**2)
    dataset = [FloatTensor(make_spd_matrix(in_features)).unsqueeze(0) for _ in range(size)]

    dataset = Variable(torch.cat(dataset, 0).reshape(size, 1, in_features, in_features))

    test_size = 30

    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    optimizer = optim.Adam(coder.parameters(), lr=0.1)
    loss_function = nn.MSELoss()

    for epoch in tqdm(range(1, n_iterations)):
        optimizer.zero_grad()

        outputs_train = coder(train_dataset)
        outputs_test = coder(test_dataset)

        loss_train = loss_function(outputs_train, train_dataset)
        loss_test = loss_function(outputs_test, test_dataset)

        loss_train.backward(retain_graph=True)
        if epoch % printing_step == 0:
            print("EPOCH: {0}, TRAIN LOSS: {1}, TEST LOSS".format(epoch, loss_train.data[0]), float(loss_test.data[0]))

        if epoch == 1000:
            optimizer.state_dict()['param_groups'][0]['lr'] == optimizer.state_dict()['param_groups'][0]['lr'] * 0.1
        optimizer.step()

    return coder, dataset


def run_test_mutag(in_features, n_iterations=2000):
    coder = MatrixEncoder(in_features)
    dataset = dt.build_dataset(dt.read_test_graphs(), nodes_number=in_features)[0]
    dataset = [dt.adjacency_tensor(x).unsqueeze(0) for x in dataset]
    shuffle(dataset)
    dataset = Variable(torch.cat(dataset, 0))

    test_size = 30

    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    optimizer = optim.Adam(coder.parameters(), lr=3e-1)
    criterion = MSELoss()

    loss_data = []

    for epoch in tqdm(range(1, n_iterations)):
        optimizer.zero_grad()

        outputs_train = coder(train_dataset)
        outputs_test = coder(test_dataset)

        loss_train = criterion(outputs_train, train_dataset)
        loss_test = criterion(outputs_test, test_dataset)

        loss_train.backward(retain_graph=True)
        if epoch % 10 == 0:
            print("EPOCH: {0}, TRAIN LOSS: {1}, TEST LOSS".format(epoch, loss_train.data[0]), float(loss_test.data[0]))

        if epoch == 1000:
            optimizer.state_dict()['param_groups'][0]['lr'] == optimizer.state_dict()['param_groups'][0]['lr'] * 0.1

        loss_data.append((loss_train.data[0], float(loss_test.data[0])))

        optimizer.step()

    fig = plt.figure()
    loss_data = np.array(loss_data)
    plt.plot(range(len(loss_data[20:])), loss_data[20:, 0], label='train', c='r')
    plt.plot(range(len(loss_data[20:])), loss_data[20:, 1], label='test', c='b')
    plt.title("MSELoss per epoch for train/test")
    plt.xlabel('epoch')
    plt.ylabel('MSELoss')
    plt.grid()
    plt.legend()
    plt.show()

    fig.savefig('loss.pdf', format='pdf')

    return coder, dataset, loss_data

class DetNet(nn.Module):
    def __init__(self, n_features):
        super(DetNet, self).__init__()
        self.n_features = n_features
        self.fc1 = nn.Linear(n_features**2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = x.view(-1, n_features**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def run_detetmenant(in_features, n_iterations=100, size=100):
    coder = DetNet(in_features)
    dataset = [FloatTensor(make_spd_matrix(in_features)).unsqueeze(0) for _ in range(size)]

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

def run_test_vae_mutag(in_features, n_iterations=1000, size=1000, printing_step=10):

    coder = VAE(in_features**2)

    dataset = dt.build_dataset(dt.read_test_graphs(), nodes_number=in_features)[0]
    dataset = [dt.adjacency_tensor(x).unsqueeze(0) for x in dataset]
    shuffle(dataset)
    dataset = Variable(torch.cat(dataset, 0))

    test_size = 3

    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    optimizer = optim.Adam(coder.parameters(), lr=1e-3)

    loss_data = []

    for epoch in tqdm(range(1, n_iterations)):
        optimizer.zero_grad()

        outputs_train, mu_train, logvar_train = coder(train_dataset)
        outputs_test, mu_test, logvar_test = coder(test_dataset)

        loss_train = loss_function(outputs_train, train_dataset, mu_train, logvar_train, in_features**2)
        loss_test = loss_function(outputs_test, test_dataset, mu_test, logvar_test, in_features**2)

        loss_train.backward(retain_graph=True)
        if epoch % printing_step == 0:
            print("EPOCH: {0}, TRAIN LOSS: {1}, TEST LOSS".format(epoch, loss_train.data[0]), float(loss_test.data[0]))

        if epoch == 1000:
            optimizer.state_dict()['param_groups'][0]['lr'] == optimizer.state_dict()['param_groups'][0]['lr'] * 0.1

        loss_data.append((loss_train.data[0], float(loss_test.data[0])))

        optimizer.step()

    fig = plt.figure()
    loss_data = np.array(loss_data)
    plt.plot(range(len(loss_data[20:])), loss_data[20:, 0], label='train', c='r')
    plt.plot(range(len(loss_data[20:])), loss_data[20:, 1], label='test', c='b')
    plt.title("MSELoss per epoch for train/test")
    plt.xlabel('epoch')
    plt.ylabel('MSELoss')
    plt.grid()
    plt.legend()
    plt.show()

    return coder, dataset


if __name__ == "__main__":
    run_test()
