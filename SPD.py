import torch 
from torch import nn
from torch import FloatTensor
from torch.autograd import Variable
from torch import optim
from torch.nn.modules.loss import MSELoss

from numpy.linalg import matrix_rank

from sklearn.datasets import make_spd_matrix

# TO DO
# test function on larger datasets 
# try to restore graph from its laplacian matrix 
# write comments

class Linear2D(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(Linear2D, self).__init__()
        self.W = nn.Parameter(torch.rand((out_features, in_features)))
        
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
            ShrinkBlock(5),
            SymmetricallyCleanLayer(),
            Linear2D(5, 3),
            ShrinkBlock(3)
        )
        
        self.decoder = nn.Sequential(
            Linear2D(3, 5),
            ShrinkBlock(5),
            SymmetricallyCleanLayer(),
            Linear2D(5, n_features),
            ShrinkBlock(n_features)
        )
        
    def forward(self, X):
        return self.decoder(self.encoder(X))


def run_test():
	coder = MatrixEncoder(6)
	matrix = Variable(FloatTensor(make_spd_matrix(6))).unsqueeze(0)
	optimizer = optim.Adam(coder.parameters(), lr=0.01)
	criterion = MSELoss()

	for epoch in range(4000):
	    optimizer.zero_grad()
	    outputs = coder(matrix)
	    loss = criterion(outputs, matrix)
	    loss.backward(retain_graph=True)
	    if epoch % 100 == 0:
	        print(loss.data[0])
	    optimizer.step()