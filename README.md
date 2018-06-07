SPD_reduce
============

This repo is dedicated to [Deep Manifold Learning of Symmetric Positive Definite (SPD) Matrices with Application to Face Recognition](https://www.google.ru/) article from AAAI 2017. 

The main idea is too apply article ideas about SPD matrixes manifold to real world problems via finding low-level representation of SPD matrixes. In this project next thee problems we're considered:
* Encoding random SPD matrixes using basic blocks from article
* Speed up determinant calculation for SPD matrixes via this encoding 
* Use basic blocks to solve problem of finding embeddings for graph ([MUTAG](https://figshare.com/articles/MUTAG_and_ENZYMES_DataSet/899875) dataset is used for this task)

Dependencies:
--------------
* pytorch (v. 0.4.0)
* networkx (v. 2.0)
* sklearn (v. 0.19.1)
* tqdm (v. 4.19.7+)

Encoding SPD matrixes
-----------------------

For that task two types of basic block are used:  
* 2D Fully Connected Layer (**SPD.Linear2D class**)
* Symmetrically Clean Layer (**SPD.SymmetricallyCleanLayer class**)

With this block's we constract model for matrix encoding (**SPD.MatrixEncoder class**) with next architecture:

```
MatrixEncoder(
  (encoder): Sequential(
    (0): Linear2D()
    (1): SymmetricallyCleanLayer(
      (relu): ReLU()
    )
    (2): Tanh()
  )
  (decoder): Sequential(
    (0): Linear2D()
    (1): ReLU()
  )
)
```

Using **sklearn.datasets.make_spd_matrix** we conctract dataset of SPD matrixes and train MatrixEncoder with simple MSELoss and Adam optimizer.

### Code examples
run test using model based on basic blocks
```
from SPD import run_test, run_test_vae, run_test_conv
coder, dataset = run_test(20) 
```
based on VAE AutoEncoder
```
coder, dataset = run_test_vae(20) 
```
based on Convolution AutoEncoder
```
coder, dataset = run_test_conv(20)
```

Determinant calculation
------------------------

Using encoder model from previous task we would like to do next: using this encoder we embed matrix A in low-dimensional matrix B and using matrix B we predict determinant of matrix A with another model.

Architecture for predictor is simple feed-forward network with 3 layers: 
```
DetNet(
  (fc1): Linear(in_features=100, out_features=64)
  (fc2): Linear(in_features=64, out_features=32)
  (fc3): Linear(in_features=32, out_features=1)
)
```
It's important to mention, that MatrixEncoder and DetNet are trained separately.

### Code examples
```
from SPD import run_detetmenant
coder, dataset = run_detetmenant(20) 
```

Embeddings for MUTAG dataset
------------------------------

Using MatrixEncoder model we'd like to constract embeddings for graph structures and solve clustering, classification, anomaly detection problems and etc. 

MUTAG dataset is used as benchmark in this task (~200 graphs with two labels).

### Code examples
run test using model based on basic blocks
```
from SPD import run_test_mutag, run_test_vae_mutag
coder, dataset = run_test_mutag(20) 
```
based on VAE AutoEncoder
```
coder, dataset = run_test_vae_mutag(20) 
```


References:
-------------
* [Deep Manifold Learning of Symmetric Positive Definite (SPD) Matrices with Application to Face Recognition](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14825/14321)
* [Visualizing Data using t-SNE](http://www.cs.toronto.edu/~hinton/absps/tsne.pdf)
* [Hessian Eigenmaps: new locally linear embedding techniques for high-dimensional data](https://pdfs.semanticscholar.org/ff6e/bb0ef618592dfe654a12ddca7a0b75c9176b.pdf)
* [Adaptive Manifold Learning](https://papers.nips.cc/paper/2560-adaptive-manifold-learning.pdf)
* [Learning with kernels](https://www.cs.utah.edu/~piyush/teaching/learning-with-kernels.pdf)
* [A Global Geometric Frameworkfor Nonlinear DimensionalityReduction](http://web.mit.edu/cocosci/Papers/sci_reprint.pdf)

