SPD_reduce
============

This repo is dedicated to [Deep Manifold Learning of Symmetric Positive Definite (SPD) Matrices with Application to Face Recognition](https://www.google.ru/) article from AAAI 2017. 

The main idea is too apply article ideas about SPD matrixes manifold to real world problems via finding low-level representation of SPD matrixes. In this project next thee problems we're considered:
* Encoding random SPD matrixes using basic blocks from article
* Speed up determinant calculation for SPD matrixes via this encoding 
* Use basic blocks to solve problem of finding embeddings for graph ([MUTAG](https://figshare.com/articles/MUTAG_and_ENZYMES_DataSet/899875) dataset is used for this task)

Encoding SPD matrixes
-----------------------

For that task two types of basic block are used:  
* 2D Fully Connected Layer (SPD.Linear2D class)
* Symmetrically Clean Layer (SPD.SymmetricallyCleanLayer class)
