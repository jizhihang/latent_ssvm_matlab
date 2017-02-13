# latent_ssvm_matlab


Matlab mex wrapper for Latent Structural SVM (Jan 2017)
--------------------------------------------
Implemented by Zhile Ren based on

1) the SVM-light code by Thorsten Joachims,

2) latent S-SVM code by Chun-Nam Yu and

3) SVMstruct-matlab by Dr Andrea Vedaldi.


INSTALLATION
------------
1. See Makefile to specify matlab directories.
2. Type 'make' in the directory and make sure that the compilation is successful.
3. Currently the code only supports 64-bit linux systems.
4. Currently the code only support linear kernels.

EXAMPLE
------------
script_test.m contain a simple SVM example to get you started. The example is adapted from svm-struct-matlab: http://www.robots.ox.ac.uk/~vedaldi//svmstruct.html. 

REFERENCES
----------
[1] C.-N. Yu and T. Joachims: Learning Structural SVMs with Latent Variables, ICML 2009

[2] I. Tsochantaridis, T. Hofmann, T. Joachims, and Y. Altun: Support Vector Learning for Interdependent and Structured Output Spaces, ICML 2004

[3] A. Vedaldi: MATLAB wrapper of SVM-Struct. http://www.robots.ox.ac.uk/~vedaldi//svmstruct.html
