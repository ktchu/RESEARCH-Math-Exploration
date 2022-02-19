#!/usr/bin/env python
# coding: utf-8

# ## 2022-02-15: Distribution of $m$-Blade Volumes in $\mathbb{R}^n$
# 
# (Python Version)
# 
# *Last Updated*: 2022-02-18
# 
# ### Authors
# * Kevin Chu (kevin@velexi.com)
# 
# ### Overview
# In this Jupyter notebook, we explore the distribution of the volume of the blade defined by $m$ vectors independently drawn from a uniform distribution over a unit hypersphere in $\mathbb{R}^n$.
# 
# ### Methodology
# 
# * Draw `num_samples` sets of $m$ vectors from a multivariate normal distribution with unit covariance matrix.
# 
#     * Using a multivariate normal distribution with diagonal covariance matrix (1) ensures that vectors are drawn uniformly over directions while (2) allowing us to treat each vector component as an independent random variable.
#     
# * For each sample $x_1, \ldots, x_m$, we perform the following computations to generate the volume distribution.
# 
#     * Normalize all of the vectors to have unit length.
# 
#     * Compute the $\vert x_1 \wedge \cdots \wedge x_m \vert$ by computing determinant of the $R$ matrix of the QR decomposition of the matrix $[x_1 | \cdots | x_m]$.
# 
# ### User Parameters
# 
# * `num_samples`: number of samples to use for estimating probability distributions
# 
# * `n`: dimension of space
# 
# * `m`: dimension of blades to estimate probability distribution for

# In[1]:


# --- User Parameters

# Number of samples to use to estimate probability distributions
num_samples = 100;

# Dimension of space
n = 10;

# Blade dimension
m = 2;


# In[2]:


# --- Imports

# External packages
import numpy as np


# In[3]:


# --- Generate samples

# Generate sample of vectors drawn from a uniform distribution over an
# n-dimensional unit hypersphere
vector_samples = np.random.randn(n, num_samples * m)
norms = np.linalg.norm(vector_samples, ord=2, axis=0)
vector_samples = np.array([vector_samples[:,i]/norms[i] for i in range(num_samples)])

# Generate sample of blades
blades = [vector_samples[:, i*m:(i+1)*m] for i in range(num_samples)]

