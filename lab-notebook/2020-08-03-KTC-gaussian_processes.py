#!/usr/bin/env python
# coding: utf-8

# ## 2020-08-03: Exploring Gaussian Processes
# 
# *Last Updated*: 2020-08-03
# 
# ### Authors
# * Kevin Chu (kevin@velexi.com)
# 
# ### Overview
# In this Jupyter notebook, we explore Gaussian processes.
# 
# ### User parameters
# 
# #### Gaussian process parameters
# * `domain`: domain for Gaussian process
# * `length_scale`: length scale for Gaussian process
# * `variance_scale`: variance scale for Gaussian process
# 
# #### Sample parameters
# * `num_samples`: number of samples to generate

# In[1]:


# --- Imports

# External packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


# --- User parameters

# Gaussian process parameters
domain = {'min': -5.0, 'max': 5.0}
length_scale = 2
variance_scale = 1

# Sample parameters
num_samples = 5


# In[3]:


# --- Preparations

# Seaborn configuration
sns.set(color_codes=True)


# In[4]:


# --- Generate samples from Gaussian process

# Preparations
dX = 0.01
X = np.arange(domain['min'], domain['max'], dX)

# Set mean function
m = np.zeros(X.shape)

# Compute covariance function kernel function
diff_X = np.subtract.outer(X, X)
K = variance_scale * np.exp(-1/2*np.square(diff_X) / length_scale)

# Generate samples from Gaussian process
f = []
for _ in range(num_samples):
    f.append(np.random.multivariate_normal(m, K))
    
# Plot samples
for i in range(num_samples):
    plt.plot(X, f[i])

