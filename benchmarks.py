# -*- coding: utf-8 -*-
"""
@author: Milan

This code contains the implementation of benchmarks on a dummy data set

"""

#%% Packages

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import math

#%% Dummy data

# Setting - time series:  
#     Outcome:             Y 
#     Treatment:           T 
#     Observed covariates: X_vec = [x1, ..., xp]

# Simulation setup:
#     Y = alpha*T + beta_vec*X_vec

# Simulate data
n = 100; p = 10; 
X = np.random.normal(loc=0, scale=1, size=(n,p))
beta_vec = np.random.normal(loc=1, scale=1, size=p)
T = np.random.choice(a=2, size=n)
alpha = np.random.normal(loc=1, scale=1, size=1)
Y = alpha*T + np.dot(X, beta_vec)
data = np.concatenate((Y.reshape(n,1),T.reshape(n,1),X), axis=1)
d_train, d_test = train_test_split(data, test_size=0.1)
w = np.random.random(size=n)

#%% Benchmarks implementation (4 models)

# With respect to data structure, each benchmark has two implementations:
#     1) without weights w (deletion, imputation) and 2) with weights w (reweighting)

############################## Benchmarks train ##############################

# Input: train data of shape [Y, T, X]
# Output: trained model 

# 1) Linear regression (OLS)

# 2) Causal forest (CF)

# 3) Treatment agnostic representation network (TARNet)

# 4) Counterfactual regression Wasserstein (CFR-WASS)


############################## Benchmarks test ##############################

# Input: test data, trained model
# Output: ITE prediction on test data

# 1) Linear regression (OLS)

# 2) Causal forest (CF)

# 3) Treatment agnostic representation network (TARNet)

# 4) Counterfactual regression Wasserstein (CFR-WASS)
