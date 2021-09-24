# -*- coding: utf-8 -*-
"""

This code contains the implementation of our method on a dummy data set

"""

#%% Packages

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

#%% Dummy data

# Setting - time series:  
#     Outcome:                 Y 
#     Treatment:               T 
#     Observed covariates:     X_vec = [x1, ..., xp]
#     Treatment observability: R

# Simulation setup:
#     Y = alpha*T + beta_vec*X_vec

# Simulate data
n = 100; p = 10; 
X = np.random.normal(loc=0, scale=1, size=(n,p))
beta_vec = np.random.normal(loc=1, scale=1, size=p)
T = np.random.choice(a=2, size=n)
alpha = np.random.normal(loc=1, scale=1, size=1)
Y = alpha*T + np.dot(X, beta_vec)

# Covariate-dependent masking of treatment (~50% missingness in T)
R = np.sum(X, axis=1) > 0
data = np.concatenate((Y.reshape(n,1),T.reshape(n,1),X,R.reshape(n,1)), axis=1)  

# Train-Validation-Test split (72/18/10)
d_tr, d_test = train_test_split(data, test_size=0.1, shuffle=True)
d_train, d_val= train_test_split(d_tr, test_size=0.2, shuffle=True)

#%% Method implementation

############################## Method train ##################################

# Input: train and validation data of shape [Y, T, X, R]
# Output: trained model 

# Device configuration
device = torch.device('cuda')

# Hyperparameters
n_layers = [1, 2, 3][0]; lay_size = [50, 100, 200][0]; drop = [0.1, 0.2, 0.3][0]
lr = [0.01, 0.005, 0.001][0]; n_epochs = [50, 100, 200][0]; b_size = [32, 64, 128][0]
alpha = [0.5, 1, 2][0]; theta = [0.5, 1, 2][0]; lam = [0.5, 1, 2][0]

# Data pre-processing 
train = torch.from_numpy(d_train.astype(np.float32))
validation = torch.from_numpy(d_val.astype(np.float32))
test = torch.from_numpy(d_test.astype(np.float32))
train_loader = DataLoader(dataset=train, batch_size=b_size, shuffle=True)

# Model





############################## Method test  ##################################

# Input: test data, trained model
# Output: ITE prediction on test data

