# -*- coding: utf-8 -*-
"""

The code contains experiments with IHDP dataset

"""

#%% Packages

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#%% Data

# Load data
data_train = np.load('C:/Users/Milan/Dropbox/MK/ETH PhD Applied Probabilistic AI/projects_long/MT/code/data/ihdp_npci_1-100.train.npz')
data_test = np.load('C:/Users/Milan/Dropbox/MK/ETH PhD Applied Probabilistic AI/projects_long/MT/code/data/ihdp_npci_1-100.test.npz')
# print(d_train.files)

# Extract from raw data and merge
X = np.concatenate((data_train['x'], data_test['x']), axis=0)
T = np.concatenate((data_train['t'], data_test['t']), axis=0)
Y = np.concatenate((data_train['yf'], data_test['yf']), axis=0)
mu0 = np.concatenate((data_train['mu0'], data_test['mu0']), axis=0)
mu1 = np.concatenate((data_train['mu1'], data_test['mu1']), axis=0)
ite = mu1 - mu0
n = X.shape[0]
p = X.shape[1]
n_sim = X.shape[2]
data = np.empty(shape=(n_sim, n, p+4))
for i in range(n_sim):
    data[i,:,:-1] = np.concatenate((ite[:,i].reshape(n,1), Y[:,i].reshape(n,1), 
                                  T[:,i].reshape(n,1), X[:,:,i]), axis=1)

# Covariate-dependent masking of treatment (~40% missingness in T)
for i in range(n_sim):
    p_m, p_o = [np.ones(n), np.ones(n)]
    mean_x = np.mean(data[i,:,3:(p+3)], axis=0)
    for j in range(n):
        for k in range(3,18): 
            if data[i,j,k] > mean_x[k-3]:
                p_m[j] *= 0.1
                p_o[j] *= 0.9
            else:
                p_m[j] *= 0.9
                p_o[j] *= 0.1
        for k in range(18,p+3): 
            if data[i,j,k] < mean_x[k-3]:
                p_m[j] *= 0.1
                p_o[j] *= 0.9
            else:
                p_m[j] *= 0.9
                p_o[j] *= 0.1                
        p_sum = p_m[j] + p_o[j]
        p_m[j] = p_m[j]/p_sum
        p_o[j] = p_o[j]/p_sum
        data[i,j,-1] = np.random.choice(a=[0,1], p=[p_m[j], p_o[j]])
    
# Save results for n_sim runs
ihdp_results = {
        'MTHD':[[],[],[]],
        'MTHD_noinv':[[],[],[]],
        'MTHD_nomt':[[],[],[]],
        'OLS_del':[[],[],[]],
        'OLS_imp':[[],[],[]],
        'OLS_rew':[[],[],[]],
        'CF_del':[[],[],[]],
        'CF_imp':[[],[],[]],
        'CF_rew':[[],[],[]],
        'TARNet_del':[[],[],[]],
        'TARNet_imp':[[],[],[]],
        'TARNet_rew':[[],[],[]],
        'CFR_del':[[],[],[]],
        'CFR_imp':[[],[],[]],
        'CFR_rew':[[],[],[]]
        }

#%% Experiments

# for run in range(n_sim):
run = 0
    
# Train-Test split
d_train, d_test = train_test_split(data[run,:,:], test_size=0.1, shuffle=True)
d_train2, d_val = train_test_split(d_train, test_size=0.2, shuffle=True)

# Deletion method data (delete cases with R=0)
d_train_del = d_train[(d_train[:,-1] == 1),:]

# Imputation method data (impute according to p(T|X))
modelT = LogisticRegression(penalty='none', solver='lbfgs', max_iter=500)
modelT.fit(d_train_del[:,3:-1], d_train_del[:,2])
p_t = modelT.predict_proba(d_train[:, 3:-1])[:,1]
d_train_imp = np.empty(shape=(d_train.shape[0],p+4))
d_train_imp[:,:] = d_train[:,:]
for i in range(d_train_imp.shape[0]):
    if d_train_imp[i,-1] == 0:
        d_train_imp[i,2] = np.random.choice(a=[0,1], p=[1-p_t[i], p_t[i]])

# Reweighting method data (reweight using p(R|X))
modelR = LogisticRegression(penalty='none', solver='lbfgs', max_iter=500)
modelR.fit(d_train[:,3:-1], d_train[:,-1])
w = np.empty(shape=d_train_del.shape[0])
w = 1/modelR.predict_proba(d_train_del[:, 3:-1])[:,1]

# Splitting test data into observed T and missing T
d_test_ot = d_test[(d_test[:,-1] == 1),:]
d_test_mt = d_test[(d_test[:,-1] == 0),:]
    
# Performance evaluation metric: true PEHE

# Implementation of method with ablation - 4 models (d_train2, d_val)

# Performance evaluation for method (d_test_ot, d_test_mt, d_test)

# Implementation of benchmarks - 3 * 4 models (d_train_del, d_train_imp, d_train_del + w)

# Performance evaluation for benchmarks (d_test_ot, d_test_mt, d_test)

