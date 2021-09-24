# -*- coding: utf-8 -*-
"""
@author: Milan
"""

#%% Packages

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#%% Data

# Load data
d = pd.read_csv('C:/Users/Milan/Dropbox/MK/ETH PhD Applied Probabilistic AI/projects_long/MT/code/data/Twin_Data.csv.gz', compression='gzip')
data = d.values
    
# Extract from raw data 
X = data[:,:30]
Y_po = data[:,30:]
n = X.shape[0]
p = X.shape[1]
Y_po = np.array(Y_po < 9999, dtype=float) # Die in one year (Y=1), otherwise (Y=0) 
coef = np.random.uniform(-0.01, 0.01, size=(p,1))
prob_temp = expit(np.matmul(X,coef).reshape(n) + np.random.normal(0,0.01, size=n))
prob_t = prob_temp/(2*np.mean(prob_temp))
prob_t[prob_t>1] = 1
T = np.random.binomial(n=1,p=prob_t,size=n)
Y = np.empty(shape=n)
Y = T * Y_po[:,1] + (1-T) * Y_po[:,0]    
ite = Y_po[:,1] - Y_po[:,0]  
data = np.concatenate((ite.reshape(n,1), Y.reshape(n,1),
                       T.reshape(n,1), X, np.empty(shape=(n,1))), axis=1)

# Covariate-dependent masking of treatment (~30% missingness in T)
p_m, p_o = [np.ones(n), np.ones(n)]
mean_x = np.mean(data[:,3:(p+3)], axis=0)
for j in range(n):
    for k in range(3,p+3): 
        if data[j,k] < mean_x[k-3]:
            p_m[j] *= 0.9
            p_o[j] *= 0.1
        else:
            p_m[j] *= 0.1
            p_o[j] *= 0.9
    p_sum = p_m[j] + p_o[j]
    p_m[j] = p_m[j]/p_sum
    p_o[j] = p_o[j]/p_sum
    data[j,-1] = np.random.choice(a=[0,1], p=[p_m[j], p_o[j]])
        
# Save results for n_sim runs
twins_results = {
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

n_sim = 10
# for run in range(n_sim):
run = 0    

# Train-Test split
d_train, d_test = train_test_split(data, test_size=0.1, shuffle=True)
d_train2, d_val = train_test_split(d_train, test_size=0.2, shuffle=True)

# Deletion method data (delete cases with R=0)
d_train_del = d_train[(d_train[:,-1] == 1),:]

# Imputation method data (impute according to p(T|X))
modelT = LogisticRegression(penalty='none', solver='lbfgs', max_iter=5000)
modelT.fit(d_train_del[:,3:-1], d_train_del[:,2])
p_t = modelT.predict_proba(d_train[:, 3:-1])[:,1]
d_train_imp = np.empty(shape=(d_train.shape[0],p+4))
d_train_imp[:,:] = d_train[:,:]
for i in range(d_train_imp.shape[0]):
    if d_train_imp[i,-1] == 0:
        d_train_imp[i,2] = np.random.choice(a=[0,1], p=[1-p_t[i], p_t[i]])

# Reweighting method data (reweight using p(R|X))
modelR = LogisticRegression(penalty='none', solver='lbfgs', max_iter=5000)
modelR.fit(d_train[:,3:-1], d_train[:,-1])
w = np.empty(shape=d_train_del.shape[0])
w = 1/modelR.predict_proba(d_train_del[:, 3:-1])[:,1]

# Splitting test data into observed T and missing T
d_test_ot = d_test[(d_test[:,-1] == 1),:]
d_test_mt = d_test[(d_test[:,-1] == 0),:]

# Performance evaluation metric: observed PEHE

# Implementation of method with ablation - 4 models (d_train2, d_val)

# Performance evaluation for method (d_test_ot, d_test_mt, d_test)

# Implementation of benchmarks - 3 * 4 models (d_train_del, d_train_imp, d_train_del + w)

# Performance evaluation for benchmarks (d_test_ot, d_test_mt, d_test)