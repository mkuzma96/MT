# -*- coding: utf-8 -*-
"""

The code contains experiments with IHDP dataset

"""

#%% Packages

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from main import *
from benchmarks import *
from cv_hyper_search import * 

#%% Data

# Description:  
#     Outcome: Y - continuous 
#     Treatment: T - binary
#     Observed covariates: X_vec = [x1, ..., xp] - vector of covariates (p=25)

# This semi-sythetic data comprises of 100 simulated datasets with 747 observations in each: 
# 25 covariates, treatment, outcome
# -> We know the true ITE used to simulate the dataset and hence use PEHE as evaluation metric.

# Load data
data_train = np.load('data/ihdp_npci_1-100.train.npz')
data_test = np.load('data/ihdp_npci_1-100.test.npz')
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
 
# Hyperparameters
hyperparams_list = {
    'layer_size_representation': [50, 100, 200],
    'layer_size_hypothesis': [50, 100, 200],
    'learn_rate': [0.01, 0.005, 0.001],
    'dropout_rate':  [0, 0.1, 0.2, 0.3],
    'num_iterations': [1000, 2000, 3000],
    'batch_size': [50, 100, 200],
    'alphas': [10**(k/2) for k in np.linspace(-10,6,17)],
    'betas': [10**(k/2) for k in np.linspace(-10,6,17)],
    'gammas': [10**(k/2) for k in np.linspace(-10,6,17)],
    'lambdas': [10**(k/2) for k in np.linspace(-10,6,17)]
    }

# Save results for n_sim runs
ihdp_results = {
        'MTHD_adv':[[],[],[],[]],
        'MTHD_IPM':[[],[],[],[]],
        'Linear_del':[[],[],[]],
        'Linear_imp':[[],[],[]],
        'Linear_rew':[[],[],[]],
        'CF_del':[[],[],[]],
        'CF_imp':[[],[],[]],
        'CF_rew':[[],[],[]],
        'TARNet_del':[[],[],[],[]],
        'TARNet_imp':[[],[],[],[]],
        'TARNet_rew':[[],[],[],[]],
        'CFRWASS_del':[[],[],[],[]],
        'CFRWASS_imp':[[],[],[],[]],
        'CFRWASS_rew':[[],[],[],[]]
        }

#%% Experiments

n_sims = np.random.choice(n_sim, size=10, replace=False)
for run in range(10):
    
    print(run)
    run = n_sims[run]

    # Train-Test split -> shape [ITE, Y, T, X, R]
    d_train_val, d_test = train_test_split(data[run,:,:], test_size=0.1, shuffle=True)
    d_train, d_val = train_test_split(d_train_val, test_size=0.2, shuffle=True)
    
    # Validation data -> shape [Y, T, X]
    d_val = d_val[d_val[:,-1] == 1, 1:-1]

    # Train data -> shape [Y, T, X, R]
    d_train = d_train[:,1:]
    
    # Deletion method data (delete cases with R=0) -> shape [Y, T, X]
    d_train_del = d_train[(d_train[:,-1] == 1),:-1]
    
    # Imputation method data (impute according to p(T|X)) -> shape [Y, T, X]
    modelT = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000)
    modelT.fit(d_train_del[:,2:], d_train_del[:,1])
    p_t = modelT.predict_proba(d_train[:, 2:-1])[:,1]
    d_train_imp = np.empty(shape=(d_train.shape[0],p+2))
    d_train_imp[:,:] = d_train[:,:-1]
    for i in range(d_train_imp.shape[0]):
        if d_train[i,-1] == 0:
            d_train_imp[i,1] = np.random.choice(a=[0,1], p=[1-p_t[i], p_t[i]])
    
    # Reweighting method data (reweight using p(R|X)) -> shape [Y, T, X, w_miss]
    modelR = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000)
    modelR.fit(d_train[:,2:-1], d_train[:,-1])
    w_miss = np.empty(shape=d_train_del.shape[0])
    w_miss = 1/modelR.predict_proba(d_train_del[:, 2:])[:,1]
    d_train_rew = np.concatenate((d_train_del, w_miss.reshape(d_train_del.shape[0],1)), axis=1)
    
    # Splitting test data into observed T and missing T
    d_test_ot = d_test[(d_test[:,-1] == 1),:]
    d_test_mt = d_test[(d_test[:,-1] == 0),:]
        
    # Performance evaluation metric: PEHE
    def PEHE(ITE_true, ITE_est):
        if type(ITE_est) == torch.Tensor:
            ITE_est = ITE_est.cpu().detach().numpy()
        PEHE = np.square(np.subtract(ITE_true, ITE_est)).mean()
        return np.sqrt(PEHE)
    
    ITE_true = d_test[:,0:1]
    ITE_true_ot = d_test_ot[:,0:1]
    ITE_true_mt = d_test_mt[:,0:1]
    
    # Implementation of method with ablation - 4 models (d_train2, d_val)
    
    hyperpar_opt_adv = CV_hyperparam_search(data_train=d_train, data_val=d_val, y_cont=True, 
                                            metric='PEHE', model=MTRNet, model_pred = MTRNet_pred,
                                            hyperparams_list=hyperparams_list, n_search=100)
    method_adv = MTRNet(data_train=d_train, y_cont=True, hyperparams=hyperpar_opt_adv)
    
    hyperpar_opt_IPM = CV_hyperparam_search(data_train=d_train, data_val=d_val, y_cont=True, 
                                            metric='PEHE', model=MTRNetIPM, model_pred = MTRNetIPM_pred,
                                            hyperparams_list=hyperparams_list, n_search=100)
    method_IPM = MTRNetIPM(data_train=d_train, y_cont=True, hyperparams=hyperpar_opt_IPM)
    
    # Performance evaluation for method (d_test_ot, d_test_mt, d_test)
    
    ITE_est_adv = MTRNet_pred(d_test[:,3:-1], method_adv, y_cont=True)
    ITE_est_adv_ot = MTRNet_pred(d_test_ot[:,3:-1], method_adv, y_cont=True)
    ITE_est_adv_mt = MTRNet_pred(d_test_mt[:,3:-1], method_adv, y_cont=True)
    
    PEHE_hat_adv = PEHE(ITE_true, ITE_est_adv)
    PEHE_hat_adv_ot = PEHE(ITE_true_ot, ITE_est_adv_ot)
    PEHE_hat_adv_mt = PEHE(ITE_true_mt, ITE_est_adv_mt)
    
    ihdp_results['MTHD_adv'][0].append(PEHE_hat_adv)
    ihdp_results['MTHD_adv'][1].append(PEHE_hat_adv_ot)
    ihdp_results['MTHD_adv'][2].append(PEHE_hat_adv_mt)
    ihdp_results['MTHD_adv'][3].append(hyperpar_opt_adv)    
    
    ITE_est_IPM = MTRNetIPM_pred(d_test[:,3:-1], method_IPM, y_cont=True)
    ITE_est_IPM_ot = MTRNetIPM_pred(d_test_ot[:,3:-1], method_IPM, y_cont=True)
    ITE_est_IPM_mt = MTRNetIPM_pred(d_test_mt[:,3:-1], method_IPM, y_cont=True)
    
    PEHE_hat_IPM = PEHE(ITE_true, ITE_est_IPM)
    PEHE_hat_IPM_ot = PEHE(ITE_true_ot, ITE_est_IPM_ot)
    PEHE_hat_IPM_mt = PEHE(ITE_true_mt, ITE_est_IPM_mt)
    
    ihdp_results['MTHD_IPM'][0].append(PEHE_hat_IPM)
    ihdp_results['MTHD_IPM'][1].append(PEHE_hat_IPM_ot)
    ihdp_results['MTHD_IPM'][2].append(PEHE_hat_IPM_mt)
    ihdp_results['MTHD_IPM'][3].append(hyperpar_opt_IPM) 
    
    # Implementation of benchmarks - 3 * 4 models (d_train_del, d_train_imp, d_train_del + w)
    
    Linear_del = Linear(d_train_del, y_cont=True)
    Linear_imp = Linear(d_train_imp, y_cont=True)
    Linear_rew = Linear_w(d_train_rew, y_cont=True)
    
    CF_del = CF(d_train_del, y_cont=True)
    CF_imp = CF(d_train_imp, y_cont=True)
    CF_rew = CF_w(d_train_rew, y_cont=True)
    
    hyperpar_opt_del1 = CV_hyperparam_search(data_train=d_train_del, data_val=d_val, y_cont=True, 
                                             metric='PEHE', model=TARNet, model_pred = TARNet_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)    
    TARNet_del = TARNet(data_train=d_train_del, y_cont=True, hyperparams=hyperpar_opt_del1)

    hyperpar_opt_imp1 = CV_hyperparam_search(data_train=d_train_imp, data_val=d_val, y_cont=True, 
                                             metric='PEHE', model=TARNet, model_pred = TARNet_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)    
    TARNet_imp = TARNet(data_train=d_train_imp, y_cont=True, hyperparams=hyperpar_opt_imp1)

    hyperpar_opt_rew1 = CV_hyperparam_search(data_train=d_train_rew, data_val=d_val, y_cont=True, 
                                             metric='PEHE', model=TARNet_w, model_pred = TARNet_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)
    TARNet_rew = TARNet_w(data_train=d_train_rew, y_cont=True, hyperparams=hyperpar_opt_rew1)
    
    hyperpar_opt_del2 = CV_hyperparam_search(data_train=d_train_del, data_val=d_val, y_cont=True, 
                                             metric='PEHE', model=CFRWASS, model_pred=CFRWASS_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)
    CFRWASS_del = CFRWASS(data_train=d_train_del, y_cont=True, hyperparams=hyperpar_opt_del2)

    hyperpar_opt_imp2 = CV_hyperparam_search(data_train=d_train_imp, data_val=d_val, y_cont=True, 
                                             metric='PEHE', model=CFRWASS, model_pred=CFRWASS_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)
    CFRWASS_imp = CFRWASS(data_train=d_train_imp, y_cont=True, hyperparams=hyperpar_opt_imp2)

    hyperpar_opt_rew2 = CV_hyperparam_search(data_train=d_train_rew, data_val=d_val, y_cont=True, 
                                             metric='PEHE', model=CFRWASS_w, model_pred=CFRWASS_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)
    CFRWASS_rew = CFRWASS_w(data_train=d_train_rew, y_cont=True, hyperparams=hyperpar_opt_rew2)
    
    # Performance evaluation for benchmarks (d_test_ot, d_test_mt, d_test)
    
    ITE_est_Linear_del = Linear_pred(d_test[:,3:-1], Linear_del, y_cont=True)
    ITE_est_Linear_del_ot = Linear_pred(d_test_ot[:,3:-1], Linear_del, y_cont=True)
    ITE_est_Linear_del_mt = Linear_pred(d_test_mt[:,3:-1], Linear_del, y_cont=True)
    
    PEHE_hat_Linear_del = PEHE(ITE_true, ITE_est_Linear_del)
    PEHE_hat_Linear_del_ot = PEHE(ITE_true_ot, ITE_est_Linear_del_ot)
    PEHE_hat_Linear_del_mt = PEHE(ITE_true_mt, ITE_est_Linear_del_mt)
    
    ihdp_results['Linear_del'][0].append(PEHE_hat_Linear_del)
    ihdp_results['Linear_del'][1].append(PEHE_hat_Linear_del_ot)
    ihdp_results['Linear_del'][2].append(PEHE_hat_Linear_del_mt)
    
    ITE_est_Linear_imp = Linear_pred(d_test[:,3:-1], Linear_imp, y_cont=True)
    ITE_est_Linear_imp_ot = Linear_pred(d_test_ot[:,3:-1], Linear_imp, y_cont=True)
    ITE_est_Linear_imp_mt = Linear_pred(d_test_mt[:,3:-1], Linear_imp, y_cont=True)
    
    PEHE_hat_Linear_imp = PEHE(ITE_true, ITE_est_Linear_imp)
    PEHE_hat_Linear_imp_ot = PEHE(ITE_true_ot, ITE_est_Linear_imp_ot)
    PEHE_hat_Linear_imp_mt = PEHE(ITE_true_mt, ITE_est_Linear_imp_mt)
    
    ihdp_results['Linear_imp'][0].append(PEHE_hat_Linear_imp)
    ihdp_results['Linear_imp'][1].append(PEHE_hat_Linear_imp_ot)
    ihdp_results['Linear_imp'][2].append(PEHE_hat_Linear_imp_mt)
    
    ITE_est_Linear_rew = Linear_pred(d_test[:,3:-1], Linear_rew, y_cont=True)
    ITE_est_Linear_rew_ot = Linear_pred(d_test_ot[:,3:-1], Linear_rew, y_cont=True)
    ITE_est_Linear_rew_mt = Linear_pred(d_test_mt[:,3:-1], Linear_rew, y_cont=True)
    
    PEHE_hat_Linear_rew = PEHE(ITE_true, ITE_est_Linear_rew)
    PEHE_hat_Linear_rew_ot = PEHE(ITE_true_ot, ITE_est_Linear_rew_ot)
    PEHE_hat_Linear_rew_mt = PEHE(ITE_true_mt, ITE_est_Linear_rew_mt)
    
    ihdp_results['Linear_rew'][0].append(PEHE_hat_Linear_rew)
    ihdp_results['Linear_rew'][1].append(PEHE_hat_Linear_rew_ot)
    ihdp_results['Linear_rew'][2].append(PEHE_hat_Linear_rew_mt)
    
    ITE_est_CF_del = CF_pred(d_test[:,3:-1], CF_del, y_cont=True)
    ITE_est_CF_del_ot = CF_pred(d_test_ot[:,3:-1], CF_del, y_cont=True)
    ITE_est_CF_del_mt = CF_pred(d_test_mt[:,3:-1], CF_del, y_cont=True)
    
    PEHE_hat_CF_del = PEHE(ITE_true, ITE_est_CF_del)
    PEHE_hat_CF_del_ot = PEHE(ITE_true_ot, ITE_est_CF_del_ot)
    PEHE_hat_CF_del_mt = PEHE(ITE_true_mt, ITE_est_CF_del_mt)
    
    ihdp_results['CF_del'][0].append(PEHE_hat_CF_del)
    ihdp_results['CF_del'][1].append(PEHE_hat_CF_del_ot)
    ihdp_results['CF_del'][2].append(PEHE_hat_CF_del_mt)
    
    ITE_est_CF_imp = CF_pred(d_test[:,3:-1], CF_imp, y_cont=True)
    ITE_est_CF_imp_ot = CF_pred(d_test_ot[:,3:-1], CF_imp, y_cont=True)
    ITE_est_CF_imp_mt = CF_pred(d_test_mt[:,3:-1], CF_imp, y_cont=True)
    
    PEHE_hat_CF_imp = PEHE(ITE_true, ITE_est_CF_imp)
    PEHE_hat_CF_imp_ot = PEHE(ITE_true_ot, ITE_est_CF_imp_ot)
    PEHE_hat_CF_imp_mt = PEHE(ITE_true_mt, ITE_est_CF_imp_mt)
    
    ihdp_results['CF_imp'][0].append(PEHE_hat_CF_imp)
    ihdp_results['CF_imp'][1].append(PEHE_hat_CF_imp_ot)
    ihdp_results['CF_imp'][2].append(PEHE_hat_CF_imp_mt)
    
    ITE_est_CF_rew = CF_pred(d_test[:,3:-1], CF_rew, y_cont=True)
    ITE_est_CF_rew_ot = CF_pred(d_test_ot[:,3:-1], CF_rew, y_cont=True)
    ITE_est_CF_rew_mt = CF_pred(d_test_mt[:,3:-1], CF_rew, y_cont=True)
    
    PEHE_hat_CF_rew = PEHE(ITE_true, ITE_est_CF_rew)
    PEHE_hat_CF_rew_ot = PEHE(ITE_true_ot, ITE_est_CF_rew_ot)
    PEHE_hat_CF_rew_mt = PEHE(ITE_true_mt, ITE_est_CF_rew_mt)
    
    ihdp_results['CF_rew'][0].append(PEHE_hat_CF_rew)
    ihdp_results['CF_rew'][1].append(PEHE_hat_CF_rew_ot)
    ihdp_results['CF_rew'][2].append(PEHE_hat_CF_rew_mt)
    
    ITE_est_TARNet_del = TARNet_pred(d_test[:,3:-1], TARNet_del, y_cont=True)
    ITE_est_TARNet_del_ot = TARNet_pred(d_test_ot[:,3:-1], TARNet_del, y_cont=True)
    ITE_est_TARNet_del_mt = TARNet_pred(d_test_mt[:,3:-1], TARNet_del, y_cont=True)
    
    PEHE_hat_TARNet_del = PEHE(ITE_true, ITE_est_TARNet_del)
    PEHE_hat_TARNet_del_ot = PEHE(ITE_true_ot, ITE_est_TARNet_del_ot)
    PEHE_hat_TARNet_del_mt = PEHE(ITE_true_mt, ITE_est_TARNet_del_mt)
    
    ihdp_results['TARNet_del'][0].append(PEHE_hat_TARNet_del)
    ihdp_results['TARNet_del'][1].append(PEHE_hat_TARNet_del_ot)
    ihdp_results['TARNet_del'][2].append(PEHE_hat_TARNet_del_mt)
    ihdp_results['TARNet_del'][3].append(hyperpar_opt_del1)
    
    ITE_est_TARNet_imp = TARNet_pred(d_test[:,3:-1], TARNet_imp, y_cont=True)
    ITE_est_TARNet_imp_ot = TARNet_pred(d_test_ot[:,3:-1], TARNet_imp, y_cont=True)
    ITE_est_TARNet_imp_mt = TARNet_pred(d_test_mt[:,3:-1], TARNet_imp, y_cont=True)
    
    PEHE_hat_TARNet_imp = PEHE(ITE_true, ITE_est_TARNet_imp)
    PEHE_hat_TARNet_imp_ot = PEHE(ITE_true_ot, ITE_est_TARNet_imp_ot)
    PEHE_hat_TARNet_imp_mt = PEHE(ITE_true_mt, ITE_est_TARNet_imp_mt)
    
    ihdp_results['TARNet_imp'][0].append(PEHE_hat_TARNet_imp)
    ihdp_results['TARNet_imp'][1].append(PEHE_hat_TARNet_imp_ot)
    ihdp_results['TARNet_imp'][2].append(PEHE_hat_TARNet_imp_mt)
    ihdp_results['TARNet_imp'][3].append(hyperpar_opt_imp1)
    
    ITE_est_TARNet_rew = TARNet_pred(d_test[:,3:-1], TARNet_rew, y_cont=True)
    ITE_est_TARNet_rew_ot = TARNet_pred(d_test_ot[:,3:-1], TARNet_rew, y_cont=True)
    ITE_est_TARNet_rew_mt = TARNet_pred(d_test_mt[:,3:-1], TARNet_rew, y_cont=True)
    
    PEHE_hat_TARNet_rew = PEHE(ITE_true, ITE_est_TARNet_rew)
    PEHE_hat_TARNet_rew_ot = PEHE(ITE_true_ot, ITE_est_TARNet_rew_ot)
    PEHE_hat_TARNet_rew_mt = PEHE(ITE_true_mt, ITE_est_TARNet_rew_mt)
    
    ihdp_results['TARNet_rew'][0].append(PEHE_hat_TARNet_rew)
    ihdp_results['TARNet_rew'][1].append(PEHE_hat_TARNet_rew_ot)
    ihdp_results['TARNet_rew'][2].append(PEHE_hat_TARNet_rew_mt)
    ihdp_results['TARNet_rew'][3].append(hyperpar_opt_rew1)
    
    ITE_est_CFRWASS_del = CFRWASS_pred(d_test[:,3:-1], CFRWASS_del, y_cont=True)
    ITE_est_CFRWASS_del_ot = CFRWASS_pred(d_test_ot[:,3:-1], CFRWASS_del, y_cont=True)
    ITE_est_CFRWASS_del_mt = CFRWASS_pred(d_test_mt[:,3:-1], CFRWASS_del, y_cont=True)
    
    PEHE_hat_CFRWASS_del = PEHE(ITE_true, ITE_est_CFRWASS_del)
    PEHE_hat_CFRWASS_del_ot = PEHE(ITE_true_ot, ITE_est_CFRWASS_del_ot)
    PEHE_hat_CFRWASS_del_mt = PEHE(ITE_true_mt, ITE_est_CFRWASS_del_mt)
    
    ihdp_results['CFRWASS_del'][0].append(PEHE_hat_CFRWASS_del)
    ihdp_results['CFRWASS_del'][1].append(PEHE_hat_CFRWASS_del_ot)
    ihdp_results['CFRWASS_del'][2].append(PEHE_hat_CFRWASS_del_mt)
    ihdp_results['CFRWASS_del'][3].append(hyperpar_opt_del2)
    
    ITE_est_CFRWASS_imp = CFRWASS_pred(d_test[:,3:-1], CFRWASS_imp, y_cont=True)
    ITE_est_CFRWASS_imp_ot = CFRWASS_pred(d_test_ot[:,3:-1], CFRWASS_imp, y_cont=True)
    ITE_est_CFRWASS_imp_mt = CFRWASS_pred(d_test_mt[:,3:-1], CFRWASS_imp, y_cont=True)
    
    PEHE_hat_CFRWASS_imp = PEHE(ITE_true, ITE_est_CFRWASS_imp)
    PEHE_hat_CFRWASS_imp_ot = PEHE(ITE_true_ot, ITE_est_CFRWASS_imp_ot)
    PEHE_hat_CFRWASS_imp_mt = PEHE(ITE_true_mt, ITE_est_CFRWASS_imp_mt)
    
    ihdp_results['CFRWASS_imp'][0].append(PEHE_hat_CFRWASS_imp)
    ihdp_results['CFRWASS_imp'][1].append(PEHE_hat_CFRWASS_imp_ot)
    ihdp_results['CFRWASS_imp'][2].append(PEHE_hat_CFRWASS_imp_mt)
    ihdp_results['CFRWASS_imp'][3].append(hyperpar_opt_imp2)
    
    ITE_est_CFRWASS_rew = CFRWASS_pred(d_test[:,3:-1], CFRWASS_rew, y_cont=True)
    ITE_est_CFRWASS_rew_ot = CFRWASS_pred(d_test_ot[:,3:-1], CFRWASS_rew, y_cont=True)
    ITE_est_CFRWASS_rew_mt = CFRWASS_pred(d_test_mt[:,3:-1], CFRWASS_rew, y_cont=True)
    
    PEHE_hat_CFRWASS_rew = PEHE(ITE_true, ITE_est_CFRWASS_rew)
    PEHE_hat_CFRWASS_rew_ot = PEHE(ITE_true_ot, ITE_est_CFRWASS_rew_ot)
    PEHE_hat_CFRWASS_rew_mt = PEHE(ITE_true_mt, ITE_est_CFRWASS_rew_mt)
    
    ihdp_results['CFRWASS_rew'][0].append(PEHE_hat_CFRWASS_rew)
    ihdp_results['CFRWASS_rew'][1].append(PEHE_hat_CFRWASS_rew_ot)
    ihdp_results['CFRWASS_rew'][2].append(PEHE_hat_CFRWASS_rew_mt)
    ihdp_results['CFRWASS_rew'][3].append(hyperpar_opt_rew2)

