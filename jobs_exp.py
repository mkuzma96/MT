# -*- coding: utf-8 -*-
"""

The code contains experiments with Jobs dataset

"""

#%% Packages

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from main import *
from benchmarks import *
from cv_hyper_search import * 

#%% Data

# Description:  
#     Outcome: Y - binary {0,1} representing employment status (Y=1 employed, Y=0 unemployed)
#     Treatment: T - participation in government training program (T=1 participated, T=0 didn't participate)
#     Observed covariates: X_vec = [x1, ..., xp] - vector of covariates (p=7)

# The data comprises of 3212 observations combining 722 observations from randomized trial (RCT) 
# and 2490 observations from observational study: 7 covariates, treatment, outcome.
# -> True ITE is not accessible in this dataset, so as evaluation metric we use policy risk
#    evaluated on the randomized portion of the data.

# Load data
d = pd.read_csv('data/Jobs_Lalonde_Data.csv.gz', compression='gzip')
data = d.values

# Extract from raw data 
X = data[:,:7]
T = data[:,7]
Raw_Y = data[:,8] 
Y = np.array(Raw_Y > 0, dtype=float) # Employed (Y=1), unemployed (Y=0)
n = X.shape[0]
p = X.shape[1]
data = np.concatenate((Y.reshape(n,1), T.reshape(n,1), 
                       X, np.empty(shape=(n,1))), axis=1)

# Covariate-dependent masking of treatment (~35% missingness in T)
p_m, p_o = [np.ones(n), np.ones(n)]
mean_x = np.mean(data[:,2:(p+2)], axis=0)
for j in range(n):
    for k in range(2,p+2): 
        if data[j,k] > mean_x[k-2]:
            p_m[j] *= 0.9
            p_o[j] *= 0.1
        else:
            p_m[j] *= 0.1
            p_o[j] *= 0.9
    p_sum = p_m[j] + p_o[j]
    p_m[j] = p_m[j]/p_sum
    p_o[j] = p_o[j]/p_sum
    data[j,-1] = np.random.choice(a=[0,1], p=[p_m[j], p_o[j]])

RCT_No = 722
d_rand = data[:RCT_No,:]
d_nonrand = data[RCT_No:,:]

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
jobs_results = {
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

n_sims = 10
for run in range(n_sims):

    print(run)

    # Train-Test split -> shape [Y, T, X, R]
    d_train_val, d_test = train_test_split(d_rand, test_size=0.4, shuffle=True)
    d_train_val = np.concatenate((d_nonrand, d_train_val), axis=0)
    d_train, d_val = train_test_split(d_train_val, test_size=0.2, shuffle=True)
    
    # Validation data -> shape [Y, T, X]
    d_val = d_val[d_val[:,-1] == 1, :-1]
    
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
    
    # Performance evaluation metric: policy risk
    def Rpol(data_test, ITE_est):
        if type(ITE_est) == torch.Tensor:
            ITE_est = ITE_est.cpu().detach().numpy()
        p_f1 = ITE_est > 0
        p_f0 = ITE_est <= 0
        prob_pf1 = np.mean(p_f1) 
        Rpol = 1 - (np.mean(data_test[(p_f1) & (data_test[:,1]==1), 0])*prob_pf1 +\
                    np.mean(data_test[(p_f0) & (data_test[:,1]==0), 0])*(1-prob_pf1))
        if np.sum((p_f1) & (data_test[:,1]==1)) == 0:
            Rpol = 1 - np.mean(data_test[(p_f0) & (data_test[:,1]==0), 0])*(1-prob_pf1)
        if np.sum((p_f0) & (data_test[:,1]==0)) == 0:
            Rpol = 1 - np.mean(data_test[(p_f1) & (data_test[:,1]==1), 0])*prob_pf1    
        return Rpol
    
    # Implementation of method with ablation - 4 models (d_train2, d_val)
    
    hyperpar_opt_adv = CV_hyperparam_search(data_train=d_train, data_val=d_val, y_cont=False, 
                                            metric='Rpol', model=MTRNet, model_pred=MTRNet_pred,
                                            hyperparams_list=hyperparams_list, n_search=100)
    method_adv = MTRNet(data_train=d_train, y_cont=False, hyperparams=hyperpar_opt_adv)
    
    hyperpar_opt_IPM = CV_hyperparam_search(data_train=d_train, data_val=d_val, y_cont=False, 
                                            metric='Rpol', model=MTRNetIPM, model_pred=MTRNetIPM_pred,
                                            hyperparams_list=hyperparams_list, n_search=100)
    method_IPM = MTRNetIPM(data_train=d_train, y_cont=False, hyperparams=hyperpar_opt_IPM)
        
    # Performance evaluation for method (d_test_ot, d_test_mt, d_test)
    
    ITE_est_adv = MTRNet_pred(d_test[:,2:-1], method_adv, y_cont=False)
    ITE_est_adv_ot = MTRNet_pred(d_test_ot[:,2:-1], method_adv, y_cont=False)
    ITE_est_adv_mt = MTRNet_pred(d_test_mt[:,2:-1], method_adv, y_cont=False)
    
    Rpol_hat_adv = Rpol(d_test, ITE_est_adv)
    Rpol_hat_adv_ot = Rpol(d_test_ot, ITE_est_adv_ot)
    Rpol_hat_adv_mt = Rpol(d_test_mt, ITE_est_adv_mt)
    
    jobs_results['MTHD_adv'][0].append(Rpol_hat_adv)
    jobs_results['MTHD_adv'][1].append(Rpol_hat_adv_ot)
    jobs_results['MTHD_adv'][2].append(Rpol_hat_adv_mt)
    jobs_results['MTHD_adv'][3].append(hyperpar_opt_adv)    
    
    ITE_est_IPM = MTRNetIPM_pred(d_test[:,2:-1], method_IPM, y_cont=False)
    ITE_est_IPM_ot = MTRNetIPM_pred(d_test_ot[:,2:-1], method_IPM, y_cont=False)
    ITE_est_IPM_mt = MTRNetIPM_pred(d_test_mt[:,2:-1], method_IPM, y_cont=False)
    
    Rpol_hat_IPM = Rpol(d_test, ITE_est_IPM)
    Rpol_hat_IPM_ot = Rpol(d_test_ot, ITE_est_IPM_ot)
    Rpol_hat_IPM_mt = Rpol(d_test_mt, ITE_est_IPM_mt)
    
    jobs_results['MTHD_IPM'][0].append(Rpol_hat_IPM)
    jobs_results['MTHD_IPM'][1].append(Rpol_hat_IPM_ot)
    jobs_results['MTHD_IPM'][2].append(Rpol_hat_IPM_mt)
    jobs_results['MTHD_IPM'][3].append(hyperpar_opt_IPM) 
    
    # Implementation of benchmarks - 3 * 4 models (d_train_del, d_train_imp, d_train_del + w)
    
    Linear_del = Linear(d_train_del, y_cont=False)
    Linear_imp = Linear(d_train_imp, y_cont=False)
    Linear_rew = Linear_w(d_train_rew, y_cont=False)
    
    CF_del = CF(d_train_del, y_cont=False)
    CF_imp = CF(d_train_imp, y_cont=False)
    CF_rew = CF_w(d_train_rew, y_cont=False)
    
    hyperpar_opt_del1 = CV_hyperparam_search(data_train=d_train_del, data_val=d_val, y_cont=False, 
                                             metric='Rpol', model=TARNet, model_pred=TARNet_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)    
    TARNet_del = TARNet(data_train=d_train_del, y_cont=False, hyperparams=hyperpar_opt_del1)

    hyperpar_opt_imp1 = CV_hyperparam_search(data_train=d_train_imp, data_val=d_val, y_cont=False, 
                                             metric='Rpol', model=TARNet, model_pred=TARNet_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)    
    TARNet_imp = TARNet(data_train=d_train_imp, y_cont=False, hyperparams=hyperpar_opt_imp1)

    hyperpar_opt_rew1 = CV_hyperparam_search(data_train=d_train_rew, data_val=d_val, y_cont=False, 
                                             metric='Rpol', model=TARNet_w, model_pred=TARNet_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)
    TARNet_rew = TARNet_w(data_train=d_train_rew, y_cont=False, hyperparams=hyperpar_opt_rew1)
    
    hyperpar_opt_del2 = CV_hyperparam_search(data_train=d_train_del, data_val=d_val, y_cont=False, 
                                             metric='Rpol', model=CFRWASS, model_pred=CFRWASS_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)
    CFRWASS_del = CFRWASS(data_train=d_train_del, y_cont=False, hyperparams=hyperpar_opt_del2)

    hyperpar_opt_imp2 = CV_hyperparam_search(data_train=d_train_imp, data_val=d_val, y_cont=False, 
                                             metric='Rpol', model=CFRWASS, model_pred=CFRWASS_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)
    CFRWASS_imp = CFRWASS(data_train=d_train_imp, y_cont=False, hyperparams=hyperpar_opt_imp2)

    hyperpar_opt_rew2 = CV_hyperparam_search(data_train=d_train_rew, data_val=d_val, y_cont=False, 
                                             metric='Rpol', model=CFRWASS_w, model_pred=CFRWASS_pred,
                                             hyperparams_list=hyperparams_list, n_search=100)
    CFRWASS_rew = CFRWASS_w(data_train=d_train_rew, y_cont=False, hyperparams=hyperpar_opt_rew2)
    
    # Performance evaluation for benchmarks (d_test_ot, d_test_mt, d_test)
    
    ITE_est_Linear_del = Linear_pred(d_test[:,2:-1], Linear_del, y_cont=False)
    ITE_est_Linear_del_ot = Linear_pred(d_test_ot[:,2:-1], Linear_del, y_cont=False)
    ITE_est_Linear_del_mt = Linear_pred(d_test_mt[:,2:-1], Linear_del, y_cont=False)
    
    Rpol_hat_Linear_del = Rpol(d_test, ITE_est_Linear_del)
    Rpol_hat_Linear_del_ot = Rpol(d_test_ot, ITE_est_Linear_del_ot)
    Rpol_hat_Linear_del_mt = Rpol(d_test_mt, ITE_est_Linear_del_mt)
    
    jobs_results['Linear_del'][0].append(Rpol_hat_Linear_del)
    jobs_results['Linear_del'][1].append(Rpol_hat_Linear_del_ot)
    jobs_results['Linear_del'][2].append(Rpol_hat_Linear_del_mt)
    
    ITE_est_Linear_imp = Linear_pred(d_test[:,2:-1], Linear_imp, y_cont=False)
    ITE_est_Linear_imp_ot = Linear_pred(d_test_ot[:,2:-1], Linear_imp, y_cont=False)
    ITE_est_Linear_imp_mt = Linear_pred(d_test_mt[:,2:-1], Linear_imp, y_cont=False)
    
    Rpol_hat_Linear_imp = Rpol(d_test, ITE_est_Linear_imp)
    Rpol_hat_Linear_imp_ot = Rpol(d_test_ot, ITE_est_Linear_imp_ot)
    Rpol_hat_Linear_imp_mt = Rpol(d_test_mt, ITE_est_Linear_imp_mt)
    
    jobs_results['Linear_imp'][0].append(Rpol_hat_Linear_imp)
    jobs_results['Linear_imp'][1].append(Rpol_hat_Linear_imp_ot)
    jobs_results['Linear_imp'][2].append(Rpol_hat_Linear_imp_mt)
    
    ITE_est_Linear_rew = Linear_pred(d_test[:,2:-1], Linear_rew, y_cont=False)
    ITE_est_Linear_rew_ot = Linear_pred(d_test_ot[:,2:-1], Linear_rew, y_cont=False)
    ITE_est_Linear_rew_mt = Linear_pred(d_test_mt[:,2:-1], Linear_rew, y_cont=False)
    
    Rpol_hat_Linear_rew = Rpol(d_test, ITE_est_Linear_rew)
    Rpol_hat_Linear_rew_ot = Rpol(d_test_ot, ITE_est_Linear_rew_ot)
    Rpol_hat_Linear_rew_mt = Rpol(d_test_mt, ITE_est_Linear_rew_mt)
    
    jobs_results['Linear_rew'][0].append(Rpol_hat_Linear_rew)
    jobs_results['Linear_rew'][1].append(Rpol_hat_Linear_rew_ot)
    jobs_results['Linear_rew'][2].append(Rpol_hat_Linear_rew_mt)
    
    ITE_est_CF_del = CF_pred(d_test[:,2:-1], CF_del, y_cont=False)
    ITE_est_CF_del_ot = CF_pred(d_test_ot[:,2:-1], CF_del, y_cont=False)
    ITE_est_CF_del_mt = CF_pred(d_test_mt[:,2:-1], CF_del, y_cont=False)
    
    Rpol_hat_CF_del = Rpol(d_test, ITE_est_CF_del)
    Rpol_hat_CF_del_ot = Rpol(d_test_ot, ITE_est_CF_del_ot)
    Rpol_hat_CF_del_mt = Rpol(d_test_mt, ITE_est_CF_del_mt)
    
    jobs_results['CF_del'][0].append(Rpol_hat_CF_del)
    jobs_results['CF_del'][1].append(Rpol_hat_CF_del_ot)
    jobs_results['CF_del'][2].append(Rpol_hat_CF_del_mt)
    
    ITE_est_CF_imp = CF_pred(d_test[:,2:-1], CF_imp, y_cont=False)
    ITE_est_CF_imp_ot = CF_pred(d_test_ot[:,2:-1], CF_imp, y_cont=False)
    ITE_est_CF_imp_mt = CF_pred(d_test_mt[:,2:-1], CF_imp, y_cont=False)
    
    Rpol_hat_CF_imp = Rpol(d_test, ITE_est_CF_imp)
    Rpol_hat_CF_imp_ot = Rpol(d_test_ot, ITE_est_CF_imp_ot)
    Rpol_hat_CF_imp_mt = Rpol(d_test_mt, ITE_est_CF_imp_mt)
    
    jobs_results['CF_imp'][0].append(Rpol_hat_CF_imp)
    jobs_results['CF_imp'][1].append(Rpol_hat_CF_imp_ot)
    jobs_results['CF_imp'][2].append(Rpol_hat_CF_imp_mt)
    
    ITE_est_CF_rew = CF_pred(d_test[:,2:-1], CF_rew, y_cont=False)
    ITE_est_CF_rew_ot = CF_pred(d_test_ot[:,2:-1], CF_rew, y_cont=False)
    ITE_est_CF_rew_mt = CF_pred(d_test_mt[:,2:-1], CF_rew, y_cont=False)
    
    Rpol_hat_CF_rew = Rpol(d_test, ITE_est_CF_rew)
    Rpol_hat_CF_rew_ot = Rpol(d_test_ot, ITE_est_CF_rew_ot)
    Rpol_hat_CF_rew_mt = Rpol(d_test_mt, ITE_est_CF_rew_mt)
    
    jobs_results['CF_rew'][0].append(Rpol_hat_CF_rew)
    jobs_results['CF_rew'][1].append(Rpol_hat_CF_rew_ot)
    jobs_results['CF_rew'][2].append(Rpol_hat_CF_rew_mt)
    
    ITE_est_TARNet_del = TARNet_pred(d_test[:,2:-1], TARNet_del, y_cont=False)
    ITE_est_TARNet_del_ot = TARNet_pred(d_test_ot[:,2:-1], TARNet_del, y_cont=False)
    ITE_est_TARNet_del_mt = TARNet_pred(d_test_mt[:,2:-1], TARNet_del, y_cont=False)
    
    Rpol_hat_TARNet_del = Rpol(d_test, ITE_est_TARNet_del)
    Rpol_hat_TARNet_del_ot = Rpol(d_test_ot, ITE_est_TARNet_del_ot)
    Rpol_hat_TARNet_del_mt = Rpol(d_test_mt, ITE_est_TARNet_del_mt)
    
    jobs_results['TARNet_del'][0].append(Rpol_hat_TARNet_del)
    jobs_results['TARNet_del'][1].append(Rpol_hat_TARNet_del_ot)
    jobs_results['TARNet_del'][2].append(Rpol_hat_TARNet_del_mt)
    jobs_results['TARNet_del'][3].append(hyperpar_opt_del1)
    
    ITE_est_TARNet_imp = TARNet_pred(d_test[:,2:-1], TARNet_imp, y_cont=False)
    ITE_est_TARNet_imp_ot = TARNet_pred(d_test_ot[:,2:-1], TARNet_imp, y_cont=False)
    ITE_est_TARNet_imp_mt = TARNet_pred(d_test_mt[:,2:-1], TARNet_imp, y_cont=False)
    
    Rpol_hat_TARNet_imp = Rpol(d_test, ITE_est_TARNet_imp)
    Rpol_hat_TARNet_imp_ot = Rpol(d_test_ot, ITE_est_TARNet_imp_ot)
    Rpol_hat_TARNet_imp_mt = Rpol(d_test_mt, ITE_est_TARNet_imp_mt)
    
    jobs_results['TARNet_imp'][0].append(Rpol_hat_TARNet_imp)
    jobs_results['TARNet_imp'][1].append(Rpol_hat_TARNet_imp_ot)
    jobs_results['TARNet_imp'][2].append(Rpol_hat_TARNet_imp_mt)
    jobs_results['TARNet_imp'][3].append(hyperpar_opt_imp1)
    
    ITE_est_TARNet_rew = TARNet_pred(d_test[:,2:-1], TARNet_rew, y_cont=False)
    ITE_est_TARNet_rew_ot = TARNet_pred(d_test_ot[:,2:-1], TARNet_rew, y_cont=False)
    ITE_est_TARNet_rew_mt = TARNet_pred(d_test_mt[:,2:-1], TARNet_rew, y_cont=False)
    
    Rpol_hat_TARNet_rew = Rpol(d_test, ITE_est_TARNet_rew)
    Rpol_hat_TARNet_rew_ot = Rpol(d_test_ot, ITE_est_TARNet_rew_ot)
    Rpol_hat_TARNet_rew_mt = Rpol(d_test_mt, ITE_est_TARNet_rew_mt)
    
    jobs_results['TARNet_rew'][0].append(Rpol_hat_TARNet_rew)
    jobs_results['TARNet_rew'][1].append(Rpol_hat_TARNet_rew_ot)
    jobs_results['TARNet_rew'][2].append(Rpol_hat_TARNet_rew_mt)
    jobs_results['TARNet_rew'][3].append(hyperpar_opt_rew1)
    
    ITE_est_CFRWASS_del = CFRWASS_pred(d_test[:,2:-1], CFRWASS_del, y_cont=False)
    ITE_est_CFRWASS_del_ot = CFRWASS_pred(d_test_ot[:,2:-1], CFRWASS_del, y_cont=False)
    ITE_est_CFRWASS_del_mt = CFRWASS_pred(d_test_mt[:,2:-1], CFRWASS_del, y_cont=False)
    
    Rpol_hat_CFRWASS_del = Rpol(d_test, ITE_est_CFRWASS_del)
    Rpol_hat_CFRWASS_del_ot = Rpol(d_test_ot, ITE_est_CFRWASS_del_ot)
    Rpol_hat_CFRWASS_del_mt = Rpol(d_test_mt, ITE_est_CFRWASS_del_mt)
    
    jobs_results['CFRWASS_del'][0].append(Rpol_hat_CFRWASS_del)
    jobs_results['CFRWASS_del'][1].append(Rpol_hat_CFRWASS_del_ot)
    jobs_results['CFRWASS_del'][2].append(Rpol_hat_CFRWASS_del_mt)
    jobs_results['CFRWASS_del'][3].append(hyperpar_opt_del2)
    
    ITE_est_CFRWASS_imp = CFRWASS_pred(d_test[:,2:-1], CFRWASS_imp, y_cont=False)
    ITE_est_CFRWASS_imp_ot = CFRWASS_pred(d_test_ot[:,2:-1], CFRWASS_imp, y_cont=False)
    ITE_est_CFRWASS_imp_mt = CFRWASS_pred(d_test_mt[:,2:-1], CFRWASS_imp, y_cont=False)
    
    Rpol_hat_CFRWASS_imp = Rpol(d_test, ITE_est_CFRWASS_imp)
    Rpol_hat_CFRWASS_imp_ot = Rpol(d_test_ot, ITE_est_CFRWASS_imp_ot)
    Rpol_hat_CFRWASS_imp_mt = Rpol(d_test_mt, ITE_est_CFRWASS_imp_mt)
    
    jobs_results['CFRWASS_imp'][0].append(Rpol_hat_CFRWASS_imp)
    jobs_results['CFRWASS_imp'][1].append(Rpol_hat_CFRWASS_imp_ot)
    jobs_results['CFRWASS_imp'][2].append(Rpol_hat_CFRWASS_imp_mt)
    jobs_results['CFRWASS_imp'][3].append(hyperpar_opt_imp2)
    
    ITE_est_CFRWASS_rew = CFRWASS_pred(d_test[:,2:-1], CFRWASS_rew, y_cont=False)
    ITE_est_CFRWASS_rew_ot = CFRWASS_pred(d_test_ot[:,2:-1], CFRWASS_rew, y_cont=False)
    ITE_est_CFRWASS_rew_mt = CFRWASS_pred(d_test_mt[:,2:-1], CFRWASS_rew, y_cont=False)
    
    Rpol_hat_CFRWASS_rew = Rpol(d_test, ITE_est_CFRWASS_rew)
    Rpol_hat_CFRWASS_rew_ot = Rpol(d_test_ot, ITE_est_CFRWASS_rew_ot)
    Rpol_hat_CFRWASS_rew_mt = Rpol(d_test_mt, ITE_est_CFRWASS_rew_mt)
    
    jobs_results['CFRWASS_rew'][0].append(Rpol_hat_CFRWASS_rew)
    jobs_results['CFRWASS_rew'][1].append(Rpol_hat_CFRWASS_rew_ot)
    jobs_results['CFRWASS_rew'][2].append(Rpol_hat_CFRWASS_rew_mt)
    jobs_results['CFRWASS_rew'][3].append(hyperpar_opt_rew2)
    



