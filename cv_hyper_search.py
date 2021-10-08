# -*- coding: utf-8 -*-
"""

This code contains the hyperparameter search for model/data set pair via cross validation

"""

#%% Packages

import numpy as np
import pandas as pd
import torch

#%% Hyperparameter search

# Required input: 1) data (with corresponding outcome type and performance metric) 
#                 2) model class 
#                 3) list of hyperparameters for stochastic grid search

def PEHE(ITE_true, ITE_est):
     if type(ITE_est) == torch.Tensor:
         ITE_est = ITE_est.cpu().detach().numpy()
     PEHE = np.square(np.subtract(ITE_true, ITE_est)).mean()
     return np.sqrt(PEHE)
   
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
    
def CV_hyperparam_search(data_train, data_val, y_cont, metric, model, 
                         model_pred, hyperparams_list, n_search):
    
    cv_results = []
    
    # Cross validation
    for i in range(n_search):
        print(i)
        
        n_iter = np.random.choice(hyperparams_list['num_iterations'])
        lr = np.random.choice(hyperparams_list['learn_rate'])
        b_size = np.random.choice(hyperparams_list['batch_size'])
        layer_size_rep = np.random.choice(hyperparams_list['layer_size_representation'])
        layer_size_hyp = np.random.choice(hyperparams_list['layer_size_hypothesis'])
        drop = np.random.choice(hyperparams_list['dropout_rate'])
        alpha = np.random.choice(hyperparams_list['alphas'])
        beta = np.random.choice(hyperparams_list['betas'])
        gamma = np.random.choice(hyperparams_list['gammas'])
        lam = np.random.choice(hyperparams_list['lambdas'])
      
        # Hyperparameters
        hyperparams = {
            'layer_size_rep': [layer_size_rep],
            'layer_size_hyp': [layer_size_hyp],
            'lr': [lr],
            'drop': [drop],
            'n_iter': [n_iter],
            'b_size': [b_size],
            'alpha': [alpha],
            'beta': [beta],
            'gamma': [gamma],
            'lam': [lam]
            }
        
        # Predicted ITE using model
        trained_mod = model(data_train, y_cont, hyperparams)
        ITE_est = model_pred(data_val[:,2:], trained_mod, y_cont)
        
        if metric == 'PEHE':
            ITE_true = np.empty(shape=ITE_est.shape)
            if data_train.shape[1] == data_val.shape[1]:
                data_full = np.concatenate((data_train, data_val), axis=0)
            elif data_train.shape[1] == data_val.shape[1] + 1:   
                data_full = np.concatenate((data_train[:,:-1], data_val), axis=0)
            idxT0 = np.where(data_full[:,1] == 0)
            idxT1 = np.where(data_full[:,1] == 1)
            for i in range(ITE_true.shape[0]):
                Y_obs = data_val[i,0] 
                T_obs = data_val[i,1] 
                if T_obs == 1:
                    distances = np.linalg.norm(data_full[idxT0][:,2:] - data_val[i,2:], axis=1)
                    nn_id = distances.argsort()[0]
                    Y_cf = data_full[idxT0][nn_id, 0]
                    ITE_true[i] = Y_obs - Y_cf
                elif T_obs == 0:
                    distances = np.linalg.norm(data_full[idxT1][:,2:] - data_val[i,2:], axis=1)
                    nn_id = distances.argsort()[0]
                    Y_cf = data_full[idxT1][nn_id, 0]
                    ITE_true[i] = Y_cf - Y_obs 
            loss_val = PEHE(ITE_true, ITE_est)
        elif metric == 'Rpol':
            loss_val = Rpol(data_val, ITE_est)
        
        # Copmute loss and add to CV data
        cv_results.append({
            'Loss': loss_val,
            'Rep layer size': layer_size_rep,
            'Hyp layer size': layer_size_hyp,
            'Learning rate': lr,
            'Dropout': drop,
            'Num iterations': n_iter,
            'Batch size': b_size,
            'Alpha': alpha,
            'Beta': beta,
            'Gamma': gamma,
            'Lambda': lam
            })
    
    cv_results = pd.DataFrame(cv_results)
    cv_results = cv_results.sort_values(by='Loss', ascending=True)
    
    layer_size_rep_opt = cv_results['Rep layer size'][0]
    layer_size_hyp_opt = cv_results['Hyp layer size'][0]
    lr_opt = cv_results['Learning rate'][0]
    drop_opt = cv_results['Dropout'][0]
    n_iter_opt = cv_results['Num iterations'][0]
    b_size_opt = cv_results['Batch size'][0]
    alpha_opt = cv_results['Alpha'][0]
    beta_opt = cv_results['Beta'][0]
    gamma_opt = cv_results['Gamma'][0]
    lam_opt = cv_results['Lambda'][0]
    
    hyperparams_out = {
            'layer_size_rep': [layer_size_rep_opt],
            'layer_size_hyp': [layer_size_hyp_opt],
            'lr': [lr_opt],
            'drop': [drop_opt],
            'n_iter': [n_iter_opt],
            'b_size': [b_size_opt],
            'alpha': [alpha_opt],
            'beta': [beta_opt],
            'gamma': [gamma_opt],  
            'lam': [lam_opt]
            }
    
    return hyperparams_out
    
    
    
    
    
    
