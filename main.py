# -*- coding: utf-8 -*-
"""

This code contains the implementation of our method on a dummy data set

"""

#%% Packages

import math
import numpy as np
# from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#%% Dummy data

# # Setting:  
# #     Outcome:                 Y 
# #     Treatment:               T 
# #     Observed covariates:     X_vec = [x1, ..., xp]
# #     Treatment observability: R

# # Simulation setup:
# #     Y = alpha*T + beta_vec*X_vec

# # Simulate data
# n = 100; p = 10; 
# X = np.random.normal(loc=0, scale=1, size=(n,p))
# beta_vec = np.random.normal(loc=1, scale=1, size=p)
# T = np.random.choice(a=2, size=n)
# alpha = np.random.normal(loc=1, scale=1, size=1)
# Y_true = alpha*T + np.dot(X, beta_vec)
# Y = alpha*T + np.dot(X, beta_vec) + np.random.normal(size=n)

# # Covariate-dependent masking of treatment (~50% missingness in T)
# R = np.sum(X, axis=1) > 0
# data = np.concatenate((Y_true.reshape(n,1), Y.reshape(n,1),
#                         T.reshape(n,1),X,R.reshape(n,1)), axis=1)  

# # Train-Validation-Test split (90/10)
# d_train, d_test = train_test_split(data, test_size=0.1, shuffle=True)
 
# # Hyperparams
# hyperparams = {
#     'layer_size_rep': [200],
#     'layer_size_hyp': [100],
#     'lr': [0.01],
#     'drop': [0.1],
#     'n_iter': [100],
#     'b_size': [100],
#     'alpha': [0.1],
#     'beta': [0.1],
#     'gamma': [0.1],
#     'lam': [0.0001]
#     }

#%% 1) Method implementation - Adversarial learning approach

############################## Method train ##################################

# Input: train and validation data of shape [Y, T, X, R]
# Output: trained model 

def MTRNet(data_train, y_cont, hyperparams):
    
    n = data_train.shape[0]
    p = data_train.shape[1] - 3
    
    if y_cont:
        y_size = 1 # For continuous Y
    else:
        y_size = len(np.unique(data_train[:,0])) # For categorical Y
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    layer_size_rep = hyperparams['layer_size_rep'][0]
    layer_size_hyp = hyperparams['layer_size_hyp'][0]
    drop = hyperparams['drop'][0]
    lr = hyperparams['lr'][0]
    n_iter = hyperparams['n_iter'][0]
    b_size = hyperparams['b_size'][0]
    alpha = hyperparams['alpha'][0]
    beta = hyperparams['beta'][0]
    gamma = hyperparams['gamma'][0]
    lam = hyperparams['lam'][0]
    
    # Add weights for treated-control ratio
    w_trt = np.empty(n) 
    mean_trt = np.mean(data_train[:,1])
    for i in range(n):
        w_trt[i] = data_train[i,1]/(2*mean_trt) + (1-data_train[i,1])/(2*(1-mean_trt))
    d_train = np.concatenate((data_train, w_trt.reshape(n,1)), axis=1) 
            
    # Data pre-processing 
    train = torch.from_numpy(d_train.astype(np.float32))
    train_loader = DataLoader(dataset=train, batch_size=math.ceil(b_size), shuffle=True)
    
    # Model 
    class Representation(nn.Module):
        def __init__(self, x_size, layer_size_rep, drop):
            super(Representation, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(x_size, layer_size_rep),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_rep, layer_size_rep),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_rep, layer_size_rep),  
                nn.ELU(),
                nn.Dropout(drop)
                )
        def forward(self, x):
            rep = self.model(x)
            return rep  
        
    class HypothesisT0(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT0, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(layer_size_rep, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_hyp, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_hyp, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),   
                nn.Linear(layer_size_hyp, y_size)
                )            
        def forward(self, rep):
            y0_out = self.model(rep)
            return y0_out  
    
    class HypothesisT1(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT1, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(layer_size_rep, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_hyp, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_hyp, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),   
                nn.Linear(layer_size_hyp, y_size)
                )            
        def forward(self, rep):
            y1_out = self.model(rep)
            return y1_out 
        
    class X_pred(nn.Module):
        def __init__(self, layer_size_rep, x_size, drop):
            super(X_pred, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(layer_size_rep, layer_size_rep),
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_rep, x_size)
                )    
        def forward(self, rep):
            x_out = self.model(rep)
            return x_out
    
    class T_pred(nn.Module):
        def __init__(self, layer_size_rep, drop):
            super(T_pred, self).__init__()
            self.model = nn.Sequential(
                # nn.Linear(layer_size_rep, layer_size_rep),
                # nn.ELU(),
                # nn.Dropout(drop),
                nn.Linear(layer_size_rep, 2)
                )    
        def forward(self, rep):
            t_out = self.model(rep)
            return t_out
        
    class R_pred(nn.Module):
        def __init__(self, layer_size_rep, drop):
            super(R_pred, self).__init__()
            self.model = nn.Sequential(
                # nn.Linear(layer_size_rep, layer_size_rep),
                # nn.ELU(),
                # nn.Dropout(drop),
                nn.Linear(layer_size_rep, 2)
                )    
        def forward(self, rep):
            r_out = self.model(rep)
            return r_out    
    
    class MTRNet(nn.Module):
        def __init__(self, x_size, layer_size_rep, layer_size_hyp, y_size, drop):
            super(MTRNet, self).__init__()
            self.representation = Representation(x_size, layer_size_rep, drop)
            self.hypothesisT0 = HypothesisT0(layer_size_rep, layer_size_hyp, y_size, drop)
            self.hypothesisT1 = HypothesisT1(layer_size_rep, layer_size_hyp, y_size, drop)
            self.x_pred = X_pred(layer_size_rep, x_size, drop)
            self.t_pred = T_pred(layer_size_rep, drop)
            self.r_pred = R_pred(layer_size_rep, drop)
            
        def forward(self, x):
            rep = self.representation(x)
            y0_out = self.hypothesisT0(rep)
            y1_out = self.hypothesisT1(rep)
            t_out = self.t_pred(rep)
            r_out = self.r_pred(rep)
            x_out = self.x_pred(rep)
            return y0_out, y1_out, t_out, r_out, x_out
    
    mod_MTRNet = MTRNet(x_size=p, layer_size_rep=layer_size_rep, 
                        layer_size_hyp=layer_size_hyp, y_size=y_size, drop=drop).to(device) 
    optimizer = torch.optim.Adam([{'params': mod_MTRNet.representation.parameters(), 'weight_decay': 0},
                                  {'params': mod_MTRNet.hypothesisT0.parameters()},
                                  {'params': mod_MTRNet.hypothesisT1.parameters()}],
                                 lr=lr, weight_decay=lam)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.97)    
    MSE_Y = nn.MSELoss(reduction='none') # For continuous Y
    Entropy_Y = nn.CrossEntropyLoss(reduction='none') # For categorical Y
    Entropy = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    
    # Train the model
    for iteration in range(n_iter):
        batch = next(iter(train_loader))
        if y_size == 1:
            y_data = batch[:,0:1]
        else:
            y_data = batch[:,0].long()
        t_data = batch[:,1].long()
        x_data = batch[:,2:(p+2)]
        r_data = batch[:,p+2].long()
        wt_data = batch[:,(p+3):]
        y_data = y_data.to(device)
        t_data = t_data.to(device)
        x_data = x_data.to(device)
        r_data = r_data.to(device)
        wt_data = wt_data.to(device)
        
        # Forward pass
        y0_out, y1_out, t_out, r_out, x_out = mod_MTRNet(x_data)
        
        # Losses
        idxR1 = torch.where(r_data == 1)
        idxR1T0 = torch.where((r_data == 1) & (t_data == 0)) 
        idxR1T1 = torch.where((r_data == 1) & (t_data == 1))
        if y_size == 1:
            lossY = torch.mean(wt_data[idxR1T0]*MSE_Y(y0_out[idxR1T0], y_data[idxR1T0])) + \
                torch.mean(wt_data[idxR1T1]*MSE_Y(y1_out[idxR1T1], y_data[idxR1T1])) # For continuous Y
        else:
            lossY = torch.mean(wt_data[idxR1T0]*Entropy_Y(y0_out[idxR1T0], y_data[idxR1T0])) + \
                torch.mean(wt_data[idxR1T1]*Entropy_Y(y1_out[idxR1T1], y_data[idxR1T1])) # For categorical Y
        lossT = Entropy(t_out[idxR1], t_data[idxR1])
        lossR = Entropy(r_out, r_data)
        lossX = MSE(x_out, x_data)
        loss = lossY - alpha*lossT - beta*lossR + gamma*lossX
                
        # Backward and optimize 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (iteration+1) % 1 == 0:
            print (f'Epoch [{iteration+1}/{n_iter}], Loss: {loss.item():.4f}')
            print(scheduler.get_last_lr())
        scheduler.step()
            
    return mod_MTRNet

# mod_MTRNet = MTRNet(d_train[:,1:])

############################## Method test  ##################################

# Input: test data of shape [X], trained model
# Output: PO prediction on test data

def MTRNet_pred(data_test, model, y_cont):
    data_test = torch.from_numpy(data_test.astype(np.float32))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_test = data_test.to(device)
    y0_test, y1_test, _ , _ , _ = model(data_test)
    if not y_cont:
        y0_test = nn.Softmax(dim=1)(y0_test)[:,1]
        y1_test = nn.Softmax(dim=1)(y1_test)[:,1]
    ITE = y1_test - y0_test
    return ITE
