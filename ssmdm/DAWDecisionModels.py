# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:48:04 2026

@author: dweiss38
"""
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln

import ssm
from ssm import hmm, lds
from ssm.hmm import HMM
from ssm.lds import SLDS
from ssm.util import random_rotation, ensure_args_are_lists, softplus
from ssm.observations import Observations, AutoRegressiveDiagonalNoiseObservations
from ssm.transitions import Transitions, RecurrentTransitions, RecurrentOnlyTransitions
from ssm.init_state_distns import InitialStateDistribution
from ssm.emissions import _LinearEmissions, GaussianEmissions, PoissonEmissions
from ssm.preprocessing import factor_analysis_with_imputation, interpolate_data, pca_with_imputation
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, lbfgs, bfgs, convex_combination
from ssm.primitives import hmm_normalizer

import ssm.stats as stats
from scipy.stats import norm
from autograd.scipy.special import erf

from ssm.messages import _kalman_filter, make_lds_parameters

from ssmdm.misc import smooth

import copy
import scipy

from tqdm import tqdm
from tqdm.auto import trange
from autograd.scipy.special import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad


class DecisionModel(object):
    
    def __init__(self,Dacc,D,sig_dec=0.01,decdims=[0,1]):
        self.Dacc = Dacc
        self.D = D
        self.Demb = D - Dacc
        
        self.sigma_decision = sig_dec*np.eye(self.D)
        self.decdims = decdims
        
        # Create decision state readout matrix H
        mat = []
        for d1 in range(self.D):
            row = []
            for d2 in range(self.D):
                if np.isin(d1,self.decdims) and d1 == d2:
                    row.append(1)
                else:
                    row.append(0)
            mat.append(row)
        
        self.H = np.vstack(mat)

    
    def log_likelihoods(self, mus, input, mask, tag):
        K,T,D = mus.shape
        # Choice likelihood
        if tag is not None:
            
            # Create true choice vector
            wt = []
            for d in range(self.D):
                if d == self.decdims[tag]:
                    wt.append(1)
                else:
                    wt.append(0)
            wT = np.array(wt)[:,None]
            
            # logP_dec = np.array([np.squeeze(np.log(2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision)) -\
            #     0.5*(wT- (self.H @ mu[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wT- (self.H @ mu[-1])[:,None])) \
            #     for mu in mus])[None,:]
            logP_dec = np.column_stack([
                    stats.multivariate_normal_logpdf(np.tile(wT.T,(T,1)), (self.H @ mu.T).T, self.sigma_decision)
                    for mu in mus])
            
            ll_dec = np.zeros((T-1,K))
            ll_decision = np.row_stack((ll_dec,logP_dec[-1,:]))
        else:
            ll_decision = np.zeros((T,K))
          
        return ll_decision
            
    def Rchoice_likelihood(self,x,q_Sigmas,decdims=None):
        if decdims is None:
            decdims = self.decdims

        wT = []
        for decdim in decdims:
            row = []
            for d in range(self.D):
                if d == decdim:
                    row.append(1)
                else:
                    row.append(0)
            wT.append(row)
            
        wT = np.array(wT)
        wTL = wT[0,:,None]
        wTR = wT[1,:,None]
        
        H = np.zeros((self.D,self.D))
        
        for d in decdims:
            H[d,d] = 1

        
        P_R = 2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision) *\
            np.exp(-0.5*(wTR - (H @ x[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wTR - (H @ x[-1])[:,None]))
            
        P_L = 2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision) *\
            np.exp(-0.5*(wTL - (H @ x[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wTL - (H @ x[-1])[:,None]))
        
        prob = P_R/(P_R+P_L)
        
        return prob
    
    def Rchoice_likelihoods(self,xs,q_Sigmass,decdims=None):
        if decdims is None:
            decdims = self.decdims
        prob = np.zeros((len(xs),))
        for i,(x,q_Sigmas) in enumerate(zip(xs,q_Sigmass)):
            prob[i] = self.Rchoice_likelihood(x,q_Sigmas[2],decdims=decdims)
            
        return prob
    
    # def Rchoice_likelihood(self,x,q_Sigmas):
    #     q_stds = np.array([np.sqrt(np.diag(q_Sigma)) for q_Sigma in q_Sigmas])
    #     if self.Dacc > 1:
    #         delta = x[-1,self.decdims[0]]-x[-1,self.decdims[1]]
    #         z = delta / (np.sqrt(q_stds[-1,0]*q_stds[-1,1]))
    #     else:
    #         z = x[-1,0] / np.sqrt(q_stds[-1,0])
    #     prob = 0.5 * (1 + erf(z))
    #     # Avoid log(0)
    #     eps = 1e-12
    #     prob = np.clip(prob, eps, 1 - eps)
    
    #     return 1-prob
    
    # def Rchoice_likelihood(self,x,q_Sigmas,decdims=[0,1]):
    #     q_stds = np.array([np.sqrt(np.diag(q_Sigma)) for q_Sigma in q_Sigmas])
    #     if self.Dacc > 1:
    #         delta = x[-1,decdims[0]]-x[-1,decdims[1]]
    #         z = delta / (np.sqrt(q_stds[-1,0]*q_stds[-1,1]))
    #     else:
    #         z = x[-1,0] / np.sqrt(q_stds[-1,0])
    #     prob = 0.5 * (1 + erf(z))
    #     # Avoid log(0)
    #     eps = 1e-12
    #     prob = np.clip(prob, eps, 1 - eps)
    
    #     return 1-prob
    
    # def Rchoice_likelihood(self,x,q_Sigmas):
    #     # ll_decision = np.row_stack((ll_dec,logP_dec))
    #     wT = []
    #     decdims = [(d+1) * self.cascades - 1 for d in range(self.D_acc)]
    #     for decdim in decdims:
    #         row = []
    #         for d in range(self.D):
    #             if d == decdim:
    #                 row.append(1)
    #             else:
    #                 row.append(0)
    #         wT.append(row)
            
    #     wT = np.array(wT)
    #     wTL = wT[0,:,None]
    #     wTR = wT[0,:,None]
    #     # wTR = np.array([0,1,0,0])[:,None]
    #     # wTL = np.array([1,0,0,0])[:,None]
        
    #     H = np.zeros((self.D,self.D))
        
    #     for d in decdims:
    #         H[d,d] = 1

        
    #     P_R = 2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision) *\
    #         np.exp(-0.5*(wTR - (H @ x[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wTR - (H @ x[-1])[:,None]))
            
    #     P_L = 2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision) *\
    #         np.exp(-0.5*(wTL - (H @ x[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wTL - (H @ x[-1])[:,None]))
        
    #     prob = P_R/(P_R+P_L)
    #     return prob
        
    