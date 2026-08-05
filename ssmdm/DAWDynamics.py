# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:42:34 2026

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

class DAWEmbeddedAccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, D_emb=None, D_acc=None, Mphys=2,learn_V=True):
        super().__init__(K, D, M)

        assert D_emb + D_acc == D, "accum + emb dims must equal total dims"
        
        self.D_acc = D_acc
        self.D_emb = D_emb
        self.D = D_acc + D_emb
        self.M = M
        self.K = K
        self.Mphys = Mphys

        # diagonal dynamics for each state
        # only learn accumulation dynamics for accumulation state
        self._a_diag = np.ones((D_acc,1))
        # Make random dynamics matrix for embedded dimensions
        
        # Diag = np.diag(np.random.uniform(low=0.7,high=0.8,size=(D_emb,)))# * \
        #                #(np.array([1 if x > 0.5  else -1 for x in np.random.random(Demb)])))
        
        # # Make random eigenvector matrix
        # Q = np.random.uniform(low=-1,high=1,size=(D_emb,D_emb)) 
        # self._a_emb = (Q @ Diag @ np.linalg.inv(Q)).flatten()
        #self._a_emb = random_rotation(D_emb)*np.random.uniform(low=0.7,high=0.8)
        
        # (M @ A)v = (M @ \lambda)v
        
        self._a_emb = np.random.uniform(low=0.9,high=0.95,size=(D_emb,))*np.eye(D_emb) #@ random_rotation(D_emb)
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
        if D_acc > 1:
            
            ### This is new, for all dims
            self._betas = 0.1 * np.ones(M,)
            self._betas_emb = 0.1 * np.ones(D_emb*M,)
            r1 = self._betas*np.eye(D_acc, M)
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
        else:
            # set input Accumulation params, each Mphys inputs are accumulated on one
            # dimension in opposite directions
            # rest of M-D inputs are applied to each dimension
            self._betas = np.array([0.1,-0.1])*np.ones(Mphys,)
            self._betas_emb = 0.1 * np.ones(D_emb*M,)
            r1 = np.hstack((self._betas[None,:],np.zeros((1,M-Mphys))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)

        # self.Vs[0] = np.vstack((r1, r2))

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self._betas_emb,  self.accum_log_sigmasq, self._a_diag, self._a_emb
        return params

    @params.setter
    def params(self, value):
        self._betas, self._betas_emb, self.accum_log_sigmasq, self._a_diag, self._a_emb = value

        K, D, M, D_acc, D_emb = self.K, self.D, self.M, self.D_acc, \
            self.D_emb

        # Update V
        ### This is original, for first D_acc dims
        # k1 = np.vstack((self._betas * np.eye(self.D_acc), np.zeros((self.D_emb,self.D_acc)))) # state K = 0
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        ### This is new, for all dims
        # k1 = self._betas * np.eye(self.D, self.M)
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        
        # Update V
        if D_acc > 1:
            r1 = self._betas*np.eye(D_acc, M)
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
        else:
            # set input Accumulation params, each Mphys inputs are accumulated on one
            # dimension in opposite directions
            # rest of M-D inputs are applied to each dimension
            r1 = np.hstack((self._betas,np.zeros((1,M-self.Mphys))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
       
        

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        # mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = self._a_diag*mask1 + mask2
        
        # top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        # bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        # A_acc = np.vstack((top_rows,bottom_rows))
        # mask1 = np.vstack( (A_acc[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = mask1 + mask2
        
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        #bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),np.reshape(self._a_emb,(D_emb,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
    def Rchoice_likelihood(self,x,q_Sigmas):
        q_stds = np.array([np.sqrt(np.diag(q_Sigma)) for q_Sigma in q_Sigmas])
        delta = x[-1,0]-x[-1,1]
        z = delta / (np.sqrt(q_stds[-1,0]*q_stds[-1,1]))
        prob = 0.5 * (1 + erf(z))
        # Avoid log(0)
        eps = 1e-12
        prob = np.clip(prob, eps, 1 - eps)
        return 1-prob
    
    def Rchoice_likelihoods(self,xs,q_Sigmass):
        prob = np.zeros((len(xs),))
        for i,(x,q_Sigmas) in enumerate(zip(xs,q_Sigmass)):
            prob[i] = self.Rchoice_likelihood(x,q_Sigmas[2])
            
        return prob
    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask

    def log_prior(self):
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum( -(alpha+1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass
    
    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


class DAWEmbeddedAccumulationObservationsWithOpto(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, D_emb=None, D_acc=None, Mphys=None, Mopto=None,learn_V=True):
        super().__init__(K, D, M)

        assert D_emb + D_acc == D, "accum + emb dims must equal total dims"
        #assert Mphys + Mopto == M, "accum + emb dims must equal total dims"
        
        self.D_acc = D_acc
        self.D_emb = D_emb
        self.D = D_acc + D_emb
        self.Mphys = Mphys
        self.Mopto = Mopto
        self.M = M
        self.K = K
        
        self.sigma_decision = 0.1*np.eye(self.D)  # Variance scale for decision model

        # diagonal dynamics for each state
        # only learn accumulation dynamics for accumulation state
        self._a_diag = np.ones((D_acc,1))
        # Make random dynamics matrix for embedded dimensions
        
        # Diag = np.diag(np.random.uniform(low=0.7,high=0.8,size=(D_emb,)))# * \
        #                #(np.array([1 if x > 0.5  else -1 for x in np.random.random(Demb)])))
        
        # # Make random eigenvector matrix
        # Q = np.random.uniform(low=-1,high=1,size=(D_emb,D_emb)) 
        # self._a_emb = (Q @ Diag @ np.linalg.inv(Q)).flatten()
        #self._a_emb = random_rotation(D_emb)*np.random.uniform(low=0.7,high=0.8)
        
        # (M @ A)v = (M @ \lambda)v
        self._a_emb = (np.random.uniform(low=0.85,high=0.9,size=(D_emb,))*np.eye(D_emb)) @ random_rotation(D_emb)
        
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        ### This is original, for first D_acc dims
        # self._betas = 0.1*np.ones(D_acc,)
        # r1 = self._betas*np.eye(D_acc, D_acc)
        # r2 = np.zeros((self.D_emb, M)) # is D_emb rows by input # of columns 
        # self.Vs[0] = np.vstack((r1, r2))
        ### This is new, for all dims
        self._betas = 0.1 * np.ones(Mphys,)
        self._betas_emb = 0.1 * np.ones(D_emb*Mphys,)
        r1 = self._betas*np.eye(D_acc, Mphys)
        r2 = np.reshape(self._betas_emb,(D_emb,Mphys))
        k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,Mphys))))
        #k2 = np.tile(r2[None,:,:],(K,1,1))
        k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,Mphys))))
        betas_phys = np.concatenate([k1,k2],axis=1)
        self._betas_opto = 0.1 * np.ones((D, Mopto))
        k3 = np.tile(self._betas_opto[None,:,:], (K,1,1))
        
        self.Vs = np.concatenate([betas_phys,k3],axis=2)
        # self.Vs[0] = np.vstack((r1, r2))

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self._betas_emb, self._betas_opto, self.accum_log_sigmasq, self._a_diag, self._a_emb
        return params

    @params.setter
    def params(self, value):
        self._betas, self._betas_emb, self._betas_opto, self.accum_log_sigmasq, self._a_diag, self._a_emb = value

        K, D, M, D_acc, D_emb, Mphys, Mopto = self.K, self.D, self.M, self.D_acc, \
            self.D_emb, self.Mphys, self.Mopto

        # Update V
        ### This is original, for first D_acc dims
        # k1 = np.vstack((self._betas * np.eye(self.D_acc), np.zeros((self.D_emb,self.D_acc)))) # state K = 0
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        ### This is new, for all dims
        # k1 = self._betas * np.eye(self.D, self.M)
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))

        # r1 = self._betas*np.eye(D_acc, M)
        # k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
        # r2 = np.reshape(self._betas_emb,(D_emb,M))
        # #k2 = np.tile(r2[None,:,:],(K,1,1))
        # k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
        # self.Vs = np.concatenate([k1,k2],axis=1)
        
        r1 = self._betas*np.eye(D_acc, Mphys)
        r2 = np.reshape(self._betas_emb,(D_emb,Mphys))
        k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,Mphys))))
        #k2 = np.tile(r2[None,:,:],(K,1,1))
        k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,Mphys))))
        betas_phys = np.concatenate([k1,k2],axis=1)     
        k3 = np.tile(self._betas_opto[None,:,:], (K,1,1))
        
        self.Vs = np.concatenate([betas_phys,k3],axis=2)
        

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        # mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = self._a_diag*mask1 + mask2
        
        # top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        # bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        # A_acc = np.vstack((top_rows,bottom_rows))
        # mask1 = np.vstack( (A_acc[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = mask1 + mask2
        
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        #bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),np.reshape(self._a_emb,(D_emb,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
    def log_likelihoods(self, data, input, mask, tag=None):
        assert np.all(mask), "Cannot compute likelihood of autoregressive observations with missing data."
        L = self.lags
        T, D = data.shape
        K = self.K
        
    
        # 1. Compute AR means per discrete state
        mus = self._compute_mus(data, input, mask, tag)  # shape: K x T x D
        
       
        ll_init = np.column_stack([
            stats.multivariate_normal_logpdf(data[:L], mu[:L], Sigma)
            for mu, Sigma in zip(mus, self.Sigmas_init)
        ])
    
        ll_ar = np.column_stack([
            stats.multivariate_normal_logpdf(data[L:], mu[L:], Sigma)
            for mu, Sigma in zip(mus, self.Sigmas)
        ])
        
        # Choice likelihood
        if tag is not None:
            # decision = tag # expected to be 0 or 1
            # wT = 1 if decision == 0 else -1
            # H =np.array( [1,-1,0,0])
            # logP_dec = np.log(1/np.sqrt(2*np.pi*self.sigma_decision**2))-1/2*(wT-mus[:,-1] @ H)*1/self.sigma_decision**2*wT-mus[:,-1] @ H   
            # ll_dec = np.zeros((ll_ar.shape[0]-1,ll_ar.shape[1]))
            # ll_decision = np.row_stack((ll_dec,logP_dec))
            decision = tag
            wT = np.zeros((D,1))
            wT[decision] = 1
            H = np.zeros((D,D))
            H[0,0] = 1
            H[1,1] = 1
            logP_dec = np.array([np.squeeze(np.log(2*np.pi**(-D/2)*np.linalg.det(self.sigma_decision)) -\
                0.5*(wT- (H @ mu[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wT- (H @ mu[-1])[:,None])) \
                for mu in mus])[None,:]
            ll_dec = np.zeros((ll_ar.shape[0]-1,ll_ar.shape[1]))
            ll_decision = np.row_stack((ll_dec,logP_dec))
            # delta = mus[:,-1,0]-mus[:,-1,1]
            # z = delta / (np.sqrt(2) * self.sigma_decision)
            # prob = 0.5 * (1 + erf(z))
            # # Avoid log(0)
            # eps = 1e-12
            # prob = np.clip(prob, eps, 1 - eps)
            # ll_decision = np.log(prob) if decision == 0 else np.log(1 - prob)
            
            
            ll = np.row_stack((ll_init, ll_ar + ll_decision))  # shape (T, K)
        else:
            ll = np.row_stack((ll_init, ll_ar))
        
        return ll
    
    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        assert np.all(mask), "Cannot compute negative Hessian of autoregressive obsevations with missing data."
        assert self.lags == 1, "Does not compute negative Hessian of autoregressive observations with lags > 1"

        # initial distribution contributes a Gaussian term to first diagonal block
        J_ini = np.sum(Ez[0, :, None, None] * np.linalg.inv(self.Sigmas_init), axis=0)

        # first part is transition dynamics - goes to all terms except final one
        # E_q(z) x_{t} A_{z_t+1}.T Sigma_{z_t+1}^{-1} A_{z_t+1} x_{t}
        inv_Sigmas = np.linalg.inv(self.Sigmas)
        dynamics_terms = np.array([A.T@inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)]) # A^T Qinv A terms
        J_dyn_11 = np.sum(Ez[1:,:,None,None] * dynamics_terms[None,:], axis=1)

        # second part of diagonal blocks are inverse covariance matrices - goes to all but first time bin
        # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} x_{t+1}
        J_dyn_22 = np.sum(Ez[1:,:,None,None] * inv_Sigmas[None,:], axis=1)
        
        # # Add final J for decision
        Hs = np.zeros((inv_Sigmas.shape[0],data.shape[1],data.shape[1]))
        Hs[:,0,0] = 1
        Hs[:,1,1] = 1
        blocks = np.array([-1.0 * H @ np.linalg.inv(self.sigma_decision) @ H
                            for H in Hs])
        J_dec = np.sum(Ez[-1,:,None,None] * blocks, axis=0)
        J_dyn_22[-1] += -J_dec

        # lower diagonal blocks are (T-1,D,D):
        # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} A_{z_t+1} x_t
        off_diag_terms = np.array([inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)])
        J_dyn_21 = -1 * np.sum(Ez[1:,:,None,None] * off_diag_terms[None,:], axis=1)
        
        return J_ini, J_dyn_11, J_dyn_21, J_dyn_22
    
    def Rchoice_likelihood(self,x,q_Sigmas):
        # ll_decision = np.row_stack((ll_dec,logP_dec))
        wTR = np.array([0,1,0,0])[:,None]
        wTL = np.array([1,0,0,0])[:,None]
        H = np.zeros((self.D,self.D))
        H[0,0] = 1
        H[1,1] = 1
        
        P_R = 2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision) *\
            np.exp(-0.5*(wTR - (H @ x[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wTR - (H @ x[-1])[:,None]))
            
        P_L = 2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision) *\
            np.exp(-0.5*(wTL - (H @ x[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wTL - (H @ x[-1])[:,None]))
        
        prob = P_R/(P_R+P_L)
        return prob
        # q_stds = np.array([np.sqrt(np.diag(q_Sigma)) for q_Sigma in q_Sigmas])
        # delta = x[-1,0]-x[-1,1]
        # z = delta / (np.sqrt(q_stds[-1,0]*q_stds[-1,1]))
        # prob = 0.5 * (1 + erf(z))
        # # Avoid log(0)
        # eps = 1e-12
        # prob = np.clip(prob, eps, 1 - eps)
        # return 1-prob
    
    def Rchoice_likelihoods(self,xs,q_Sigmass):
        prob = np.zeros((len(xs),))
        for i,(x,q_Sigmas) in enumerate(zip(xs,q_Sigmass)):
            prob[i] = self.Rchoice_likelihood(x,q_Sigmas[2])
            
        return prob
    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask
    # def decision_to_latent(self, decision, sigma=0.1, target_value=1.0, baseline=0.0):
    #     """
    #     Convert binary decision into a latent target in R^2 space.
    
    #     Args:
    #         decision: int, 0 or 1
    #         sigma: how confident you are in this endpoint (smaller = stronger pull)
    #         target_value: the desired activation of the chosen latent dim
    #         baseline: the "null" activation of the unchosen dim
    
    #     Returns:
    #         x_T: np.array of shape (2,) — target latent mean at final time
    #     """
    #     x_T = np.full(self.D, baseline)
    #     x_T[decision] = target_value
    #     return x_T

    # def _compute_mus(self, data, input, mask, tag):
    #     K, M = self.K, self.M
    #     T, D = data.shape
    #     As, bs, Vs, mu0s = self.As, self.bs, self.Vs, self.mu_init
    #     backward_weight = np.eye(D) * 0.2  # Store outside if constant
    
    #     mus = []
    
    #     for k in range(K):
    #         A = As[k]
    #         b = bs[k]
    #         V = Vs[k]
    #         mu0 = mu0s[k]
    
    #         # Initial condition
    #         mus_k_init = np.tile(mu0, (self.lags, 1))
    
    #         # AR-driven means (forward pass)
    #         mus_k_ar = np.dot(input[self.lags:, :M], V.T)
    #         for l in range(self.lags):
    #             Al = A[:, l*D:(l + 1)*D]
    #             mus_k_ar += np.dot(data[self.lags - l - 1:-l - 1], Al.T)
    #         mus_k_ar += b
    
    #         # Stack initial and forward mean
    #         mus_k = np.vstack((mus_k_init, mus_k_ar))
    
    #         # # Final mean from decision
    #         # x_T = self.decision_to_latent(tag, sigma=0.1)  # shape (D,)
    #         # mus_k = np.concatenate([
    #         #     mus_k[:-1],
    #         #     x_T[None, :]  # overwrite final timestep
    #         # ], axis=0)
    
    #         # # Backward smoothing: no in-place mutation
    #         # for t in range(T-1, self.lags, -1):
    #         #     for l in range(self.lags):
    #         #         Al = A[:, l*D:(l + 1)*D]
    #         #         prev_t = t - l - 1
    #         #         if prev_t >= self.lags:
    #         #             forward_mean = np.dot(Al, mus_k[prev_t])
    #         #             residual = mus_k[t] - forward_mean
    #         #             correction = np.dot(Al.T, residual)
    #         #             corrected_prev = mus_k[prev_t] + backward_weight @ correction
    #         #             # Rebuild mus_k with corrected prev_t
    #         #             mus_k = np.concatenate([
    #         #                 mus_k[:prev_t],
    #         #                 corrected_prev[None, :],
    #         #                 mus_k[prev_t+1:]
    #         #             ], axis=0)
    
    #         mus.append(mus_k)
            
    #     # assert(False)
    
    #     mus = np.stack(mus, axis=0)
    #     return np.array(mus)
   

    
    def log_prior(self):
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum( -(alpha+1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)
        
class DecisionEmbeddedAccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, D_emb=None, D_acc=None,Mphys=2,learn_V=True):
        super().__init__(K, D, M)

        assert D_emb + D_acc == D, "accum + emb dims must equal total dims"
        
        self.D_acc = D_acc
        self.D_emb = D_emb
        self.D = D_acc + D_emb
        self.M = M
        self.K = K
        self.Mphys = Mphys
        
        self.dx = 0.25
        self.dt = 0.02
        self.epsilon = 10e-10;
        self.total_rate = 30;
        self.sigma_decision = 0.1*np.eye(self.D)

        # diagonal dynamics for each state
        # only learn accumulation dynamics for accumulation state
        self._a_diag = np.ones((D_acc,1))

        if D_emb == 1:
            self._a_emb = np.random.uniform(low=0.9,high=0.95,size=(D_emb,))*np.eye(D_emb) #@ random_rotation(D_emb)
        else:
            # a = np.random.uniform(low=0.9,high=0.95,size=(1,))
            # self._a_emb = scipy.linalg.block_diag(a,random_rotation(D_emb-1))
            self._a_emb = np.random.uniform(low=0.9,high=0.95,size=(D_emb,))*np.eye(D_emb) #@ random_rotation(D_emb)
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
        if D_acc > 1:
            # set input Accumulation params, one for each dimension
            # first D inputs are accumulated in different dimensions
            # rest of M-D inputs are applied to each dimension
            ### This is original, for first D_acc dims
            # self._betas = 0.1*np.ones(D_acc,)
            # r1 = self._betas*np.eye(D_acc, D_acc)
            # r2 = np.zeros((self.D_emb, M)) # is D_emb rows by input # of columns 
            # self.Vs[0] = np.vstack((r1, r2))
            ### This is new, for all dims
            self._betas = 0.1 * np.ones(M,)
            self._betas_emb = 0.1 * np.ones(D_emb*M,)
            r1 = self._betas*np.eye(D_acc, M)
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
        else:
            # set input Accumulation params, each Mphys inputs are accumulated on one
            # dimension in opposite directions
            # rest of M-D inputs are applied to each dimension
            self._betas = np.array([0.1,-0.1])*np.ones(Mphys,)
            self._betas_emb = 0.1 * np.ones(D_emb*M,)
            r1 = np.hstack((self._betas[None,:],np.zeros((1,M-Mphys))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)

        # self.Vs[0] = np.vstack((r1, r2))

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self._betas_emb,  self.accum_log_sigmasq, self._a_diag, self._a_emb
        return params

    @params.setter
    def params(self, value):
        self._betas, self._betas_emb, self.accum_log_sigmasq, self._a_diag, self._a_emb = value

        K, D, M, D_acc, D_emb = self.K, self.D, self.M, self.D_acc, \
            self.D_emb

        # Update V
        if D_acc > 1:
            r1 = self._betas*np.eye(D_acc, M)
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
        else:
            # set input Accumulation params, each Mphys inputs are accumulated on one
            # dimension in opposite directions
            # rest of M-D inputs are applied to each dimension
            r1 = np.hstack((self._betas,np.zeros((1,M-self.Mphys))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
       
       

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2

    def log_prior(self):
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum( -(alpha+1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass
    
    def decision_to_latent(self, decision, sigma=0.1, target_value=1.0, baseline=0.0):
        """
        Convert binary decision into a latent target in R^2 space.
    
        Args:
            decision: int, 0 or 1
            sigma: how confident you are in this endpoint (smaller = stronger pull)
            target_value: the desired activation of the chosen latent dim
            baseline: the "null" activation of the unchosen dim
    
        Returns:
            x_T: np.array of shape (2,) — target latent mean at final time
        """
        x_T = np.full(self.D, baseline)
        x_T[decision] = target_value
        return x_T

    
    
    
    # def _compute_mus(self, data, input, mask, tag):
    #     K, M = self.K, self.M
    #     T, D = data.shape
    #     As, bs, Vs, mu0s = self.As, self.bs, self.Vs, self.mu_init
    #     backward_weight = np.eye(D) * 0.8  # Store outside if constant
    
    #     mus = []
    
    #     for k in range(K):
    #         A = As[k]
    #         b = bs[k]
    #         V = Vs[k]
    #         mu0 = mu0s[k]
    
    #         # Initial condition
    #         mus_k_init = np.tile(mu0, (self.lags, 1))
    
    #         # AR-driven means (forward pass)
    #         mus_k_ar = np.dot(input[self.lags:, :M], V.T)
    #         for l in range(self.lags):
    #             Al = A[:, l*D:(l + 1)*D]
    #             mus_k_ar += np.dot(data[self.lags - l - 1:-l - 1], Al.T)
    #         mus_k_ar += b
    
    #         # Stack initial and forward mean
    #         mus_k = np.vstack((mus_k_init, mus_k_ar))
    
    #         # # Final mean from decision
    #         x_T = self.decision_to_latent(tag, sigma=0.1)  # shape (D,)
    #         mus_k = np.concatenate([
    #             mus_k[:-1],
    #             x_T[None, :]  # overwrite final timestep
    #         ], axis=0)
    
    #         # Backward smoothing: no in-place mutation
    #         for t in range(T-1, self.lags, -1):
    #             for l in range(self.lags):
    #                 Al = A[:, l*D:(l + 1)*D]
    #                 prev_t = t - l - 1
    #                 if prev_t >= self.lags:
    #                     forward_mean = np.dot(Al, mus_k[prev_t])
    #                     residual = mus_k[t] - forward_mean
    #                     correction = np.dot(Al.T, residual)
    #                     corrected_prev = mus_k[prev_t] + backward_weight @ correction
    #                     # Rebuild mus_k with corrected prev_t
    #                     mus_k = np.concatenate([
    #                         mus_k[:prev_t],
    #                         corrected_prev[None, :],
    #                         mus_k[prev_t+1:]
    #                     ], axis=0)
    
    #         mus.append(mus_k)
            
    
    #     mus = np.stack(mus, axis=0)
    #     return np.array(mus)
    
    
    
    def log_likelihoods(self, data, input, mask, tag=None):
        assert np.all(mask), "Cannot compute likelihood of autoregressive observations with missing data."
        L = self.lags
        T, D = data.shape
        K = self.K
        
    
        # 1. Compute AR means per discrete state
        mus = self._compute_mus(data, input, mask, tag)  # shape: K x T x D
        
       
        ll_init = np.column_stack([
            stats.multivariate_normal_logpdf(data[:L], mu[:L], Sigma)
            for mu, Sigma in zip(mus, self.Sigmas_init)
        ])
    
        ll_ar = np.column_stack([
            stats.multivariate_normal_logpdf(data[L:], mu[L:], Sigma)
            for mu, Sigma in zip(mus, self.Sigmas)
        ])
        
        # Choice likelihood
        if tag is not None:
            # decision = tag # expected to be 0 or 1
            # wT = 1 if decision == 0 else -1
            # H =np.array( [1,-1,0,0])
            # logP_dec = np.log(1/np.sqrt(2*np.pi*self.sigma_decision**2))-1/2*(wT-mus[:,-1] @ H)*1/self.sigma_decision**2*wT-mus[:,-1] @ H   
            # ll_dec = np.zeros((ll_ar.shape[0]-1,ll_ar.shape[1]))
            # ll_decision = np.row_stack((ll_dec,logP_dec))
            decision = tag
            wT = np.zeros((D,1))
            wT[decision] = 1
            H = np.zeros((D,D))
            H[0,0] = 1
            H[1,1] = 1
            logP_dec = np.array([np.squeeze(np.log(2*np.pi**(-D/2)*np.linalg.det(self.sigma_decision)) -\
                0.5*(wT- (H @ mu[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wT- (H @ mu[-1])[:,None])) \
                for mu in mus])[None,:]
            ll_dec = np.zeros((ll_ar.shape[0]-1,ll_ar.shape[1]))
            ll_decision = np.row_stack((ll_dec,logP_dec))
            # delta = mus[:,-1,0]-mus[:,-1,1]
            # z = delta / (np.sqrt(2) * self.sigma_decision)
            # prob = 0.5 * (1 + erf(z))
            # # Avoid log(0)
            # eps = 1e-12
            # prob = np.clip(prob, eps, 1 - eps)
            # ll_decision = np.log(prob) if decision == 0 else np.log(1 - prob)
            
            
            ll = np.row_stack((ll_init, ll_ar + ll_decision))  # shape (T, K)
        else:
            ll = np.row_stack((ll_init, ll_ar))
        
        return ll
    
    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        assert np.all(mask), "Cannot compute negative Hessian of autoregressive obsevations with missing data."
        assert self.lags == 1, "Does not compute negative Hessian of autoregressive observations with lags > 1"

        # initial distribution contributes a Gaussian term to first diagonal block
        J_ini = np.sum(Ez[0, :, None, None] * np.linalg.inv(self.Sigmas_init), axis=0)

        # first part is transition dynamics - goes to all terms except final one
        # E_q(z) x_{t} A_{z_t+1}.T Sigma_{z_t+1}^{-1} A_{z_t+1} x_{t}
        inv_Sigmas = np.linalg.inv(self.Sigmas)
        dynamics_terms = np.array([A.T@inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)]) # A^T Qinv A terms
        J_dyn_11 = np.sum(Ez[1:,:,None,None] * dynamics_terms[None,:], axis=1)

        # second part of diagonal blocks are inverse covariance matrices - goes to all but first time bin
        # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} x_{t+1}
        J_dyn_22 = np.sum(Ez[1:,:,None,None] * inv_Sigmas[None,:], axis=1)
        
        # # Add final J for decision
        Hs = np.zeros((inv_Sigmas.shape[0],data.shape[1],data.shape[1]))
        Hs[:,0,0] = 1
        Hs[:,1,1] = 1
        blocks = np.array([-1.0 * H @ np.linalg.inv(self.sigma_decision) @ H
                            for H in Hs])
        J_dec = np.sum(Ez[-1,:,None,None] * blocks, axis=0)
        J_dyn_22[-1] += -J_dec

        # lower diagonal blocks are (T-1,D,D):
        # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} A_{z_t+1} x_t
        off_diag_terms = np.array([inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)])
        J_dyn_21 = -1 * np.sum(Ez[1:,:,None,None] * off_diag_terms[None,:], axis=1)
        
        return J_ini, J_dyn_11, J_dyn_21, J_dyn_22
    
    def Rchoice_likelihood(self,x,q_Sigmas):
        # ll_decision = np.row_stack((ll_dec,logP_dec))
        wTR = np.array([0,1,0,0])[:,None]
        wTL = np.array([1,0,0,0])[:,None]
        H = np.zeros((self.D,self.D))
        H[0,0] = 1
        H[1,1] = 1
        
        P_R = 2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision) *\
            np.exp(-0.5*(wTR - (H @ x[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wTR - (H @ x[-1])[:,None]))
            
        P_L = 2*np.pi**(-self.D/2)*np.linalg.det(self.sigma_decision) *\
            np.exp(-0.5*(wTL - (H @ x[-1])[:,None]).T @ np.linalg.inv(self.sigma_decision) @ (wTL - (H @ x[-1])[:,None]))
        
        prob = P_R/(P_R+P_L)
        return prob
        # q_stds = np.array([np.sqrt(np.diag(q_Sigma)) for q_Sigma in q_Sigmas])
        # delta = x[-1,0]-x[-1,1]
        # z = delta / (np.sqrt(q_stds[-1,0]*q_stds[-1,1]))
        # prob = 0.5 * (1 + erf(z))
        # # Avoid log(0)
        # eps = 1e-12
        # prob = np.clip(prob, eps, 1 - eps)
        # return 1-prob
    
    def Rchoice_likelihoods(self,xs,q_Sigmass):
        prob = np.zeros((len(xs),))
        for i,(x,q_Sigmas) in enumerate(zip(xs,q_Sigmass)):
            prob[i] = self.Rchoice_likelihood(x,q_Sigmas[2])
            
        return prob
    # def log_likelihoods(self, data, input, mask, tag):
    #     L = self.lags
    #     T, D = data.shape
    #     K = self.K
       
    #     sigma_d = 0.1
    #     #T, D = input.shape[0] + 1, A.shape[0]
    #     H = np.eye( D)
       
        
    #     zT = np.zeros((1,D))
    #     zT[0,tag] = 1
    #     R = sigma_d**2 * np.eye(D)
    
    #     xs_smooth = []
    #     for k in range(K):
    #         x_pred = [self.mu_init[k]]
    #         P_pred = [np.eye(D)]
    #         x_filt = [self.mu_init[k]]
    #         P_filt = [np.eye(D)]
    #         x_smooth = [self.mu_init[k]]
    #         P_smooth = [np.eye(D)]
    #         # Initial mus
            
    #         A = self.As[k]
    #         B = self.Vs[k]
    #         Q = self.Sigmas[k]
    #         # Forward filter
    #         for t in range(1, T):
    #             x_minus = A @ data[t-1] + B @ input[t-1]
    #             P_minus = A @ P_filt[t-1] @ A.T + Q
        
    #             if t == T-1:
    #                 S = H @ P_minus @ H.T + R
    #                 Kd = P_minus @ H.T @ np.linalg.inv(S)
    #                 y = zT - H @ x_minus
    #                 x_upd = np.squeeze(x_minus + (Kd @ y.T).T)
    #                 P_upd = P_minus - Kd @ H @ P_minus
    #             else:
    #                 x_upd, P_upd = x_minus, P_minus
        
    #             x_pred.append(x_minus)
    #             P_pred.append(P_minus)
    #             x_filt.append(x_upd)
    #             P_filt.append(P_upd)

    #         # RTS backward smoother
        
    #         x_smooth = x_filt
    #         P_smooth = P_filt
        
    #         for t in reversed(range(T-1)):
    #             J = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t+1])
    #             x_smooth[t] = x_filt[t] + J @ (x_smooth[t+1] - x_pred[t+1])
    #             P_smooth[t] = P_filt[t] + J @ (P_smooth[t+1] - P_pred[t+1]) @ J.T
            
    #         x_smooth_arr = np.stack([row for row in x_smooth], axis=0)  # shape (T, D)
    #         xs_smooth.append(x_smooth_arr)
        
    #     xs_smooth_arr = np.stack([np.stack(row, axis=0) for row in xs_smooth], axis=0)  # shape (K, T, D)
        
    #     ll_init = np.column_stack([
    #         stats.multivariate_normal_logpdf(data[:L], mu[:L], Sigma)
    #         for mu, Sigma in zip(xs_smooth_arr, self.Sigmas_init)
    #     ])
    
    #     ll_ar = np.column_stack([
    #         stats.multivariate_normal_logpdf(data[L:], mu[L:], Sigma)
    #         for mu, Sigma in zip(xs_smooth_arr, self.Sigmas)
    #     ])
            
    #     ll = np.row_stack((ll_init, ll_ar))
        
    #     return ll
       
    
    # def log_decision_likelihood(self,x):
    #     log_decision_likelihood = - (1 / (2 * sigma**2)) * (x - 1)**2 - logsumexp([
    #     - (1 / (2 * sigma**2)) * (mu_T[0] - 1)**2,
    #     - (1 / (2 * sigma**2)) * (mu_T[1] - 1)**2
    #     ])

    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)
        
class UrgencyGatingObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, D_emb=None, D_acc=None, Mphys=2,learn_V=True):
        super().__init__(K, D, M)

        assert D_emb + D_acc == D, "accum + emb dims must equal total dims"
        
        self.D_acc = D_acc
        self.D_emb = D_emb
        self.D = D_acc + D_emb
        self.M = M
        self.K = K
        self.Mphys = Mphys

        # diagonal dynamics for each state
        # only learn accumulation dynamics for accumulation state
        self._a_diag = np.ones((D_acc,1))
        # Make random dynamics matrix for embedded dimensions
        
        # Diag = np.diag(np.random.uniform(low=0.7,high=0.8,size=(D_emb,)))# * \
        #                #(np.array([1 if x > 0.5  else -1 for x in np.random.random(Demb)])))
        
        # # Make random eigenvector matrix
        # Q = np.random.uniform(low=-1,high=1,size=(D_emb,D_emb)) 
        # self._a_emb = (Q @ Diag @ np.linalg.inv(Q)).flatten()
        #self._a_emb = random_rotation(D_emb)*np.random.uniform(low=0.7,high=0.8)
        
        # (M @ A)v = (M @ \lambda)v
        if D_emb == 1:
            self._a_emb = np.random.uniform(low=0.9,high=0.95,size=(D_emb,))*np.eye(D_emb) #@ random_rotation(D_emb)
        else:
            # a = np.random.uniform(low=0.9,high=0.95,size=(1,))
            # self._a_emb = scipy.linalg.block_diag(a,random_rotation(D_emb-1))
            self._a_emb = np.random.uniform(low=0.9,high=0.95,size=(D_emb,))*np.eye(D_emb) #@ random_rotation(D_emb)
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
        if D_acc > 1:
            # set input Accumulation params, one for each dimension
            # first D inputs are accumulated in different dimensions
            # rest of M-D inputs are applied to each dimension
            ### This is original, for first D_acc dims
            # self._betas = 0.1*np.ones(D_acc,)
            # r1 = self._betas*np.eye(D_acc, D_acc)
            # r2 = np.zeros((self.D_emb, M)) # is D_emb rows by input # of columns 
            # self.Vs[0] = np.vstack((r1, r2))
            ### This is new, for all dims
            self._betas = 0.1 * np.ones(M,)
            self._betas_emb = 0.1 * np.ones(D_emb*M,)
            r1 = self._betas*np.eye(D_acc, M)
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
        else:
            # set input Accumulation params, each Mphys inputs are accumulated on one
            # dimension in opposite directions
            # rest of M-D inputs are applied to each dimension
            self._betas = np.array([0.1,-0.1])*np.ones(Mphys,)
            self._betas_emb = 0.1 * np.ones(D_emb*M,)
            r1 = np.hstack((self._betas[None,:],np.zeros((1,M-Mphys))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)

        # self.Vs[0] = np.vstack((r1, r2))

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self._betas_emb,  self.accum_log_sigmasq, self._a_diag, self._a_emb
        return params

    @params.setter
    def params(self, value):
        self._betas, self._betas_emb, self.accum_log_sigmasq, self._a_diag, self._a_emb = value

        K, D, M, D_acc, D_emb = self.K, self.D, self.M, self.D_acc, \
            self.D_emb

        # Update V
        ### This is original, for first D_acc dims
        # k1 = np.vstack((self._betas * np.eye(self.D_acc), np.zeros((self.D_emb,self.D_acc)))) # state K = 0
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        ### This is new, for all dims
        # k1 = self._betas * np.eye(self.D, self.M)
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        
        # Update V
        if D_acc > 1:
            r1 = self._betas*np.eye(D_acc, M)
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
        else:
            # set input Accumulation params, each Mphys inputs are accumulated on one
            # dimension in opposite directions
            # rest of M-D inputs are applied to each dimension
            r1 = np.hstack((self._betas,np.zeros((1,M-self.Mphys))))
            r2 = np.reshape(self._betas_emb,(D_emb,M))
            k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
            #k2 = np.tile(r2[None,:,:],(K,1,1))
            k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
            self.Vs = np.concatenate([k1,k2],axis=1)
       
        

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        # mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = self._a_diag*mask1 + mask2
        
        # top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        # bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        # A_acc = np.vstack((top_rows,bottom_rows))
        # mask1 = np.vstack( (A_acc[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = mask1 + mask2
        
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        #bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),np.reshape(self._a_emb,(D_emb,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
    def Rchoice_likelihood(self,x,q_Sigmas):
        q_stds = np.array([np.sqrt(np.diag(q_Sigma)) for q_Sigma in q_Sigmas])
        delta = x[-1,0]-x[-1,1]
        z = delta / (np.sqrt(q_stds[-1,0]*q_stds[-1,1]))
        prob = 0.5 * (1 + erf(z))
        # Avoid log(0)
        eps = 1e-12
        prob = np.clip(prob, eps, 1 - eps)
        return 1-prob
    
    def Rchoice_likelihoods(self,xs,q_Sigmass):
        prob = np.zeros((len(xs),))
        for i,(x,q_Sigmas) in enumerate(zip(xs,q_Sigmass)):
            prob[i] = self.Rchoice_likelihood(x,q_Sigmas[2])
            
        return prob
    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask

    def log_prior(self):
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum( -(alpha+1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass
    
    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)
        
class DAWAccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, learn_A=True, learn_V=False, learn_sig_init=False):
        super(DAWAccumulationObservations, self).__init__(K, D, M)

        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        self.learn_A = learn_A
        self._a_diag = np.ones((D,1))
        if self.learn_A:
            mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
            mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
            self._As = self._a_diag*mask1 + mask2
        else:
            self._As = np.tile(np.eye(D),(K,1,1))

        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        self._betas = 0.1*np.ones(D,)
        self.learn_V = learn_V
        self._V = 0.1*np.ones((D, M-D)) # additional covariates, if they exist
        self.Vs[0] = np.hstack((self._betas*np.eye(D,D), self._V))
        for d in range(1,K):
            self.Vs[d] *= np.zeros((D,M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self.learn_sig_init = learn_sig_init 
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self.accum_log_sigmasq
        params = params + (self._a_diag,) if self.learn_A else params
        params = params + (self._V,) if self.learn_V else params
        params = params + (self.accum_log_sigmasq_init) if self.learn_sig_init else params
        return params

    @params.setter
    def params(self, value):
        self._betas, self.accum_log_sigmasq = value[:2]
        if self.learn_A:
            self._a_diag = value[2]
        if self.learn_V:
            self._V = value[-1]
        #TODO fix above
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = value[-1]

        K, D, M = self.K, self.D, self.M

        # update V
        mask0 = np.hstack((np.eye(D), np.ones((D,M-D)))) # state K = 0
        mask01 = np.concatenate([np.zeros((2,D,D)), np.ones((2,D,M-D))],2)
        mask = np.vstack((mask0[None,:,:], mask01))

        # self.Vs = self._betas * mask
        self.Vs = np.hstack((np.diag(self._betas), self._V)) * mask

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        # self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        self._As = self._a_diag*mask1 + mask2
        
class DecisionAccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, learn_A=True, learn_V=False, learn_sig_init=False):
        super(DAWAccumulationObservations, self).__init__(K, D, M)

        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        self.learn_A = learn_A
        self._a_diag = np.ones((D,1))
        if self.learn_A:
            mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
            mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
            self._As = self._a_diag*mask1 + mask2
        else:
            self._As = np.tile(np.eye(D),(K,1,1))

        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        self._betas = 0.1*np.ones(D,)
        self.learn_V = learn_V
        self._V = 0.1*np.ones((D, M-D)) # additional covariates, if they exist
        self.Vs[0] = np.hstack((self._betas*np.eye(D,D), self._V))
        for d in range(1,K):
            self.Vs[d] *= np.zeros((D,M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self.learn_sig_init = learn_sig_init 
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self.accum_log_sigmasq
        params = params + (self._a_diag,) if self.learn_A else params
        params = params + (self._V,) if self.learn_V else params
        params = params + (self.accum_log_sigmasq_init) if self.learn_sig_init else params
        return params

    @params.setter
    def params(self, value):
        self._betas, self.accum_log_sigmasq = value[:2]
        if self.learn_A:
            self._a_diag = value[2]
        if self.learn_V:
            self._V = value[-1]
        #TODO fix above
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = value[-1]

        K, D, M = self.K, self.D, self.M

        # update V
        mask0 = np.hstack((np.eye(D), np.ones((D,M-D)))) # state K = 0
        mask01 = np.concatenate([np.zeros((2,D,D)), np.ones((2,D,M-D))],2)
        mask = np.vstack((mask0[None,:,:], mask01))

        # self.Vs = self._betas * mask
        self.Vs = np.hstack((np.diag(self._betas), self._V)) * mask

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        # self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        self._As = self._a_diag*mask1 + mask2

class Accumulation1DObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, Mphys=2,lags=1, learn_A=True, learn_V=False, learn_betas=True, learn_sig_init=False):
        super(Accumulation1DObservations, self).__init__(K, D, M)
        D=1
        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        self.learn_A = learn_A
        self._a_diag = np.ones((D,1))
        if self.learn_A:
            mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
            mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
            self._As = self._a_diag*mask1 + mask2
        else:
            self._As = np.tile(np.eye(D),(K,1,1))

        # set input Accumulation params, each Mphys inputs are accumulated on one
        # dimension in opposite directions
        # rest of M-D inputs are applied to each dimension
        self.learn_betas = learn_betas
        self._betas = np.array([0.1,-0.1])*np.ones(Mphys,)
        self.learn_V = learn_V
        self._V = 0.1*np.ones((D, M-Mphys)) # additional covariates, if they exist
        self.Vs[0] = np.hstack((self._betas*np.eye(D,D), self._V))
        for d in range(1,K):
            if M > D+1:
                self.Vs[d] = np.hstack((np.zeros((D,M-(D+1))), self._V)) 
            else:
                self.Vs[d] *= np.zeros((D,M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self.learn_sig_init = learn_sig_init 
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))
        self.Mphys = Mphys
    
    @property
    def _learn_V(self):
        return self.learn_V

    @_learn_V.setter
    def _learn_V(self, value):
        self.learn_V = value
        
    @property
    def _learn_betas(self):
        return self.learn_betas

    @_learn_betas.setter
    def _learn_betas(self, value):
        assert type(value) == bool
        self.learn_betas = value
        
    @property
    def _learn_A(self):
        return self.learn_A

    @_learn_A.setter
    def _learn_A(self, value):
        assert type(value) == bool
        self.learn_A = value
        
    @property
    def params(self):
        params = self._betas, self.accum_log_sigmasq
        params = params + (self._a_diag,) if self.learn_A else params
        params = params + (self._V,) if self.learn_V else params
        params = params + (self.accum_log_sigmasq_init) if self.learn_sig_init else params
        return params

    @params.setter
    def params(self, value):
        if self.learn_betas:
            self._betas, self.accum_log_sigmasq = value[:2]
        if self.learn_A:
            self._a_diag = value[2]
        if self.learn_V:
            self._V = value[-1]
        #TODO fix above
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = value[-1]

        K, D, M = self.K, self.D, self.M

        # update V
        mask0 = np.ones((1,D,M)) # state K = 0
        mask01 = np.concatenate([np.zeros((K-1,D,self.Mphys)), np.ones((K-1,D,M-(D+1)))],2)
        mask = np.vstack((mask0, mask01))

        # self.Vs = self._betas * mask
        self.Vs = np.hstack((self._betas[None,:], self._V)) * mask

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        # self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        self._As = self._a_diag*mask1 + mask2

    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask

    def log_prior(self):
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum( -(alpha+1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)