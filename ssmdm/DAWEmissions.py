# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:38:30 2026

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


class DAWAccumulationPoissonEmissions(PoissonEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0,Mphys=0,Mopto=0):
        super(DAWAccumulationPoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs[:,:,:Mphys] *= 0
        self.Fs[:,:,Mphys:] *= 5e-3

   # Construct an emissions model
    @property
    def params(self):
        return self._Cs, self.ds

    @params.setter
    def params(self, value):
        self._Cs, self.ds = value

    def invert(self, data, input=None, mask=None, tag=None, clip=np.array([0.0,1.0]),augment=False):
#         yhat = self.link(np.clip(data, .1, np.inf))
        if self.bin_size < 1:
            yhat = smooth(data,20)
        else:
            yhat = smooth(data,5)
        # Only invert observable, non-augmented dimensions
        if augment:
            self.D = int(self.D/2)
            Cs = copy.deepcopy(self.Cs)
            self.Cs = self.Cs[:,:,:self.D]
        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        xhat = smooth(xhat,10)
        
        # Add back in augmented dimensions
        if augment:
            self.D = int(self.D*2)
            self.Cs = Cs
            xhat = np.hstack((xhat,np.zeros_like(xhat)))
        
        # DAW Commented
        if self.bin_size < 1:
            xhat = np.clip(xhat, -0.95, 0.95)

        # in all models, x starts in between boundaries at [-1,1]
        if np.abs(xhat[0]).any()>1.0:
                xhat[0] = 0.05*npr.randn(1,self.D)
        return xhat
    
    # def log_likelihoods(self, data, input, mask, tag, x):
    #     assert data.dtype == int
    #     lambdas = self.mean(self.forward(np.clip(x,-15,15), input, tag))
    #     mask = np.ones_like(data, dtype=bool) if mask is None else mask
    #     lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
    #     return np.sum(lls * mask[:, None, :], axis=2)
    
    def initialize(self, base_model, datas, inputs=None, masks=None, tags=None,
                   emission_optimizer="bfgs", num_optimizer_iters=1000):
        print("Initializing Emissions parameters...")

        if self.D == 1 and base_model.transitions.__class__.__name__ == "DDMTransitions":
        # if self.D == 0:
            d_init = np.mean([y[0:3] for y in datas],axis=(0,1))
            u_sum = np.array([np.sum(u) for u in inputs])
            y_end = np.array([y[-3:] for y in datas])
            u_l, u_u = np.percentile(u_sum, [20,80]) # use 20th and 80th percentile input
            y_U = y_end[np.where(u_sum>=u_u)]
            y_L = y_end[np.where(u_sum<=u_l)]
            C_init = (1.0/2.0)*np.mean((np.mean(y_U,axis=0) - np.mean(y_L,axis=0)),axis=0)
            self.Cs = C_init.reshape([1,self.N,self.D]) / self.bin_size
            self.ds = d_init.reshape([1,self.N]) / self.bin_size

        else:
            datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

            Td = sum([data.shape[0] for data in datas])
            xs = [base_model.sample(T=data.shape[0],input=input)[1] for data, input in zip(datas, inputs)]
            def _objective(params, itr):
                self.params = params
                # self.Cs = params
                obj = 0
                obj += self.log_prior()
                for data, input, mask, tag, x in \
                    zip(datas, inputs, masks, tags, xs):
                    obj += np.sum(self.log_likelihoods(data, input, mask, tag, x))
                return -obj / Td

            # Optimize emissions log-likelihood
            optimizer = dict(bfgs=bfgs, lbfgs=lbfgs)[emission_optimizer]
            self.params = \
                optimizer(_objective,
                          self.params,
                          num_iters=num_optimizer_iters,
                          full_output=False)
                
class EmbeddedCascadedPoissonEmissions(PoissonEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0, cascades=1, D_acc=None, D_emb=None):
        self.cascades = cascades
        self.D_acc = D_acc
        self.D_emb = D_emb
        self.D_acc_total = D_acc * cascades
        
        # Verify total dimension matches cascade + embedded
        assert self.D_acc_total + D_emb == D, "accum (scaled by cascades) + emb dims must equal total dims D"
        
        super().__init__(N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size)
        
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0
        
        # Enforce zero weights for intermediate/early cascade stages immediately
        self._apply_cascade_mask()

    def _apply_cascade_mask(self):
        """Zero out columns of Cs corresponding to intermediate cascade buffer states."""
        if hasattr(self, '_Cs') and self._Cs is not None:
            mask = np.zeros(self.D, dtype=float)
            # 1. Only allow the final stage of each accumulator cascade to affect emissions
            for d in range(self.D_acc):
                final_stage_index = (d + 1) * self.cascades - 1
                mask[final_stage_index] = 1
            # 2. Allow all embedded dimensions to affect emissions normally
            mask[self.D_acc_total:] = 1
            
            # Apply mask along the latent dimension axis (last axis of Cs)
            self._Cs *= mask

    # Construct an emissions model
    @property
    def params(self):
        return self._Cs, self.ds

    @params.setter
    def params(self, value):
        self._Cs, self.ds = value
        # Ensure intermediate cascade stages stay zeroed out whenever parameters update
        self._apply_cascade_mask()

    def invert(self, data, input=None, mask=None, tag=None, clip=np.array([0.0, 1.0]), augment=False):
        if self.bin_size < 1:
            yhat = smooth(data, 20)
        else:
            yhat = smooth(data, 5)
            
        # Handle augmented/observable dimensions adjustment based on cascade structure
        if augment:
            original_D = self.D
            self.D = self.D_acc + self.D_emb
            Cs = copy.deepcopy(self.Cs)
            self.Cs = self.Cs[:, :, :self.D]
            
        xhat_observable = self.link(np.clip(yhat, 0.01, np.inf))
        xhat_observable = self._invert(xhat_observable, input=input, mask=mask, tag=tag)
        xhat_observable = smooth(xhat_observable, 10)
        
        if augment:
            self.D = original_D
            self.Cs = Cs
            padding_dims = self.D - xhat_observable.shape[1]
            xhat = np.hstack((xhat_observable, np.zeros((xhat_observable.shape[0], padding_dims))))
        else:
            xhat = xhat_observable

        if self.bin_size < 1:
            xhat = np.clip(xhat, -0.95, 0.95)

        # In all models, x starts in between boundaries at [-1, 1]
        if np.abs(xhat[0]).any() > 1.0:
            xhat[0] = 0.05 * npr.randn(1, self.D)
            
        return xhat
    
    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        # Invert with the average emission parameters
        C = np.mean(self.Cs, axis=0)
        F = np.mean(self.Fs, axis=0)
        d = np.mean(self.ds, axis=0)
        C_pseudoinv = np.linalg.pinv(C).T
        #C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        if input is not None:
            bias = input.dot(F.T) + d
        else:
            bias = d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)
    
    def initialize(self, base_model, datas, inputs=None, masks=None, tags=None,
                    emission_optimizer="bfgs", num_optimizer_iters=1000):
        print("Initializing Emissions parameters...")

        if self.D == 1 and base_model.transitions.__class__.__name__ == "DDMTransitions":
            d_init = np.mean([y[0:3] for y in datas], axis=(0, 1))
            u_sum = np.array([np.sum(u) for u in inputs])
            y_end = np.array([y[-3:] for y in datas])
            u_l, u_u = np.percentile(u_sum, [20, 80])
            y_U = y_end[np.where(u_sum >= u_u)]
            y_L = y_end[np.where(u_sum <= u_l)]
            C_init = (1.0 / 2.0) * np.mean((np.mean(y_U, axis=0) - np.mean(y_L, axis=0)), axis=0)
            self.Cs = C_init.reshape([1, self.N, self.D]) / self.bin_size
            self.ds = d_init.reshape([1, self.N]) / self.bin_size

        else:
            datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

            Td = sum([data.shape[0] for data in datas])
            xs = [base_model.sample(T=data.shape[0], input=input)[1] for data, input in zip(datas, inputs)]
            
            def _objective(params, itr):
                self.params = params
                obj = 0
                obj += self.log_prior()
                for data, input, mask, tag, x in zip(datas, inputs, masks, tags, xs):
                    obj += np.sum(self.log_likelihoods(data, input, mask, tag, x))
                return -obj / Td

            optimizer = dict(bfgs=bfgs, lbfgs=lbfgs)[emission_optimizer]
            self.params = optimizer(_objective,
                                    self.params,
                                    num_iters=num_optimizer_iters,
                                    full_output=False)