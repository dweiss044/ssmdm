# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:28:04 2026

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

class DDMEmbeddedTransitions(RecurrentTransitions):
    def __init__(self, K, D, M=0, scale=200, D_acc=None, D_emb=None):
        assert K == D_acc+2
        assert D_acc == 1
        assert D_acc + D_emb == D
        # assert M == 1
        super(DDMEmbeddedTransitions, self).__init__(K, D, M)

        # DDM has one accumulation state and boundary states at +/- 1
        self.log_Ps = -scale*np.ones((K,K)) + np.diag(np.concatenate(([scale],2.0*scale*np.ones(K-1))))
        self.Ws = np.zeros((K, M))
        #self.Rs = np.array([0, scale, -scale]).reshape((3, 1))
        R1 = np.array([0, scale, -scale]).reshape((K, D_acc))
        R2 = np.zeros((K, D_emb))
        self.Rs = np.hstack((R1, R2))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass
        # scale = value
        # K, M = self.K, self.M
        # self.log_Ps = -scale*np.ones((K,K)) + np.diag(np.concatenate(([scale],2.0*scale*np.ones(K-1))))
        # self.Ws = np.zeros((K, M))
        # self.Rs = np.array([0, scale, -scale]).reshape((3, 1))
        # pass

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass
    
class DAWAccumulationRaceTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=100, D_acc=None, D_emb=None):
        assert K == D+1
        #assert K == D_acc+1
        # assert D_acc >= 1
        # assert D_acc + D_emb == D
        super(DAWAccumulationRaceTransitions, self).__init__(K, D, M)

        # "Race" transitions with D+1 states
        # Transition to state d when x_d > 1.0
        # State 0 is the accumulation state
        # scale determines sharpness of the threshold
        # Transitions out of boundary states occur w/ very low probability
        # top_row = np.concatenate(([0.0],-scale*np.ones(D)))
        # rest_rows = np.hstack((-scale*np.ones((D,1)),-scale*np.ones((D,D)) + np.diag(2.0*scale*np.ones(D))))
        # self.log_Ps = np.vstack((top_row,rest_rows))
        self.Ws = np.zeros((K,M))
        self.Rs = np.vstack((np.zeros(D),scale*np.eye(D)))
        # R1 = np.vstack((np.zeros(D_acc),scale*np.eye(D_acc)))
        # R2 = np.zeros((K, D_emb))
        # self.Rs = np.hstack((R1, R2))
        self.r = np.concatenate(([0],-scale*np.ones(K-1)))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass
    
    
    # def log_transition_matrices(self, data, input, mask, tag):
    #     # Clip latent for numerically stable transitions
    #     data = np.clip(data, -0.95, 0.95)
    #     log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
    #     log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]     # past observations
    #     log_Ps = log_Ps + self.r                                       # bias
    #     log_Ps = np.tile(log_Ps, (1, self.K, 1))                       # expand
    #     return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize
    
    # def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
    #     # Clip latent for numerically stable transitions
    #     data = np.clip(data, -0.95, 0.95)
    #     # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
    #     v = np.dot(input[1:], self.Ws.T) + np.dot(data[:-1], self.Rs.T) + self.r
    #     shifted_exp = np.exp(v - np.max(v,axis=1,keepdims=True))
    #     vtilde = shifted_exp / np.sum(shifted_exp,axis=1,keepdims=True) # normalized probabilities
    #     Rv = vtilde@self.Rs
    #     hess = np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
    #            + np.einsum('ti, tj -> tij', Rv, Rv)
    #     return -1 * hess

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass
    
class EmbeddedAccumulationRaceSoftTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=200, D_acc=None, D_emb=None):
        assert K == D_acc+1
        assert D_acc >= 1
        assert D_acc + D_emb == D
        super(EmbeddedAccumulationRaceSoftTransitions, self).__init__(K, D, M)

        # "Race" transitions with D+1 states
        # Transition to state d when x_d > 1.0
        # State 0 is the accumulation state
        # scale determines sharpness of the threshold
        # Transitions out of boundary states occur w/ very low probability
        #top_row = np.concatenate(([0.0],-scale*np.ones(D_acc)))
        #rest_rows = np.hstack((-scale*np.ones((D_acc,1)),-scale*np.ones((D_acc,D_acc)) + np.diag(2.0*scale*np.ones(D_acc))))
        #self.log_Ps = np.vstack((top_row,rest_rows))
        self.Ws = np.zeros((K,M))
        R1 = np.vstack((np.zeros(D_acc),scale*np.eye(D_acc)))
        R2 = np.zeros((K, D_emb))
        self.Rs = np.hstack((R1, R2))
        self.r = np.concatenate(([0],-scale*np.ones(D_acc)))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass
    
# class AccumulationRaceSoftTransitions(RecurrentOnlyTransitions):
#     def __init__(self, K, D, M=0, scale=100):
#         assert K == D+1
#         assert D >= 1
#         super(AccumulationRaceSoftTransitions, self).__init__(K, D, M)

#         # Like Race Transitions but soft boundaries
#         # Transitions depend on previous x only
#         # Transition to state d when x_d > 1.0
#         self.Ws = np.zeros((K,M))
#         self.Rs = np.vstack((np.zeros(D),scale*np.eye(D)))
#         self.r = np.concatenate(([0],-scale*np.ones(D)))

#     @property
#     def params(self):
#         return ()

#     @params.setter
#     def params(self, value):
#         pass

#     def initialize(self, datas, inputs=None, masks=None, tags=None):
#         pass

#     def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
#         pass


class AccumulationRaceSoft1DTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=100):
        assert K == D+2
        assert D >= 1
        super(AccumulationRaceSoft1DTransitions, self).__init__(K, D, M)

        # Like Race Transitions but soft boundaries
        # Transitions depend on previous x only
        # Transition to state d when x_d > 1.0
        self.Ws = np.zeros((K,M))
        self.Rs = np.vstack((np.zeros(D),scale*np.eye(D),-scale*np.eye(D)))
        self.r = np.concatenate(([0],-scale*np.ones(D),-scale*np.ones(D)))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass