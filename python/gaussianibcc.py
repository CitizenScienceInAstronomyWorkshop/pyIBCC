'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix
from scipy.special import psi
from scipy.stats import norm, gamma
from ibcc import IBCC

class GaussianIBCC(IBCC):
       
    # hyperparameters
    lamb0 = None  # prior mean coefficient
    m0 = None  # prior mean
    gam_alpha0 = None  # prior precision shape
    gam_beta0 = None  # prior precision rate
    
    # posterior hyperparameters,
    lamb = None  # JxK matrix
    m = None  # JxK matrix
    gam_alpha = None  # JxK matrix
    gam_beta = None  # JxK matrix

    #likelihood parameters
    prec = None  # expected precision matrix for the base classifiers, NxK
    mu = None  # expected mean matrix for the base classifiers, NxK
       
    # indicator vector showing which agents should be modelled as Gaussians
    agent_gauss_indicator = []
    # table containing the continuous valued scores provided by the agents being modelled by Gaussians
    Cgauss = None
    # number of Gaussian-modelled agents
    Kgauss = 1

    def expec_lnPi(self): # update the likelihood parameters      
        Nj = np.zeros((self.nclasses, self.Kgaus))
        xbarjk = np.zeros((self.nclasses, self.Kgaus))
        Sjk = np.zeros((self.nclasses, self.Kgaus))
        for j in range(self.nclasses):
            Nj[j, :] = np.sum(self.E_t[self.observed_idxs, j] * ~np.isnan(self.Cgauss), axis=0)
            xbarjk[j, :] = np.nansum(self.E_t[self.observed_idxs, j] * self.Cgauss, axis=0) / Nj[j]
            Sjk[j, :] = np.nansum(self.E_t[self.observed_idxs, j] * (self.Cgauss - xbarjk[j, :]) ** 2)  # equivalent to the pseudocounts for standard IBCC
        self.lamb = self.lamb0 + Nj
        self.m = (self.lamb0*self.m0 +  Nj*xbarjk) / self.lamb
        self.gam_alpha = self.gam_alpha0 + Nj
        self.gam_beta = 1 / (1 / self.gam_beta0 + Nj * Sjk + ((self.lamb0 * Nj) / (self.lamb)) * (xbarjk - self.m0) ** 2)
        # expected parameters
        self.mu = self.m
        self.prec = self.gam_alpha / self.gam_beta
        
    def lnjoint(self):
        lnjoint = np.zeros((self.N, self.nclasses))
        for j in range(self.nclasses):
            # square difference from mean, normalised by precision
            mu_deviation = self.gam_alpha[j, :] * self.gam_beta[j, :] * (self.Cgauss - self.m) ** 2 + 1.0 / self.lamb[j, :]
            lnPrec = psi(self.gam_alpha[j, :]) - np.log(self.gam_beta[j, :])
            lnjoint[:, j] = np.nansum((lnPrec - self.ln(2 * np.pi) - mu_deviation) / 2, axis=1) + self.lnkappa[j]
        return lnjoint

    def lowerbound(self, lnjoint):
        #probability of these targets is 1 as they are training labels
        lnpCT = self.post_lnjoint_ct(lnjoint)                    
        lnpPi = np.sum(norm.pdf(self.mu, loc=self.m0, scale=1 / (self.lamb0 * self.prec)) \
                                                    *  gamma.pdf(self.prec, shape=self.gam_alpha0, scale=1 / self.gam_beta0))
        lnpKappa = self.post_lnkappa()
        EEnergy = lnpCT + lnpPi + lnpKappa
        
        lnqT = self.q_ln_t()
        lnqPi = np.sum(norm.pdf(self.mu, loc=self.m, scale=1 / (self.lamb * self.prec)) \
                                                    *  gamma.pdf(self.prec, shape=self.gam_alpha, scale=1 / self.gam_beta))
        lnqKappa = self.q_lnkappa()
        H = - lnqT - lnqPi - lnqKappa
        L = EEnergy + H
        #logging.debug('EEnergy ' + str(EEnergy) + ', H ' + str(H))
        return L
        
    def preprocess_crowdlabels(self):
        # ensure we don't have a matrix by mistake
        if not isinstance(self.crowdlabels, np.ndarray):
            self.crowdlabels = np.array(self.crowdlabels)
        Cgauss = {}
        if self.table_format_flag:
            Cgauss = self.crowdlabels
            self.observed_idxs = np.argwhere(~np.isnan(np.sum(self.crowdlabels, axis=1))).reshape(-1)
        else:
            data = self.crowdlabels[:, 2]  # the continuous values (NaN where unavailable)
            rows = self.crowdlabels[:, 1]  # the object ids
            cols = self.crowdlabels[:, 0]  # the agent ids
            Cgauss = coo_matrix((data, (rows, cols)), shape=(self.N, self.Kgaus))
            self.observed_idxs = np.unique(self.crowdlabels[:, 1])
        self.Cgauss = Cgauss

    def init_K(self):
        if self.table_format_flag :
            newK = self.crowdlabels.shape[1]
        else:
            newK = np.max(self.crowdlabels[:, 0]) + 1  # +1 since we start from 0
        if self.K + self.Kgauss <= newK:
            if self.agent_gauss_indicator:
                self.K = np.sum(~self.agent_gauss_indicator)
                # the remainder are Gaussian, even if we now have more agents than in the specified indicator vector
                self.Kgauss = newK - self.K
                if len(self.agent_gauss_indicator)<newK:
                    newindices = newK - len(self.agent_gauss_indicator)
                    self.agent_gauss_indicator = np.concatenate((self.agent_gauss_indicator, np.ones(newindices)))
            else:
                self.Kgauss = newK
            self.init_params()

    def init_lnPi(self):
        if self.gam_alpha != [] and self.gam_alpha.shape[1] == self.Kgauss:
            return  # already set up
        if len(self.lamb0.shape) < 2:
            self.lamb0 = np.array(self.lamb0[:, np.newaxis], dtype=np.float64)
            self.lamb0 = np.repeat(self.lamb0, self.Kgauss, axis=1)
        if len(self.m0.shape) < 2:
            self.m0 = np.array(self.m0[:, np.newaxis], dtype=np.float64)
            self.m0 = np.repeat(self.m0, self.Kgauss, axis=1)
        if len(self.gam_alpha0.shape) < 2:
            self.gam_alpha0 = np.array(self.gam_alpha0[:, np.newaxis], dtype=np.float64)
            self.gam_alpha0 = np.repeat(self.gam_alpha0, self.Kgauss, axis=1)
        if len(self.gam_beta0.shape) < 2:
            self.gam_beta0 = np.array(self.gam_beta0[:, np.newaxis], dtype=np.float64)
            self.gam_beta0 = np.repeat(self.gam_beta0, self.Kgauss, axis=1)

        oldK = self.gam_alpha0.shape[1]
        if oldK < self.Kgauss:
            nnew = self.Kgauss - oldK
            lamb0new = self.gam_alpha0[:, 0]
            lamb0new = lamb0new[:, np.newaxis]
            lamb0new = np.repeat(lamb0new, nnew, axis=1)
            self.lamb0 = np.concatenate((self.lamb0, lamb0new), axis=1)
            m0new = self.m0[:, 0]
            m0new = m0new[:, np.newaxis]
            m0new = np.repeat(m0new, nnew, axis=1)
            self.m0 = np.concatenate((self.m0, m0new), axis=1)
            gam_alpha0new = self.gam_alpha0[:, 0]
            gam_alpha0new = gam_alpha0new[:, np.newaxis]
            gam_alpha0new = np.repeat(gam_alpha0new, nnew, axis=1)
            self.gam_alpha0 = np.concatenate((self.gam_alpha0, gam_alpha0new), axis=1)
            gam_beta0new = self.gam_beta0[:, 0]
            gam_beta0new = gam_beta0new[:, np.newaxis]
            gam_beta0new = np.repeat(gam_beta0new, nnew, axis=1)
            self.gam_beta0 = np.concatenate((self.gam_beta0, gam_beta0new), axis=1)
        self.lamb = deepcopy(np.float64(self.lamb0))
        self.m = deepcopy(np.float64(self.m0))
        self.gam_alpha = deepcopy(np.float64(self.gam_alpha0))
        self.gam_beta = deepcopy(np.float64(self.gam_beta0))
        # initialise parameter expectations
        self.mu = self.m
        self.prec = self.gam_alpha / self.gam_beta

    def __init__(self, nclasses=2, nscores=2, gam_alpha0=None, nu0=None, K=1, table_format=False, dh=None, agent_gauss_indicator=None):
        super(GaussianIBCC, self).__init__(nclasses=2, nscores=2, gam_alpha0=None, nu0=None, K=1, table_format=False, dh=None)
        if agent_gauss_indicator != None:
            self.agent_gauss_indicator = agent_gauss_indicator
            self.discreteIBCC = IBCC(nclasses=2, nscores=2, gam_alpha0=None, nu0=None, K=1, table_format=False, dh=None)
