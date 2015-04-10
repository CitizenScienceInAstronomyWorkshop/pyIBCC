'''
@author: Edwin Simpson
'''
import logging
import numpy as np
from scipy.optimize import fmin
from ibcc import IBCC
from scipy.stats import gamma

class BalancedIBCC(IBCC):

    def init_lnkappa(self):
        self.nu = np.ones(self.nclasses) * 1000000 # use some default value that gives equal proportions
        proportion = np.log(1.0 / float(self.nclasses))
        self.lnkappa = np.ones(self.nclasses) * proportion
# Expectations: methods for calculating expectations with respect to parameters for the VB algorithm ---------------
    def expec_lnkappa(self):
        self.nu = np.ones(self.nclasses) * 1000000 
        proportion = np.log(1.0 / float(self.nclasses))
        self.lnkappa = np.ones(self.nclasses) * proportion

    def post_lnkappa(self):
        lnpKappa = 0
        return lnpKappa
        
    def q_lnkappa(self):
        lnqKappa = 0
        return lnqKappa

    def ln_modelprior(self):
        #Check and initialise the hyper-hyper-parameters if necessary
        if self.gam_scale_alpha==[]:
            self.gam_shape_alpha = np.float(self.gam_shape_alpha)
            # if the scale was not set, assume current values of alpha0 are the means given by the hyper-prior
            self.gam_scale_alpha = self.alpha0/self.gam_shape_alpha
        #Gamma distribution over each value. Set the parameters of the gammas.
        p_alpha0 = gamma.logpdf(self.alpha0, self.gam_shape_alpha, scale=self.gam_scale_alpha)
        return np.sum(p_alpha0)