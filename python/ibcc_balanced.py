'''
@author: Edwin Simpson
'''
import logging
import numpy as np
from scipy.optimize import fmin_bfgs, fmin
from ibcc import IBCC
from scipy.stats import expon

class BalancedIBCC(IBCC):

    def init_lnkappa(self):
        self.nu = np.ones(self.nclasses) * 1000000
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

    def logmodeljoint(self, alphamean):
        '''
        Weight the marginal likelihood/model evidence by the hyperprior. Unnormalised posterior over the hyperparams.
        '''
        marglh = self.logmarginallikelihood()
        # assume an exponential distribution over the hyperparameters
        palpha0 = np.sum(expon.logpdf(self.alpha0, scale=alphamean))
        return marglh + palpha0

    def optimize_ibcc(self, alphamean, crowdlabels, train_t, testidxs, maxiter=100):
        ''' 
        Assuming exponential distribution over the hyperparameters, we find the MAP values. The combiner object is updated
        to contain the optimal values, searched for using BFGS.
        '''
        def map_target_function(hyperparams, combiner, alphamean):
            logging.debug("Hyperparms: %s" % str(hyperparams))
            if np.any(np.isnan(hyperparams)) or np.any(hyperparams <= 0):
                return np.inf
            initialK = len(hyperparams) / (combiner.nclasses * combiner.nscores)
            alpha0_flat = hyperparams[0:combiner.nclasses * combiner.nscores * initialK]
            combiner.alpha0 = alpha0_flat.reshape((combiner.nclasses, combiner.nscores, initialK))
            # run the combiner
            combiner.combine_classifications([], [], preprocess=False)
            # evaluate the result
            neg_lml = -combiner.logmodeljoint(alphamean)
            logging.debug("Negative log joint probability of the model & obs: %f" % neg_lml)
            return neg_lml
        # set up combiner object
        self.preprocess_data(crowdlabels, train_t, testidxs)
        hyperparams0 = alphamean.flatten()
        opt_hyperparams = fmin(map_target_function, hyperparams0, args=(self, alphamean), maxiter=maxiter)
        logging.debug("Optimal hyperparams: %s" % str(opt_hyperparams))
        map_target_function(opt_hyperparams, self, alphamean)
