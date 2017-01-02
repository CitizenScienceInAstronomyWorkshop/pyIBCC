'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import psi, gammaln, digamma
from ibccdata import DataHandler
from scipy.optimize import fmin, fmin_cobyla
from scipy.stats import gamma

from ibcc import IBCC
from sklearn.mixture import BayesianGaussianMixture

class CBCC(IBCC):
      
# Model parameters and hyper-parameters -----------------------------------------------------------------------------

    conc_prior = 1 # concentration hyperparameter
    nclusters = 100 

# Initialisation ---------------------------------------------------------------------------------------------------

    def init_weights(self):
        self.expec_weights()
    
    def init_responsibilities(self):
        self.r = 1.0 / self.nclusters + np.zeros((self.K, self.nclusters))
        self.logr = np.log(self.r)        
        
    def init_t(self):
        self.init_responsibilities()
        self.init_weights()
        super(CBCC, self).init_t()
        
    def init_lnPi(self):
        '''
        Always creates new self.alpha and self.lnPi objects and calculates self.alpha and self.lnPi values according to 
        either the prior, or where available, values of self.E_t from previous runs.
        '''
        self.alpha0 = self.alpha0.astype(float)
        # if we specify different alpha0 for some agents, we need to do so for all K agents. The last agent passed in 
        # will be duplicated for any missing agents.
        if np.any(self.clusteridxs_alpha0): # map from diags list of cluster IDs
            if not np.any(self.alpha0_cluster):
                self.alpha0_cluster = self.alpha0
                self.alpha0_length = self.alpha0_cluster.shape[2]
            self.alpha0 = self.alpha0_cluster[:, :, self.clusteridxs_alpha0]        
        elif len(self.alpha0.shape) == 3 and self.alpha0.shape[2] < self.nclusters:
            # We have a new dataset with more agents than before -- create more priors.
            nnew = self.nclusters - self.alpha0.shape[2]
            alpha0new = self.alpha0[:, :, 0]
            alpha0new = alpha0new[:, :, np.newaxis]
            alpha0new = np.repeat(alpha0new, nnew, axis=2)
            self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
        elif len(self.alpha0.shape)==2:
            self.alpha0  = self.alpha0[:,:,np.newaxis]
        # Make sure self.alpha is the right size as well. Values of self.alpha not important as we recalculate below
        self.alpha0 = self.alpha0[:, :, :self.nclusters] # make this the right size if there are fewer classifiers than expected
        self.alpha = np.zeros((self.nclasses, self.nscores, self.nclusters), dtype=np.float) + self.alpha0
        self.lnPi = np.zeros((self.nclasses, self.nscores, self.K)) 
        self.expec_lnPi(posterior=False) # calculate alpha from the initial/prior values only in the first iteration

# Posterior Updates to Hyperparameters --------------------------------------------------------------------------------

    def expec_weights(self):
        #nk is the sum of responsibilities for the agents, shape (nclusters,)
        #nk[::-1] reverses the order
        #np.cumsum(nk[::-1]) gives the total weights for clusters with ID >= i for each index i in the array
        # (np.cumsum(nk[::-1])[-2::-1], 0) flips the order back and skips the first value so that we have the total 
        #weight for clusterIDs > i
        #np.hstack((blah..., 0)) appends 0 to the end since zero weight for cluster IDs > the last cluster
        # resulting weight_concentration_ has two values, effectively parameters to a beta distribution where [0] is 
        # the probability of this cluster and 
        
        nk = np.sum(self.r, axis=0)
        
        weight_concentration_ = (1. + nk, (self.conc_prior + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))))
        self.weight_conc = np.array(weight_concentration_)
        # estimate_log_weights
        
        digamma_sum = digamma(weight_concentration_[0] + weight_concentration_[1])
        digamma_a = digamma(weight_concentration_[0])
        
        logp_current_vs_subsequent_clusters = digamma_a - digamma_sum
        
        digamma_b = digamma(weight_concentration_[1])
        logp_subsequent_vs_current = digamma_b - digamma_sum
        logp_subsequent = np.cumsum(logp_subsequent_vs_current)[:-1]
        logp_current_or_subsequent = np.hstack((0, logp_subsequent))  
        
        self.Elogp_clusters = np.array((logp_current_vs_subsequent_clusters, logp_subsequent_vs_current))
        self.logw = logp_current_vs_subsequent_clusters + logp_current_or_subsequent

    def expec_t(self):
        super(CBCC, self).expec_t()

    def expec_responsibilities(self):
        # for each cluster value, compute the log likelihoods of the data. This will be a sum  
        # over lnPi columns for the observed labels multiplied by E_t and summed over all classes and all data points
        # to get logp(C^{(k)} | cluster_k = cluster)        
        loglikelihoods = np.zeros((self.K, self.nclusters))
        
        self.cluster_lnPi = np.zeros((self.nclasses, self.nscores, self.nclusters))         
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for s in range(self.nscores):
            self.cluster_lnPi[:, s, :] = digamma(self.alpha[:, s, :]) - psiSumAlpha
        
        for cl in range(self.nclusters):
            for j in range(self.nclasses):
                data = []
                for l in range(self.nscores):
                    if self.table_format_flag:
                        data_l = self.C[l] * self.cluster_lnPi[j, l, cl]
                    else:   
                        data_l = self.C[l].multiply(self.cluster_lnPi[j, l, cl])                        
                    data = data_l if data==[] else data+data_l
                loglikelihoods[:, cl:cl+1] += data.T.dot(self.E_t[:, j][:, np.newaxis])
        
        logweights = self.logw[np.newaxis, :]
        
        weighted_log_prob = loglikelihoods + logweights
        log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob), axis=1))
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        self.lnr = log_resp
        self.r = np.exp(self.lnr)

    def post_Alpha(self):  # Posterior Hyperparams
        # Save the counts from the training data so we only recalculate the test data on every iteration
        if not len(self.alpha_tr):
            self.alpha_tr = np.zeros(self.alpha.shape)
            if self.Ntrain:
                for j in range(self.nclasses):
                    for l in range(self.nscores):
                        for cl in range(self.nclusters):
                            Tj = self.E_t[self.trainidxs, j].reshape((self.Ntrain, 1))
                            if self.table_format_flag:
                                self.alpha_tr[j,l,cl] = np.sum(                                
                                   (self.C[l][self.trainidxs,:] * self.r[:, cl][np.newaxis, :]).T.dot(Tj).reshape(-1) )
                            else:
                                self.alpha_tr[j,l,cl] = np.sum( 
                                   self.C[l][self.trainidxs,:].multiply(self.r[:, cl][np.newaxis, :]).T.dot(Tj).reshape(-1) )
                            
            self.alpha_tr += self.alpha0
        # Add the counts from the test data
        for j in range(self.nclasses):
            for l in range(self.nscores):
                Tj = self.E_t[self.testidxs, j].reshape((self.Ntest, 1))
                for cl in range(self.nclusters):
                    if self.table_format_flag:
                        counts = (self.Ctest[l] * self.r[:, cl][np.newaxis, :]).T.dot(Tj).reshape(-1)
                    else:
                        counts = (self.Ctest[l].multiply(self.r[:, cl][np.newaxis, :])).T.dot(Tj).reshape(-1)
                        
                self.alpha[j, l, cl] = self.alpha_tr[j, l, cl] + np.sum(counts)

    def expec_lnPi(self, posterior=True):
        self.expec_responsibilities()
        self.expec_weights()
        
        # check if E_t has been initialised. Only update alpha if it has. Otherwise E[lnPi] is given by the prior
        if np.any(self.E_t) and posterior:
            self.post_Alpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for j in range(self.nclasses):
            for s in range(self.nscores): 
                self.lnPi[:, s, :] = (psi(self.alpha[:, s, :]) - psiSumAlpha)[np.newaxis, :].dot(self.r.T)
 
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def post_lnpi(self):
        x = np.sum((self.alpha0-1) * self.cluster_lnPi,1)
        z = gammaln(np.sum(self.alpha0,1)) - np.sum(gammaln(self.alpha0),1)
        
        #cluster weights
        weight_prior_params = np.array([1.0, self.conc_prior])[:, np.newaxis]
        w_x = np.sum((weight_prior_params - 1) * self.Elogp_clusters)
        w_z = np.sum(gammaln(np.sum(weight_prior_params, axis=0)) - np.sum(gammaln(weight_prior_params), axis=0))  
        
        # responsibilities
        logp_membership = np.sum(self.r * self.logw[np.newaxis, :])
        
        return np.sum(x+z) + w_x + w_z + logp_membership
                    
    def q_lnPi(self):
        x = np.sum((self.alpha-1) * self.cluster_lnPi,1)
        z = gammaln(np.sum(self.alpha,1)) - np.sum(gammaln(self.alpha),1)

        #cluster weights        
        w_x = np.sum((self.weight_conc - 1) * self.Elogp_clusters)
        w_z = np.sum(gammaln(np.sum(self.weight_conc, axis=0)) - np.sum(gammaln(self.weight_conc), axis=0))  
        
        #responsibilities
        logq_membership = np.sum(self.r * self.lnr)
        
        return np.sum(x+z) + w_x + w_z + logq_membership
    
# Loader and Runner helper functions -------------------------------------------------------------------------------
def load_combiner(config_file, ibcc_class=None):
    dh = DataHandler()
    dh.loadData(config_file)
    if ibcc_class==None:
        combiner = IBCC(dh=dh)
    else:
        combiner = ibcc_class(dh=dh)
    return combiner, dh    
    
def load_and_run_ibcc(configFile, ibcc_class=None, optimise_hyperparams=False):
    combiner, dh = load_combiner(configFile, ibcc_class=CBCC)
    #combine labels
    combiner.verbose = True
    combiner.uselowerbound = True
    pT = combiner.combine_classifications(dh.crowdlabels, dh.goldlabels, optimise_hyperparams=optimise_hyperparams, 
                                          table_format=dh.table_format)

    if dh.output_file is not None:
        dh.save_targets(pT)

    dh.save_pi(combiner.alpha, combiner.nclasses, combiner.nscores)
    dh.save_hyperparams(combiner.alpha, combiner.nu)
    pT = dh.map_predictions_to_original_IDs(pT)
    return pT, combiner
    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    load_and_run_ibcc(configFile)
