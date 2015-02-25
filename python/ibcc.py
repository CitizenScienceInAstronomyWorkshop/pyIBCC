'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix
from scipy.special import psi, gammaln
from ibccdata import DataHandler
from scipy.stats import expon
from scipy.optimize import fmin_bfgs, fmin

class IBCC(object):
# Configuration for variational Bayes (VB) algorithm for approximate inference -------------------------------------
    # determine convergence by calculating lower bound? Quicker to set=False so we check convergence of target variables
    uselowerbound = True
    min_iterations = 1
    max_iterations = 500
    conv_threshold = 0.0001
    keeprunning = True  # Set to false to cause the combine_classifications method to exit without completing; useful
    # if another thread is checking whether IBCC is taking too long.
# Dataset attributes -----------------------------------------------------------------------------------------------
    table_format_flag = False  # crowdlabels as a full KxN table? If false, use a sparse 3-column list, where 1st
    # column=classifier ID, 2nd column = obj ID, 3rd column = score.
    nclasses = None
    nscores = None
    alpha0 = None
    nu0 = None
    K = None
    N = 0 #number of objects
    # Sparsity handling
    observed_idxs = []
    full_train_t = []
    full_N = 0
    # The data from the crowd
    crowdlabels = None
    train_t = None
    # Indices into the current dataset
    trainidxs = None
    testidxs = None
    conf_mat_ind = []  # indicies into the confusion matrices corresponding to the current set of crowdlabels
# Model parameters and hyperparameters -----------------------------------------------------------------------------
    #The model
    lnkappa = []
    nu = []
    lnPi = []
    alpha = []
    E_t = []
    lnpCT = None  # the joint likelihood (interim value)
# Initialisation ---------------------------------------------------------------------------------------------------
    def __init__(self, nclasses=2, nscores=2, alpha0=None, nu0=None, K=1, table_format=False, dh=None):
        if dh != None:
            self.nclasses = dh.nclasses
            self.nscores = len(dh.scores)
            self.alpha0 = dh.alpha0
            self.nu0 = dh.nu0
            self.K = dh.K
            table_format = dh.table_format
        else:
            self.nclasses = nclasses
            self.nscores = nscores
            self.alpha0 = alpha0
            self.nu0 = nu0
            self.K = K
        if table_format:
            self.table_format_flag = True
        else:
            self.table_format_flag = None

    def init_params(self):
        logging.debug('Initialising parameters...')
        logging.debug('Alpha0: ' + str(self.alpha0))
        self.init_lnPi()

        logging.debug('Nu0: ' + str(self.nu0))
        self.init_lnkappa()

    def init_lnkappa(self):
        if self.nu != []:
            return
        # Ensure we have float arrays so we can do division with these parameters properly
        self.nu0 = self.nu0.astype(float)
        self.nu = deepcopy(np.float64(self.nu0))
        sumNu = np.sum(self.nu)
        self.lnkappa = psi(self.nu) - psi(sumNu)

    def init_lnPi(self):
        if self.alpha != [] and self.alpha.shape[2] == self.K:
            return
        self.alpha0 = self.alpha0.astype(float)
        if len(self.alpha0.shape) < 3:
            self.alpha0 = np.array(self.alpha0[:, :, np.newaxis], dtype=np.float64)
            self.alpha0 = np.repeat(self.alpha0, self.K, axis=2)
        oldK = self.alpha0.shape[2]
        if oldK < self.K:
            nnew = self.K - oldK
            alpha0new = self.alpha0[:, :, 0]
            alpha0new = alpha0new[:, :, np.newaxis]
            alpha0new = np.repeat(alpha0new, nnew, axis=2)
            self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
        # Ensure we have a float so we can do arithmetic with alpha0 and alpha
        self.alpha = deepcopy(np.float64(self.alpha0))  # np.float64(self.alpha0[:,:,np.newaxis])

        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        self.lnPi = np.zeros((self.nclasses, self.nscores, self.K))
        for s in range(self.nscores):        
            self.lnPi[:, s, :] = psi(self.alpha[:, s, :]) - psiSumAlpha

    def init_t(self):
        kappa = self.nu / np.sum(self.nu, axis=0)
        if len(self.E_t) > 0:
            if self.sparse:
                oldE_t = self.E_t_sparse
            else:
                oldE_t = self.E_t
            Nold = oldE_t.shape[0]
            if Nold > self.N:
                Nold = self.N
        else:
            oldE_t = []
        self.E_t = np.zeros((self.N, self.nclasses)) + kappa
        if len(oldE_t) > 0:
            self.E_t[0:Nold, :] = oldE_t[0:Nold, :]
        uncert_trainidxs = self.trainidxs.copy()  # look for labels that are not discrete values of valid classes
        for j in range(self.nclasses):
            # training labels
            row = np.zeros((1, self.nclasses))
            row[0, j] = 1
            jidxs = self.train_t == j
            uncert_trainidxs = uncert_trainidxs - jidxs
            self.E_t[jidxs, :] = row
        # deal with uncertain training idxs
        for j in range(self.nclasses):
            # values a fraction above class j
            partly_j_idxs = np.bitwise_and(self.train_t[uncert_trainidxs] > j, self.train_t[uncert_trainidxs] < j + 1)
            partly_j_idxs = uncert_trainidxs[partly_j_idxs]
            self.E_t[partly_j_idxs, j] = 1 - self.train_t[partly_j_idxs] + j
            # values a fraction below class j
            partly_j_idxs = np.bitwise_and(self.train_t[uncert_trainidxs] < j, self.train_t[uncert_trainidxs] > j - 1)
            partly_j_idxs = uncert_trainidxs[partly_j_idxs]
            self.E_t[partly_j_idxs, j] = self.train_t[partly_j_idxs] - (j - 1)
        if self.sparse:
            self.E_t_sparse = self.E_t  # current working version is sparse

# Data preprocessing and helper functions --------------------------------------------------------------------------
    def crowdtable_to_sparselist(self, table):
        valididxs = np.bitwise_and(np.isfinite(table), table >= 0)
        agents = np.arange(table.shape[1])[np.newaxis, :]
        objects = np.arange(table.shape[0])[:, np.newaxis]
        agents = np.tile(agents, (self.N, 1))
        objects = np.tile(objects, (1, self.K))
        agents = agents[valididxs][:, np.newaxis]
        objects = objects[valididxs][:, np.newaxis]
        scores = table[valididxs][:, np.newaxis]
        return np.concatenate((agents, objects, scores), axis=1)
        
    def desparsify_crowdlabels(self, crowdlabels):
        if crowdlabels.shape[1]!=3:
            self.table_format_flag = True
            # First, record which objects were actually observed.
            self.observed_idxs = np.argwhere(np.nansum(crowdlabels, axis=1) >= 0).reshape(-1)
            if crowdlabels.shape[0] > len(self.observed_idxs):
                self.sparse = True
                self.full_N = crowdlabels.shape[0]
                # cut out the unobserved data points. We'll put them back in at the end of the classification procedure.
                crowdlabels = crowdlabels[self.observed_idxs, :]
        else:
            self.observed_idxs, mappedidxs = np.unique(crowdlabels[:, 1], return_inverse=True)
            self.full_N = np.max(crowdlabels[:,1])
            if self.full_N > len(self.observed_idxs):
                self.sparse = True
                # map the IDs so we skip unobserved data points. We'll map back at the end of the classification procedure.
                crowdlabels[:, 1] = mappedidxs
        return crowdlabels

    def preprocess_data(self, crowdlabels, train_t=None, testidxs=None):
        crowdlabels = self.desparsify_crowdlabels(crowdlabels)
        if train_t != None and self.sparse:
            self.full_train_t = train_t
            if self.full_N < len(self.full_train_t):
                self.full_N = len(self.full_train_t)
            train_t = train_t[self.observed_idxs]
        # Find out how much training data and how many total data points we have
        len_t = 0  # length of the training vector
        if train_t != None:
            len_t = train_t.shape[0]
        len_c = crowdlabels.shape[0]  # length of the crowdlabels
        if not self.table_format_flag:
            len_c = np.max(crowdlabels[:, 1])
        # How many data points in total?
        if len_c > len_t:
            self.N = len_c
        else:
            self.N = len_t
        # Make sure that train_t is the right size in case we have passed in a training set for the first idxs
        if train_t == None:
            self.train_t = np.zeros(self.N) - 1
        elif train_t.shape[0] < self.N:
            extension = np.zeros(self.N - train_t.shape[0]) - 1
            self.train_t = np.concatenate((train_t, extension))
        else:
            self.train_t = train_t
        # record the test and train idxs
        self.trainidxs = self.train_t > -1
        if testidxs != None:
            self.testidxs = testidxs[self.observed_idxs]
        else:
            self.testidxs = np.bitwise_or(np.isnan(self.train_t), self.train_t < 0)
        # Now preprocess the crowdlabels.
        #Check that we have the right number of agents/base classifiers, K
        if self.table_format_flag :
            newK = crowdlabels.shape[1]
        else:
            newK = np.max(crowdlabels[:, 0]) + 1  # +1 since we start from 0
        if self.K <= newK or self.nu == []:  # data shape has changed or not initialised yet
            self.K = newK
            self.init_params() 
        C = {}
        if self.table_format_flag:            
            for l in range(self.nscores):
                Cl = np.zeros(crowdlabels.shape)
                Cl[crowdlabels == l] = 1
                C[l] = Cl
        else:            
            for l in range(self.nscores):
                lIdxs = np.where(crowdlabels[:, 2] == l)[0]
                data = np.array(np.ones((len(lIdxs),1))).reshape(-1)
                rows = np.array(crowdlabels[lIdxs, 1]).reshape(-1)
                cols = np.array(crowdlabels[lIdxs, 0]).reshape(-1)
                Cl = coo_matrix((data,(rows,cols)), shape=(self.N, self.K))
                C[l] = Cl
        # Set and reset object properties for the new dataset
        self.C = C
        self.crowdlabels = crowdlabels
        self.lnpCT = np.zeros((self.N, self.nclasses))
        self.conf_mat_ind = []
        if self.table_format_flag:
            agentIdx = np.tile(np.arange(self.K), (self.N, 1))
            self.lnPi_table_all = np.zeros((self.N, self.K))
            self.lnPi_table_test = np.zeros((np.sum(self.testidxs), self.K))
            #pre-compute the indices into the pi arrays
            self.all_crowdlabel_idxs = np.bitwise_and(np.isfinite(crowdlabels), crowdlabels >= 0)
            self.all_crowd_labels = crowdlabels[self.all_crowdlabel_idxs].astype(int)
            self.all_agent_idxs = agentIdx[self.all_crowdlabel_idxs].astype(int)
            #repeat for test labels only
            crowdlabels = crowdlabels[self.testidxs, :]
            self.test_crowdlabel_idxs = np.bitwise_and(np.isfinite(crowdlabels), crowdlabels >= 0)
            self.test_crowd_labels = crowdlabels[self.test_crowdlabel_idxs].astype(int)
            self.test_agent_idxs = agentIdx[self.test_crowdlabel_idxs].astype(int)
# Run the inference algorithm --------------------------------------------------------------------------------------
    def combine_classifications(self, crowdlabels, train_t=None, testidxs=None, preprocess=True):
        if preprocess:
            self.preprocess_data(crowdlabels, train_t, testidxs)
        self.init_t()
        logging.info('IBCC: combining to predict ' + str(np.sum(self.testidxs)) + " test data points...")
        oldL = -np.inf
        converged = False
        self.nIts = 0 #object state so we can check it later
        while not converged and self.keeprunning:
            oldET = self.E_t.copy()
            #update targets
            self.expec_t()
            #update params
            self.expec_lnkappa()
            self.expec_lnPi()
            #check convergence        
            if self.uselowerbound:
                L = self.lowerbound()
                logging.debug('Lower bound: ' + str(L) + ', increased by ' + str(L-oldL))
                change = L-oldL                
                oldL = L
            else:
                change = np.max(np.sum(np.absolute(oldET - self.E_t)))
            if (self.nIts>=self.max_iterations or change<self.conv_threshold) and self.nIts>self.min_iterations:
                converged = True
            self.nIts+=1
            if change<0:
                logging.warning('IBCC iteration ' + str(self.nIts) + ' absolute change was ' + str(change) + '. Possible bug or rounding error?')            
            else:
                logging.debug('IBCC iteration ' + str(self.nIts) + ' absolute change was ' + str(change))
        logging.info('IBCC finished in ' + str(self.nIts) + ' iterations (max iterations allowed = ' + str(self.max_iterations) + ').')
        # Convert back to original idxs in case of unobserved IDs
        if self.sparse:
            E_t_full = np.zeros((self.full_N,self.nclasses))
            E_t_full[:] = (np.exp(self.lnkappa) / np.sum(np.exp(self.lnkappa))).reshape((1, self.nclasses))
            E_t_full[self.observed_idxs,:] = self.E_t
            self.E_t_sparse = self.E_t  # save the sparse version
            self.E_t = E_t_full        
        return self.E_t
# Posterior Updates to Hyperparameters -----------------------------------------------------------------------------
    def post_Alpha(self):  # Posterior Hyperparams
        for j in range(self.nclasses):
            for l in range(self.nscores):
                Tj = self.E_t[:, j].reshape((self.N, 1))
                counts = self.C[l].T.dot(Tj).reshape(-1)
                self.alpha[j, l, :] = self.alpha0[j, l, :] + counts
# Expectations: methods for calculating expectations with respect to parameters for the VB algorithm ---------------
    def expec_lnkappa(self):
        sumET = np.sum(self.E_t, 0)
        for j in range(self.nclasses):
            self.nu[j] = self.nu0[j] + sumET[j]
        self.lnkappa = psi(self.nu) - psi(np.sum(self.nu))

    def expec_lnPi(self):
        self.post_Alpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for s in range(self.nscores):        
            self.lnPi[:, s, :] = psi(self.alpha[:, s, :]) - psiSumAlpha

    def expec_t(self):
        lnjoint = self.lnjoint()
        if not self.uselowerbound:
            lnjoint = lnjoint[self.testidxs, :]
        # ensure that the values are not too small
        largest = np.max(lnjoint, 1)[:, np.newaxis]
        joint = lnjoint - largest
        joint = np.exp(joint)
        norma = np.sum(joint, axis=1)[:, np.newaxis]
        pT = joint / norma
        if self.uselowerbound:
            self.E_t[self.testidxs] = pT[self.testidxs, :]
        else:
            self.E_t[self.testidxs, :] = pT
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def lnjoint(self, alldata=False):
        # Checking type. Subclasses that need other types can override this log joint implementation
        if self.crowdlabels.dtype != np.int:
            logging.warning("Converting input data matrix to integers.")
            self.crowdlabels = self.crowdlabels.astype(int)
        if self.table_format_flag:
            self.lnjoint_table(alldata)
        else:
            self.lnjoint_sparselist()
        return self.lnpCT

    def lnjoint_table(self, alldata=False):
        '''
        For use when crowdsourced data was supplied as a table
        '''
        if self.uselowerbound or alldata:
            idxs = np.ones(self.N, dtype=np.bool)
            for j in range(self.nclasses):
                data = self.lnPi[j, self.all_crowd_labels, self.all_agent_idxs]
                self.lnPi_table_all[self.all_crowdlabel_idxs] = data
                self.lnpCT[idxs, j] = np.sum(self.lnPi_table_all, 1) + self.lnkappa[j]
        else:  # no need to calculate in full
            idxs = self.testidxs
            for j in range(self.nclasses):
                data = self.lnPi[j, self.test_crowd_labels, self.test_agent_idxs]
                self.lnPi_table_test[self.test_crowdlabel_idxs] = data
                self.lnpCT[idxs, j] = np.sum(self.lnPi_table_test, 1) + self.lnkappa[j]
        
    def lnjoint_sparselist(self):
        '''
        For when crowdsourced data was supplied in three columns, for agent ID, data point ID, and classification score.
        '''
        if self.conf_mat_ind == []:
            crowdlabels = self.crowdlabels.astype(int)
            logging.info("Using only discrete labels at the moment.")
            self.conf_mat_ind = np.ravel_multi_index((crowdlabels[:, 2], crowdlabels[:, 0]), dims=(self.nscores, self.K))
        for j in range(self.nclasses):
            weights = self.lnPi[j, :].ravel()[self.conf_mat_ind]
            self.lnpCT[:, j] = np.bincount(self.crowdlabels[:, 1], weights=weights, minlength=self.N) + self.lnkappa[j]

    def post_lnkappa(self):
        lnpKappa = gammaln(np.sum(self.nu0)) - np.sum(gammaln(self.nu0)) \
                    + sum((self.nu0 - 1) * self.lnkappa)
        return lnpKappa
        
    def q_lnkappa(self):
        lnqKappa = gammaln(np.sum(self.nu)) - np.sum(gammaln(self.nu)) \
                        + np.sum((self.nu - 1) * self.lnkappa)
        return lnqKappa

    def q_ln_t(self):
        ET = self.E_t[self.E_t != 0]
        return np.sum(ET * np.log(ET))
    
    def logmarginallikelihood(self):
        if self.sparse:
            lnpCT = np.sum(self.lnjoint(True) * self.E_t_sparse)
            nmissing = self.full_N - self.N
            lnpCT_missing = np.sum(self.lnkappa * self.nu / np.sum(self.nu)) * nmissing
            lnpCT += lnpCT_missing
        else:
            lnpCT = np.sum(self.lnjoint(True) * self.E_t)
        return lnpCT
# Lower Bound ------------------------------------------------------------------------------------------------------
    def lowerbound(self):
        # Expected Energy: entropy given the current parameter expectations
        lnpCT = self.logmarginallikelihood()
        lnpPi = gammaln(np.sum(self.alpha0, 1)) - np.sum(gammaln(self.alpha0), 1) \
                    + np.sum((self.alpha0 - 1) * self.lnPi, 1)
        lnpPi = np.sum(np.sum(lnpPi))
        lnpKappa = self.post_lnkappa()
        EEnergy = lnpCT + lnpPi + lnpKappa
        # Entropy of the variational distribution
        lnqT = self.q_ln_t()
        lnqPi = gammaln(np.sum(self.alpha, 1)) - np.sum(gammaln(self.alpha), 1) + \
                    np.sum((self.alpha - 1) * self.lnPi, 1)
        lnqPi = np.sum(np.sum(lnqPi))
        lnqKappa = self.q_lnkappa()
        H = lnqT + lnqPi + lnqKappa
        # Lower Bound
        L = EEnergy - H
        # logging.debug('EEnergy ' + str(EEnergy) + ', H ' + str(H))
        return L
# Hyperparameter Optimisation --------------------------------------------------------------------------------------
    def logmodeljoint(self, alphamean, numean):
        '''
        Weight the marginal likelihood/model evidence by the hyperprior. Unnormalised posterior over the hyperparams.
        '''
        marglh = self.logmarginallikelihood()
        # assume an exponential distribution over the hyperparameters
        palpha0 = np.sum(expon.logpdf(self.alpha0, scale=alphamean))
        pnu0 = np.sum(expon.logpdf(self.nu0, scale=numean))
        return marglh + palpha0 + pnu0

    def optimize_ibcc(self, alphamean, numean, crowdlabels, train_t, testidxs, maxiter=100):
        ''' 
        Assuming exponential distribution over the hyperparameters, we find the MAP values. The combiner object is updated
        to contain the optimal values, searched for using BFGS.
        '''
        def map_target_function(hyperparams, combiner, alphamean, numean):
            logging.debug("Hyperparms: %s" % str(hyperparams))
            if np.any(np.isnan(hyperparams)) or np.any(hyperparams <= 0):
                return np.inf
            combiner.nu0 = hyperparams[-combiner.nclasses:]
            initialK = (len(hyperparams) - combiner.nclasses) / (combiner.nclasses * combiner.nscores)
            alpha0_flat = hyperparams[0:combiner.nclasses * combiner.nscores * initialK]
            combiner.alpha0 = alpha0_flat.reshape((combiner.nclasses, combiner.nscores, initialK))
            # run the combiner
            combiner.combine_classifications([], [], preprocess=False)
            # evaluate the result
            neg_lml = -combiner.logmodeljoint(alphamean, numean)
            logging.debug("Negative log joint probability of the model & obs: %f" % neg_lml)
            return neg_lml
        # set up combiner object
        self.preprocess_data(crowdlabels, train_t, testidxs)
        hyperparams0 = np.concatenate((alphamean.flatten(), numean.flatten()))
        opt_hyperparams = fmin(map_target_function, hyperparams0, args=(self, alphamean, numean), maxiter=maxiter)
        logging.debug("Optimal hyperparams: %s" % str(opt_hyperparams))
        map_target_function(opt_hyperparams, self, alphamean, numean)

# Loader and Runner helper functions -------------------------------------------------------------------------------
def load_combiner(config_file, ibcc_class=None):
    dh = DataHandler()
    dh.loadData(config_file)
    if ibcc_class==None:
        combiner = IBCC(dh=dh)
    else:
        combiner = ibcc_class(dh=dh)
    return combiner, dh

def load_and_run_ibcc(configFile, ibcc_class=None):
    combiner, dh = load_combiner(configFile, ibcc_class)
    #combine labels
    pT = combiner.combine_classifications(dh.crowdlabels, dh.goldlabels)

    if dh.output_file != None:
        dh.saveTargets(pT)

    dh.save_pi(combiner.alpha, combiner.nclasses, combiner.nscores)
    return pT, combiner
    
if __name__ == '__main__':
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    load_and_run_ibcc(configFile)
