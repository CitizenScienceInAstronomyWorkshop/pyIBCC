'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import psi, gammaln
from ibccdata import DataHandler
from scipy.optimize import fmin
from scipy.stats import gamma

class IBCC(object):
# Print extra debug info
    verbose = True
    keeprunning = True # set to false causes the combine_classifications method to exit without completing if another 
    # thread is checking whether IBCC is taking too long. Probably won't work well if the optimize_hyperparams is true.
    noptiter = 0 # Number of optimisation iterations used to find the current hyper-parameters.    
# Configuration for variational Bayes (VB) algorithm for approximate inference -------------------------------------
    # determine convergence by calculating lower bound? Quicker to set=False so we check convergence of target variables
    uselowerbound = False
    min_iterations = 1
    max_iterations = 500
    conv_threshold = 0.0001
# Data set attributes -----------------------------------------------------------------------------------------------
    discretedecisions = False  # If true, decisions are rounded to discrete integers. If false, you can submit undecided
    # responses as fractions between two classes. E.g. 2.3 means that 0.3 of the decision will go to class 3, and 0.7
    # will go to class 2. Only neighbouring classes can have uncertain decisions in this way.
    table_format_flag = False  
    nclasses = None
    nscores = None
    K = None
    N = 0 #number of objects
    Ntrain = 0  # no. training objects
    Ntest = 0  # no. test objects
    # Sparsity handling
    sparse = False
    observed_idxs = []
    full_N = 0
    # The data from the crowd
    C = None
    Ctest = None # data for the test points (excluding training)
    goldlabels = None
    # Indices into the current data set
    trainidxs = None
    testidxs = None
    conf_mat_ind = []  # indices into the confusion matrices corresponding to the current set of crowd labels
    # the joint likelihood (interim value saved to reduce computation)
    lnpCT = None  
# Model parameters and hyper-parameters -----------------------------------------------------------------------------
    #The model
    alpha0 = None
    nu0 = None    
    lnkappa = []
    nu = []
    lnPi = []
    alpha = []
    E_t = []
    #hyper-hyper-parameters: the parameters for the hyper-prior over the hyper-parameters. These are only used if you
    # run optimize_hyperparams
    gam_scale_alpha = []  #Gamma distribution scale parameters 
    gam_shape_alpha = 10 #Gamma distribution shape parameters --> confidence in seed values
    gam_scale_nu = []
    gam_shape_nu = 200
# Initialisation ---------------------------------------------------------------------------------------------------
    def __init__(self, nclasses=2, nscores=2, alpha0=None, nu0=None, K=1, dh=None):
        if dh != None:
            self.nclasses = dh.nclasses
            self.nscores = len(dh.scores)
            self.alpha0 = dh.alpha0
            self.nu0 = dh.nu0
            self.K = dh.K
            self.uselowerbound = dh.uselowerbound
        else:
            self.nclasses = nclasses
            self.nscores = nscores
            self.alpha0 = alpha0
            self.nu0 = nu0
            self.K = K
            
        # Ensure we have float arrays so we can do division with these parameters properly
        self.nu0 = self.nu0.astype(float)
        if self.nu0.ndim==1:
            self.nu0 = self.nu0.reshape((self.nclasses,1))
        elif self.nu0.shape[0]!=self.nclasses and self.nu0.shape[1]==self.nclasses:
            self.nu0 = self.nu0.T            
        
    def init_params(self, force_reset=False):
        '''
        Checks that parameters are intialized, but doesn't overwrite them if already set up.
        '''
        if self.verbose:
            logging.debug('Initialising parameters...Alpha0: ' + str(self.alpha0))
        #if alpha is already initialised, and no new agents, skip this
        if self.alpha == [] or self.alpha.shape[2] != self.K or force_reset:
            self.init_lnPi()
        if self.verbose:
            logging.debug('Nu0: ' + str(self.nu0))
        if self.nu ==[] or force_reset:
            self.init_lnkappa()

    def init_lnkappa(self):
        self.nu = deepcopy(np.float64(self.nu0))
        sumNu = np.sum(self.nu)
        self.lnkappa = psi(self.nu) - psi(sumNu)

    def init_lnPi(self):
        '''
        Always creates new self.alpha and self.lnPi objects and calculates self.alpha and self.lnPi values according to 
        either the prior, or where available, values of self.E_t from previous runs.
        '''
        self.alpha0 = self.alpha0.astype(float)
        # if we specify different alpha0 for some agents, we need to do so for all K agents. The last agent passed in 
        # will be duplicated for any missing agents.
        if len(self.alpha0.shape) == 3 and self.alpha0.shape[2] < self.K:
            # We have a new dataset with more agents than before -- create more priors.
            nnew = self.K - self.alpha0.shape[2]
            alpha0new = self.alpha0[:, :, 0]
            alpha0new = alpha0new[:, :, np.newaxis]
            alpha0new = np.repeat(alpha0new, nnew, axis=2)
            self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
        elif len(self.alpha0.shape)==2:
            self.alpha0  = self.alpha0[:,:,np.newaxis]
        # Make sure self.alpha is the right size as well. Values of self.alpha not important as we recalculate below
        self.alpha = np.zeros((self.nclasses, self.nscores, self.K), dtype=np.float) + self.alpha0
        self.lnPi = np.zeros((self.nclasses, self.nscores, self.K))        
        self.expec_lnPi()

    def init_t(self):
        kappa = (self.nu0 / np.sum(self.nu0, axis=0)).T
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
            jidxs = self.goldlabels == j
            uncert_trainidxs = uncert_trainidxs - jidxs
            self.E_t[jidxs, :] = row
        # deal with uncertain training idxs
        for j in range(self.nclasses):
            # values a fraction above class j
            partly_j_idxs = np.bitwise_and(self.goldlabels[uncert_trainidxs] > j, self.goldlabels[uncert_trainidxs] < j + 1)
            partly_j_idxs = uncert_trainidxs[partly_j_idxs]
            self.E_t[partly_j_idxs, j] = (j + 1) - self.goldlabels[partly_j_idxs]
            # values a fraction below class j
            partly_j_idxs = np.bitwise_and(self.goldlabels[uncert_trainidxs] < j, self.goldlabels[uncert_trainidxs] > j - 1)
            partly_j_idxs = uncert_trainidxs[partly_j_idxs]
            self.E_t[partly_j_idxs, j] = self.goldlabels[partly_j_idxs] - j + 1
        if self.sparse:
            self.E_t_sparse = self.E_t  # current working version is sparse            

# Data preprocessing and helper functions --------------------------------------------------------------------------
    def desparsify_crowdlabels(self, crowdlabels):
        '''
        Converts the IDs of data points in the crowdlabels to a set of consecutive integer indexes. If a data point has
        no crowdlabels when using table format, it will be skipped.   
        '''
        if self.table_format_flag:
            # First, record which objects were actually observed.
            self.observed_idxs = np.argwhere(np.sum(np.isfinite(crowdlabels), axis=1) > 0).reshape(-1)
            # full set of test points will include those with crowd labels = NaN, unless gold labels are passed in 
            self.full_N = crowdlabels.shape[0]             
            if crowdlabels.shape[0] > len(self.observed_idxs):
                self.sparse = True
                # cut out the unobserved data points. We'll put them back in at the end of the classification procedure.
                crowdlabels = crowdlabels[self.observed_idxs, :]
        else:
            crowdobjects = crowdlabels[:,1].astype(int)
            self.observed_idxs, mappedidxs = np.unique(crowdobjects, return_inverse=True)
            self.full_N = np.max(crowdlabels[:,1])
            if self.full_N > len(self.observed_idxs):
                self.sparse = True
                # map the IDs so we skip unobserved data points. We'll map back at the end of the classification procedure.
                crowdlabels[:, 1] = mappedidxs
        return crowdlabels

    def preprocess_goldlabels(self, goldlabels):
        if goldlabels != None and self.sparse:
            if self.full_N<len(goldlabels):
                # the full set of test points that we output will come from the gold labels if longer than crowd labels
                self.full_N = len(goldlabels)
            goldlabels = goldlabels[self.observed_idxs]
            
        # Find out how much training data and how many total data points we have
        if goldlabels != None:
            len_t = goldlabels.shape[0]
            goldlabels[np.isnan(goldlabels)] = -1
        else:
            len_t = 0  # length of the training vector
        len_c = len(self.observed_idxs)# length of the crowdlabels
        # How many data points in total?
        if len_c > len_t:
            self.N = len_c
        else:
            self.N = len_t        
             
        # Make sure that goldlabels is the right size in case we have passed in a training set for the first idxs
        if goldlabels == None:
            self.goldlabels = np.zeros(self.N) - 1
        elif goldlabels.shape[0] < self.N:
            extension = np.zeros(self.N - goldlabels.shape[0]) - 1
            self.goldlabels = np.concatenate((goldlabels, extension))
        else:
            self.goldlabels = goldlabels
            
    def set_test_and_train_idxs(self, testidxs=None):
        # record the test and train idxs
        self.trainidxs = self.goldlabels > -1
        self.Ntrain = np.sum(self.trainidxs)
        
        if testidxs != None:  # include the pre-specified set of unlabelled data points in the inference process. All
            # other data points are either training data or ignored.
            self.testidxs = testidxs[self.observed_idxs]
        else:  # If the test indexes are not specified explicitly, assume that all data points with a NaN or a -1 in the
            # training data must be test indexes.
            self.testidxs = np.bitwise_or(np.isnan(self.goldlabels), self.goldlabels < 0)
        self.Ntest = np.sum(self.testidxs)            
            
    def preprocess_crowdlabels(self, crowdlabels):
        # Initialise all objects relating to the crowd labels.
        C = {}
        crowdlabels[np.isnan(crowdlabels)] = -1
        if self.discretedecisions:
            crowdlabels = np.round(crowdlabels).astype(int)
        if self.table_format_flag:# crowd labels as a full KxN table? If false, use a sparse 3-column list, where 1st
            # column=classifier ID, 2nd column = obj ID, 3rd column = score.
            self.K = crowdlabels.shape[1]
            for l in range(self.nscores):
                Cl = np.zeros((self.N, self.K))
                #crowd labels may not be supplied for all N data points in the gold labels, so use argwhere
                lidxs = np.argwhere(crowdlabels==l)
                Cl[lidxs[:,0], lidxs[:,1]] = 1
                if not self.discretedecisions:
                    if l + 1 < self.nscores:
                        partly_l_idxs = np.bitwise_and(crowdlabels > l, crowdlabels < (l+1))  # partly above l
                        Cl[partly_l_idxs] = (l + 1) - crowdlabels[partly_l_idxs]
                    if l > 0:
                        partly_l_idxs = np.bitwise_and(crowdlabels < l, crowdlabels > (l-1))  # partly below l
                        Cl[partly_l_idxs] = crowdlabels[partly_l_idxs] - l + 1
                C[l] = Cl
        else:
            self.K = np.nanmax(crowdlabels[:,0])+1 # add one because indexes start from 0
            for l in range(self.nscores):
                lIdxs = np.argwhere(crowdlabels[:, 2] == l)[:,0]
                data = np.ones((len(lIdxs), 1)).reshape(-1)
                rows = np.array(crowdlabels[lIdxs, 1]).reshape(-1)
                cols = np.array(crowdlabels[lIdxs, 0]).reshape(-1)
                
                if not self.discretedecisions:
                    partly_l_idxs = np.bitwise_and(crowdlabels[:, 2] > l, crowdlabels[:, 2] < l + 1)  # partly above l
                    data = np.concatenate((data, (l + 1) - crowdlabels[partly_l_idxs, 2]))
                    rows = np.concatenate((rows, crowdlabels[partly_l_idxs, 1].reshape(-1)))
                    cols = np.concatenate((cols, crowdlabels[partly_l_idxs, 0].reshape(-1)))
                    
                    partly_l_idxs = np.bitwise_and(crowdlabels[:, 2] < l, crowdlabels[:, 2] > l - 1)  # partly below l
                    data = np.concatenate((data, crowdlabels[partly_l_idxs, 2] - l + 1))
                    rows = np.concatenate((rows, crowdlabels[partly_l_idxs, 1].reshape(-1)))
                    cols = np.concatenate((cols, crowdlabels[partly_l_idxs, 0].reshape(-1)))
                    Cl = csr_matrix(coo_matrix((data,(rows,cols)), shape=(self.N, self.K)))
                C[l] = Cl
        # Set and reset object properties for the new dataset
        self.C = C
        self.lnpCT = np.zeros((self.N, self.nclasses))
        self.conf_mat_ind = []
        # pre-compute the indices into the pi arrays
        # repeat for test labels only
        
        self.Ctest = {}
        for l in range(self.nscores):
            self.Ctest[l] = C[l][self.testidxs, :]
        # Reset the pre-calculated data for the training set in case goldlabels has changed
        self.alpha_tr = []
        
    def resparsify_t(self):
        '''
        Puts the expectations of target values, E_t, at the points we observed crowd labels back to their original 
        indexes in the output array. Values are inserted for the unobserved indices using only kappa (class proportions).
        '''
        E_t_full = np.zeros((self.full_N,self.nclasses))
        E_t_full[:] = (np.exp(self.lnkappa) / np.sum(np.exp(self.lnkappa),axis=0)).T
        E_t_full[self.observed_idxs,:] = self.E_t
        self.E_t_sparse = self.E_t  # save the sparse version
        self.E_t = E_t_full        
        
# Run the inference algorithm --------------------------------------------------------------------------------------    
    def combine_classifications(self, crowdlabels, goldlabels=None, testidxs=None, optimise_hyperparams=False, table_format=False):
        '''
        Takes crowdlabels in either sparse list or table formats, along with optional training labels (goldlabels)
        and applies data-preprocessing steps before running inference for the model parameters and target labels.
        Returns the expected values of the target labels.
        '''
        self.table_format_flag = table_format
        oldK = self.K
        crowdlabels = self.desparsify_crowdlabels(crowdlabels)
        self.preprocess_goldlabels(goldlabels)        
        self.set_test_and_train_idxs(testidxs)
        self.preprocess_crowdlabels(crowdlabels)
        self.init_t()
        #Check that we have the right number of agents/base classifiers, K, and initialise parameters if necessary
        if self.K != oldK or self.nu == [] or self.alpha==[]:  # data shape has changed or not initialised yet
            self.init_params()
        # Either run the model optimisation or just use the inference method with fixed hyper-parameters  
        if optimise_hyperparams:
            self.optimize_hyperparams()
        else:
            self.run_inference() 
        if self.sparse:
            self.resparsify_t()
        return self.E_t        

    def run_inference(self):   
        '''
        Variational approximate inference. Assumes that all data and hyper-parameters are ready for use. Overwrite
        do implement EP or Gibbs' sampling etc.
        '''
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
                if self.verbose:
                    logging.debug('Lower bound: ' + str(L) + ', increased by ' + str(L - oldL))
                change = L-oldL
                oldL = L
            else:
                change = np.max(np.sum(np.absolute(oldET - self.E_t)))
            if (self.nIts>=self.max_iterations or change<self.conv_threshold) and self.nIts>self.min_iterations:
                converged = True
            self.nIts+=1
            if change < -0.001 and self.verbose:                
                logging.warning('IBCC iteration ' + str(self.nIts) + ' absolute change was ' + str(change) + '. Possible bug or rounding error?')            
            elif self.verbose:
                logging.debug('IBCC iteration ' + str(self.nIts) + ' absolute change was ' + str(change))
        logging.info('IBCC finished in ' + str(self.nIts) + ' iterations (max iterations allowed = ' + str(self.max_iterations) + ').')

# Posterior Updates to Hyperparameters --------------------------------------------------------------------------------
    def post_Alpha(self):  # Posterior Hyperparams
        # Save the counts from the training data so we only recalculate the test data on every iteration
        if self.alpha_tr == []:
            self.alpha_tr = self.alpha.copy()
            for j in range(self.nclasses):
                for l in range(self.nscores):
                    Tj = self.E_t[self.trainidxs, j].reshape((self.Ntrain, 1))
                    self.alpha_tr[j,l,:] = self.C[l][self.trainidxs,:].T.dot(Tj).reshape(-1)
            self.alpha_tr += self.alpha0
        # Add the counts from the test data
        for j in range(self.nclasses):
            for l in range(self.nscores):
                Tj = self.E_t[self.testidxs, j].reshape((self.Ntest, 1))
                counts = self.Ctest[l].T.dot(Tj).reshape(-1)
                self.alpha[j, l, :] = self.alpha_tr[j, l, :] + counts
# Expectations: methods for calculating expectations with respect to parameters for the VB algorithm ------------------
    def expec_lnkappa(self):
        sumET = np.sum(self.E_t, 0)
        for j in range(self.nclasses):
            self.nu[j] = self.nu0[j] + sumET[j]
        self.lnkappa = psi(self.nu) - psi(np.sum(self.nu))

    def expec_lnPi(self):
        # check if E_t has been initialised. Only update alpha if it has. Otherwise E[lnPi] is given by the prior
        if self.E_t != []:
            self.post_Alpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for s in range(self.nscores): 
            self.lnPi[:, s, :] = psi(self.alpha[:, s, :]) - psiSumAlpha      

    def expec_t(self):
        self.lnjoint()
        joint = self.lnpCT
        joint = joint[self.testidxs, :]
        # ensure that the values are not too small
        largest = np.max(joint, 1)[:, np.newaxis]
        joint = joint - largest
        joint = np.exp(joint)
        norma = np.sum(joint, axis=1)[:, np.newaxis]
        pT = joint / norma
        self.E_t[self.testidxs, :] = pT      
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def lnjoint(self, alldata=False):
        '''
        For use with crowdsourced data in table format (should be converted on input)
        '''
        if self.uselowerbound or alldata:
            for j in range(self.nclasses):
                data = []
                for l in range(self.nscores):
                    if self.table_format_flag:
                        data_l = self.C[l] * self.lnPi[j, l, :][np.newaxis,:]
                    else:
                        data_l = self.C[l].multiply(self.lnPi[j, l, :][np.newaxis,:])
                    data = data_l if data==[] else data+data_l
                self.lnpCT[:, j] = np.array(np.sum(data, 1)).reshape(-1) + self.lnkappa[j]
        else:  # no need to calculate in full
            for j in range(self.nclasses):
                data = []
                for l in range(self.nscores):
                    if self.table_format_flag:
                        data_l = self.Ctest[l] * self.lnPi[j, l, :][np.newaxis,:]
                    else:
                        data_l = self.Ctest[l].multiply(self.lnPi[j, l, :][np.newaxis,:])
                    data = data_l if data==[] else data+data_l
                self.lnpCT[self.testidxs, j] = np.array(np.sum(data, 1)).reshape(-1) + self.lnkappa[j]
        
    def post_lnkappa(self):
        lnpKappa = gammaln(np.sum(self.nu0)) - np.sum(gammaln(self.nu0)) + sum((self.nu0 - 1) * self.lnkappa)
        return lnpKappa
        
    def q_lnkappa(self):
        lnqKappa = gammaln(np.sum(self.nu)) - np.sum(gammaln(self.nu)) + np.sum((self.nu - 1) * self.lnkappa)
        return lnqKappa

    def q_ln_t(self):
        ET = self.E_t[self.E_t != 0]
        return np.sum(ET * np.log(ET))         

    def post_lnpi(self):
        x = np.sum((self.alpha0-1) * self.lnPi,1)
        z = gammaln(np.sum(self.alpha0,1)) - np.sum(gammaln(self.alpha0),1)
        return np.sum(x+z)
                    
    def q_lnPi(self):
        x = np.sum((self.alpha-1) * self.lnPi,1)
        z = gammaln(np.sum(self.alpha,1)) - np.sum(gammaln(self.alpha),1)
        return np.sum(x+z)
# Lower Bound ---------------------------------------------------------------------------------------------------------       
    def lowerbound(self):
        # Expected Energy: entropy given the current parameter expectations
        lnpCT = self.post_lnjoint_ct()
        lnpPi = self.post_lnpi()
        lnpKappa = self.post_lnkappa()
        EEnergy = lnpCT + lnpPi + lnpKappa
        # Entropy of the variational distribution
        lnqT = self.q_ln_t()
        lnqPi = self.q_lnPi()
        lnqKappa = self.q_lnkappa()
        H = lnqT + lnqPi + lnqKappa
        # Lower Bound
        L = EEnergy - H
        # logging.debug('EEnergy ' + str(EEnergy) + ', H ' + str(H))
        return L
# Hyperparameter Optimisation ------------------------------------------------------------------------------------------
    def unflatten_hyperparams(self,hyperparams):
        initialK = (len(hyperparams) - self.nclasses) / (self.nclasses * self.nscores)
        alpha_shape = (self.nclasses, self.nscores, initialK)
        n_alpha_elements = np.prod(alpha_shape)        
        alpha0 = hyperparams[0:n_alpha_elements].reshape(alpha_shape)
        nu0 = hyperparams[-self.nclasses:]
        return alpha0,nu0
    
    def post_lnjoint_ct(self):
        # If we have not already calculated lnpCT for the lower bound, then make sure we recalculate using all data
        if not self.uselowerbound:
            self.lnjoint(alldata=True)
        return np.sum(self.E_t * self.lnpCT)
                
    def ln_modelprior(self):
        #Check and initialise the hyper-hyper-parameters if necessary
        if self.gam_scale_alpha==[]:
            self.gam_shape_alpha = np.float(self.gam_shape_alpha)
            # if the scale was not set, assume current values of alpha0 are the means given by the hyper-prior
            self.gam_scale_alpha = self.alpha0/self.gam_shape_alpha
        if self.gam_scale_nu==[]:
            self.gam_shape_nu = np.float(self.gam_shape_nu)
            # if the scale was not set, assume current values of nu0 are the means given by the hyper-prior
            self.gam_scale_nu = self.nu0/self.gam_shape_nu
        
        #Gamma distribution over each value. Set the parameters of the gammas.
        p_alpha0 = gamma.logpdf(self.alpha0, self.gam_shape_alpha, scale=self.gam_scale_alpha)
        p_nu0 = gamma.logpdf(self.nu0, self.gam_shape_nu, scale=self.gam_scale_nu)
        
        return np.sum(p_alpha0) + np.sum(p_nu0)

    def neg_marginal_likelihood(self, hyperparams):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if self.verbose:
            logging.debug("Hyper-parameters: %s" % str(hyperparams))
        if np.any(np.isnan(hyperparams)) or np.any(hyperparams <= 0):
            return np.inf
        self.alpha0, self.nu0 = self.unflatten_hyperparams(hyperparams)
        
        #ensure new alpha0 and nu0 values are used when updating E_t
        self.init_params(force_reset=True)
        #run inference algorithm
        self.run_inference() 
        
        #calculate likelihood from the fitted model
        data_loglikelihood = self.post_lnjoint_ct()
        log_model_prior = self.ln_modelprior()
        lml = data_loglikelihood + log_model_prior
        logging.debug("Log joint probability of the model & data: %f" % lml)
        return -lml #returns Negative!
 
    def optimize_hyperparams(self, maxiter=200):
        ''' 
        Assuming gamma distributions over the hyper-parameters, we find the MAP values. The combiner object is updated
        to contain the optimal values, searched for using BFGS.
        '''
        #Evaluate the first guess using the current value of the hyper-parameters
        initialguess = np.concatenate((self.alpha0.flatten(), self.nu0.flatten()))
        opt_hyperparams,_,niterations,_,_ = fmin(self.neg_marginal_likelihood, initialguess, maxiter=maxiter, full_output=True)
        #also try fmin_cq(func=combfunc, x0=initialguess, maxiter=10000, fprime=???)
        
        self.alpha0, self.nu0 = self.unflatten_hyperparams(opt_hyperparams)
        logging.info("Hyperparameters optimised using ML or MAP estimation: ")
        logging.info("alpha0: " + str(self.alpha0))
        logging.info("nu0: " + str(self.nu0))        
        self.noptiter = niterations 
        logging.debug("Optimal hyperparams: %s" % str(opt_hyperparams))
        
        return self.E_t
    
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
    combiner, dh = load_combiner(configFile, ibcc_class)
    #combine labels
    combiner.verbose = True
    pT = combiner.combine_classifications(dh.crowdlabels, dh.goldlabels, optimise_hyperparams=optimise_hyperparams, 
                                          table_format=dh.table_format)

    if dh.output_file != None:
        dh.save_targets(pT)

    dh.save_pi(combiner.alpha, combiner.nclasses, combiner.nscores)
    dh.save_hyperparams(combiner.alpha, combiner.nu, combiner.noptiter)
    pT = dh.map_predictions_to_original_IDs(pT)
    return pT, combiner
    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    load_and_run_ibcc(configFile)
