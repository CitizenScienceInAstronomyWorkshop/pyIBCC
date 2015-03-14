'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix
from scipy.special import psi, gammaln
from ibccdata import DataHandler
from scipy.optimize import fmin
from scipy.stats import gamma

class IBCC(object):
    
    uselowerbound = True #may be quicker not to calculate this
    
    min_iterations = 1
    max_iterations = 500
    conv_threshold = 0.0001
    
    table_format_flag = False #crowd labels as a full KxnObjs table? 
    #Otherwise, use 
    # a sparse 3-column list, where 1st column=classifier ID, 2nd column = 
    # obj ID, 3rd column = score.
    crowdlabels = None
    
    nclasses = None
    nscores = None
    alpha0 = None
    nu0 = None
    K = None
    N = 0 #number of objects
    
    #The model
    lnkappa = []
    nu = []
    lnPi = []
    alpha = []
    E_t = []
    
    observed_idxs = []
    
    keeprunning = True # set to false causes the combine_classifications method to exit without completing
    
    #hyperprior parameters
    gam_scale_alpha = None  #Gamma distribution params 
    gam_shape_alpha = 5 #Gamma distribution params --> confidence in seed values
    
    gam_scale_nu = None
    gam_shape_nu = 5
    
    noptiter = 0

    def expec_lnkappa(self):#Posterior Hyperparams
        sumET = np.sum(self.E_t[self.observed_idxs,:], 0)
        for j in range(self.nclasses):
            self.nu[j] = self.nu0[j] + sumET[j]
        self.lnkappa = psi(self.nu) - psi(np.sum(self.nu))
       
    def post_Alpha(self):#Posterior Hyperparams -- move back to static IBCC
        for j in range(self.nclasses):
            for l in range(self.nscores):
                Tj = self.E_t[:,j].reshape((self.N,1))
                counts = self.C[l].T.dot(Tj).reshape(-1)
                self.alpha[j,l,:] = self.alpha0[j,l,:] + counts
       
    def expec_lnPi(self):#Posterior Hyperparams
        self.post_Alpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for s in range(self.nscores):        
            self.lnPi[:,s,:] = psi(self.alpha[:,s,:]) - psiSumAlpha
        
    def lnjoint_table(self):
        lnjoint = np.zeros((self.N, self.nclasses))
        agentIdx = np.tile(np.transpose(range(self.K)), (self.N,1)) 
    
        for j in range(self.nclasses):
            lnjoint[:,j] = np.sum(self.lnPi[j,self.table_format_flag,agentIdx],1) + self.lnkappa[j]
        return lnjoint
        
    def lnjoint_sparselist(self):
        lnjoint = np.zeros((self.N, self.nclasses))
        for j in range(self.nclasses):
            data = self.lnPi[j,self.crowdlabels[:,2],self.crowdlabels[:,0]].reshape(-1)
            rows = self.crowdlabels[:,1].reshape(-1)
            cols = np.zeros(self.crowdlabels.shape[0])
            
            likelihood_j = coo_matrix((data, (rows,cols)), shape=(self.N,1)).todense()
            lnjoint[:,j] = likelihood_j.reshape(-1) + self.lnkappa[j]      
        return lnjoint
        
    def lnjoint(self):
        if self.table_format_flag != None:
            return self.lnjoint_table()
        else:
            return self.lnjoint_sparselist()

    def expec_t(self):
        
        self.E_t = np.zeros((self.N, self.nclasses))
        pT = joint = np.zeros((self.N, self.nclasses))
        lnjoint = self.lnjoint()
            
        #ensure that the values are not too small
        largest = np.max(lnjoint, 1)
        for j in range(self.nclasses):
            joint[:,j] = lnjoint[:,j] - largest
            
        joint = np.exp(joint)
        norma = np.sum(joint, axis=1)
        for j in range(self.nclasses):
            pT[:,j] = joint[:,j]/norma
            self.E_t[:,j] = pT[:,j]
            
        for j in range(self.nclasses):            
            #training labels
            row = np.zeros((1,self.nclasses))
            row[0,j] = 1
            self.E_t[self.train_t==j,:] = row    
            
        return lnjoint
      
    def post_lnjoint_ct(self, lnjoint):
        lnpCT = np.sum(np.sum( lnjoint*self.E_t ))                        
        return lnpCT      
            
    #Is this right?!!! Should be prior not post?
    def post_lnkappa(self):
        lnpKappa = gammaln(np.sum(self.nu0))-np.sum(gammaln(self.nu0)) \
                    + sum((self.nu0-1)*self.lnkappa)
        return lnpKappa
        
    def q_lnkappa(self):
        lnqKappa = gammaln(np.sum(self.nu))-np.sum(gammaln(self.nu)) \
                        + np.sum((self.nu-1)*self.lnkappa)
        return lnqKappa
    
    def q_ln_t(self):
        ET = self.E_t[self.E_t!=0]
        return np.sum( ET*np.log(ET) )
        
    def lowerbound(self, lnjoint):
                        
        #probability of these targets is 1 as they are training labels
        #lnjoint[self.train_t!=-1,:] -= np.reshape(self.lnkappa, (1,self.nclasses))
        lnpCT = self.post_lnjoint_ct(lnjoint)                    
                        
        #alpha0 = np.reshape(self.alpha0, (self.nclasses, self.nscores, self.K))
        lnpPi = gammaln(np.sum(self.alpha0, 1))-np.sum(gammaln(self.alpha0),1) \
                    + np.sum((self.alpha0-1)*self.lnPi, 1)
        lnpPi = np.sum(np.sum(lnpPi))
            
        lnpKappa = self.post_lnkappa()
            
        EEnergy = lnpCT + lnpPi + lnpKappa
        
        lnqT = self.q_ln_t()

        lnqPi = gammaln(np.sum(self.alpha, 1))-np.sum(gammaln(self.alpha),1) + \
                    np.sum( (self.alpha-1)*self.lnPi, 1)
        lnqPi = np.sum(np.sum(lnqPi))        
            
        lnqKappa = self.q_lnkappa()
            
        H = - lnqT - lnqPi - lnqKappa
        L = EEnergy + H
        
        #logging.debug('EEnergy ' + str(EEnergy) + ', H ' + str(H))
        return L
        
    def preprocess_training(self, crowdlabels, train_t=None):
        
        # Is this necessary? Prevents uncertain labels from crowd!
        crowdlabels = crowdlabels.astype(int) 
        self.crowdlabels = crowdlabels
        if crowdlabels.shape[1]!=3:
            self.table_format_flag = True
        
        if train_t==None:
            if self.table_format_flag:
                train_t = np.zeros(self.crowdlabels.shape[0]) -1
            else:
                train_t = np.zeros( len(np.unique(self.crowdlabels[:,1])) ) -1
        
        self.train_t = train_t
        self.N = train_t.shape[0]        
        
    def preprocess_crowdlabels(self):
        #ensure we don't have a matrix by mistake
        if not isinstance(self.crowdlabels, np.ndarray):
            self.crowdlabels = np.array(self.crowdlabels)
        C = {}
        if self.table_format_flag:            
            for l in range(self.nscores):
                Cl = np.zeros(self.crowdlabels.shape)
                Cl[self.crowdlabels==l] = 1
                C[l] = Cl
            self.observed_idxs = np.argwhere(np.sum(self.crowdlabels,axis=1)>=0)             
        else:            
            for l in range(self.nscores):
                lIdxs = np.where(self.crowdlabels[:,2]==l)[0]     
                data = np.array(np.ones((len(lIdxs),1))).reshape(-1)
                rows = np.array(self.crowdlabels[lIdxs,1]).reshape(-1)
                cols = np.array(self.crowdlabels[lIdxs,0]).reshape(-1)     
                Cl = coo_matrix((data,(rows,cols)), shape=(self.N, self.K))
                C[l] = Cl
            self.observed_idxs = np.unique(self.crowdlabels[:,1])
        self.C = C
        
    def init_K(self):
        if self.table_format_flag :
            newK = self.crowdlabels.shape[1]
        else:
            newK = np.max(self.crowdlabels[:,0])
        if self.K<=newK:
            self.K = newK+1 #+1 since we start from 0
            self.init_params() 
    
    def vb_inference(self):
              
        logging.info('IBCC Combining...')
        oldL = -np.inf
        converged = False
        self.nIts = 0 #object state so we can check it later

        while not converged and self.keeprunning:
            oldET = self.E_t
            #play around with the order you start these in:
            #Either update the params using the priors+training labels for t
            #Or update the targets using the priors for the params
            #Usually prefer the latter so that all data points contribute meaningful info,
            #since the targets for the test data will be inited only to the kappa-priors, 
            #and training data is often insufficient -> could lead to biased result 

            #update targets
            lnjoint = self.expec_t() 

            #update params
            self.expec_lnkappa()
            self.expec_lnPi()
        
            #check convergence        
            if self.uselowerbound:
                L = self.lowerbound(lnjoint)
                logging.debug('Lower bound: ' + str(L) + ', increased by ' + str(L-oldL))
                change = L-oldL                
                oldL = L
            else:
                change = np.sum(np.sum(np.absolute(oldET - self.E_t)))            
            self.nIts+=1
            if (self.nIts>=self.max_iterations or change<self.conv_threshold) and self.nIts>=self.min_iterations:
                converged = True
            if change<-0.00001: #use this small value so we don't worry about likely rounding errors
                logging.warning('IBCC iteration ' + str(self.nIts) + ' absolute change was ' + str(change) + '. Possible bug or rounding error?')            
            else:
                logging.debug('IBCC iteration ' + str(self.nIts) + ' absolute change was ' + str(change))
                
        logging.info('IBCC finished in ' + str(self.nIts) + ' iterations (max iterations allowed = ' + str(self.max_iterations) + ').')
        return self.E_t
        
    def init_params(self):
        logging.debug('Initialising parameters...') 
        logging.debug('Alpha0: ' + str(self.alpha0))
        self.init_lnPi()
        
        logging.debug('Nu0: ' + str(self.nu0))
        self.init_lnkappa()
        
    def init_lnkappa(self):
        if self.nu!=[]:
            return
        self.nu = deepcopy(np.float64(self.nu0))
        sumNu = np.sum(self.nu)
        self.lnkappa = psi(self.nu) - psi(sumNu)
        
    def init_alpha0(self):
        if len(self.alpha0_seed.shape)<3: #alpha0_seed needs to be duplicated
            self.alpha0 = np.array(self.alpha0_seed[:,:,np.newaxis], dtype=np.float64)
            self.alpha0 = np.repeat(self.alpha0, self.K, axis=2)
        else: #alpha0_seed is the complete set of initial pseudocounts
            self.alpha0 = self.alpha0_seed        
        
    def init_lnPi(self):
        #if alpha is already initialised, and no new agents, skip this
        if self.alpha!=[] and self.alpha.shape[2]==self.K:
            return
                
        #ensure alpha0 is the right size
        #First run -- alpha0 is not initialised
        if self.alpha0==None:
            self.init_alpha0()
        #Not first run but new agents have submitted crowdlabels
        elif self.alpha0.shape[2] < self.K:
            nnew = self.K - self.alpha0.shape[2]
            alpha0new = self.alpha0[:,:,0]
            alpha0new = alpha0new[:,:,np.newaxis]
            alpha0new = np.repeat(alpha0new, nnew, axis=2)
            self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
            
        self.alpha = deepcopy(np.float64(self.alpha0))#np.float64(self.alpha0[:,:,np.newaxis])

        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        self.lnPi = np.zeros((self.nclasses,self.nscores,self.K))
        for s in range(self.nscores):        
            self.lnPi[:,s,:] = psi(self.alpha[:,s,:]) - psiSumAlpha 
        
    def init_t(self):        
        kappa = self.nu / np.sum(self.nu, axis=0)        
        self.E_t = np.zeros((self.N,self.nclasses)) + kappa  
        
    def __init__(self, nclasses=2, nscores=2, alpha0=None, nu0=None, K=1, table_format=False, dh=None):
        if dh != None:
            self.nclasses = dh.nclasses
            self.nscores = len(dh.scores)
            self.alpha0_seed = dh.alpha0
            self.nu0 = dh.nu0
            self.K = dh.K
            table_format = dh.table_format
        else:
            self.nclasses = nclasses
            self.nscores = nscores
            self.alpha0_seed = alpha0
            self.nu0 = nu0
            self.K = K
        
        self.init_params()
        if table_format:
            self.table_format_flag = True
        else:
            self.table_format_flag = None        
        
    def unflatten_hyperparams(self,hyperparams):
        alpha_shape = self.alpha0_seed.shape
        n_alpha_elements = np.prod(alpha_shape)        
        alpha0 = hyperparams[0:n_alpha_elements].reshape(alpha_shape)
        nu0 = hyperparams[n_alpha_elements:]
        
        return alpha0,nu0
            
    def ln_modelprior(self):
        #Gamma distribution over each value. Set the params of the gammas.
        p_alpha0 = gamma.logpdf(self.alpha0_seed, self.gam_shape_alpha, scale=self.gam_scale_alpha)
        p_nu0 = gamma.logpdf(self.nu0, self.gam_shape_nu, scale=self.gam_scale_nu)
        
        return np.sum(np.sum(np.sum(p_alpha0))) + np.sum(p_nu0)
            
    def neg_marginal_likelihood(self, hyperparams):
        #Reshape to get alpha0 again
        self.alpha0_seed, self.nu0 = self.unflatten_hyperparams(hyperparams)
        self.init_alpha0() #reinitialise with new hyperparams
        self.expec_lnPi() #ensure new alpha0 values are used
        self.vb_inference() #run inference algorithm
        
        lnjoint = self.expec_t() 
        data_loglikelihood = self.post_lnjoint_ct(lnjoint)
        
        log_model_prior = self.ln_modelprior()
        
        ml = data_loglikelihood + log_model_prior
        
        return -ml #returns Negative!
        
    def combine_classifications(self, crowdlabels, goldlabels, optimise_hyperparams=True):
                
        self.preprocess_training(crowdlabels, goldlabels)
        self.init_t()
        self.init_K()
        self.preprocess_crowdlabels()          
        
        if optimise_hyperparams:
            self.optimise_hyperparams()
            return self.E_t
        else:
            return self.vb_inference()
        
    def optimise_hyperparams(self):
        #Initialise the hyperhyperparams
        self.gam_shape_alpha = np.float(self.gam_shape_alpha)
        self.gam_shape_nu = np.float(self.gam_shape_nu)
        
        self.gam_scale_alpha = self.alpha0_seed/self.gam_shape_alpha
        self.gam_scale_nu = self.nu0/self.gam_shape_nu

        #Evaluate the first guess using the mean hyperparams
        initialguess = np.concatenate((self.alpha0_seed.flatten(),self.nu0.flatten()))
        negml = self.neg_marginal_likelihood(initialguess)
        logging.info("Initial guess marginal log likelihood: " + str(-negml))        
        
        #Run the optimisation
        combfunc = self.neg_marginal_likelihood
        xopt,_,niterations,_,_ = fmin(func=combfunc, x0=initialguess, maxiter=1000, full_output=True)
        #also try fmin_cq(func=combfunc, x0=initialguess, maxiter=10000, fprime=???)
        self.alpha0_seed, self.nu0 = self.unflatten_hyperparams(xopt)
        
        logging.info("Hyperparameters optimised using ML or MAP estimation: ")
        logging.info("alpha0: " + str(self.alpha0_seed))
        logging.info("nu0: " + str(self.nu0))
        self.noptiter = niterations 
        
        logging.info("Rerunning to produce results with these optimal hyperparams.")   
        
        #Return an evaluation using the chosen values
        negml = self.neg_marginal_likelihood(xopt)
        logging.info("Maximum marginal log likelihood: " + str(-negml))
        
            
def load_combiner(config_file, ibcc_class=None):
    dh = DataHandler()
    dh.loadData(config_file)
    if ibcc_class==None:
        combiner = IBCC(dh=dh)
    else:
        combiner = ibcc_class(dh=dh)
    return combiner, dh

def runIbcc(configFile, ibcc_class=None, optimise_hyperparams=True):
    combiner, dh = load_combiner(configFile, ibcc_class)
    #combine labels
    pT = combiner.combine_classifications(dh.crowdlabels, dh.goldlabels, optimise_hyperparams)

    if dh.output_file != None:
        dh.save_targets(pT)

    dh.save_pi(combiner.alpha, combiner.nclasses, combiner.nscores)
    dh.save_hyperparams(combiner.alpha, combiner.nu, combiner.noptiter)
    return pT, combiner
    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    runIbcc(configFile)
    
    
