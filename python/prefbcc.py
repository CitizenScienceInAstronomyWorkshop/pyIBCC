'''
Created on 21 Jan 2016

@author: edwin
'''

from ibcc import IBCC
import numpy as np
import logging
from scipy.sparse import coo_matrix, csr_matrix

class PrefBCC(IBCC):
    '''
    The functions below are what we may need to change to get preferences working with IBCC. The data can only be 
    supplied as a sparse list of preference pairs, where columns are:
    0: agent ID
    1: first object ID
    2: second object ID
    3: score, either 0 (1 < 2), 1 (1 == 2) or 2 (1 > 2).
    '''
    
    def __init__(self, nclasses=2, nscores=3, alpha0=None, nu0=None, K=1, uselowerbound=False, dh=None):
        # ignore nscores and set it to 3
        nscores = 3
        super(PrefBCC, self).__init__(nclasses, nscores, alpha0, nu0, K)     

    def init_lnPi(self):
        '''
        Always creates new self.alpha and self.lnPi objects and calculates self.alpha and self.lnPi values according to 
        either the prior, or where available, values of self.E_t from previous runs.
        '''
        self.alpha0 = self.alpha0.astype(float)
        self.alpha0 = np.tile(self.alpha0, self.nclasses, 1)
        self.alpha0  = self.alpha0[:,:,np.newaxis]
        
        self.alpha = np.zeros((self.nclasses**2, self.nscores, self.K), dtype=np.float) + self.alpha0
        self.lnPi = np.zeros((self.nclasses**2, self.nscores, self.K))
        self.expec_lnPi(posterior=False) # calculate alpha from the initial/prior values only in the first iteration
    
# Data preprocessing and helper functions --------------------------------------------------------------------------
   
    def preprocess_crowdlabels(self, crowdlabels):
        # Initialise all objects relating to the crowd labels.
        C = {}
        crowdlabels[np.isnan(crowdlabels)] = -1
        if self.discretedecisions:
            crowdlabels = np.round(crowdlabels).astype(int)
        if self.table_format_flag:# crowd labels as a full KxN table? If false, use diags sparse 3-column list, where 1st
            logging.error("Can't use table format with preference pairs at the moment.")
            return
        
        if self.K < int(np.nanmax(crowdlabels[:,0]))+1:
            self.K = int(np.nanmax(crowdlabels[:,0]))+1 # add one because indexes start from 0
            
        for l in range(self.nscores):
            lIdxs = np.argwhere(crowdlabels[:, 3] == l)[:,0]
            data = np.ones((len(lIdxs), 1)).reshape(-1)
            rows = np.array(crowdlabels[lIdxs, 1]).reshape(-1) * self.N + crowdlabels[lIdxs, 2]
            cols = np.array(crowdlabels[lIdxs, 0]).reshape(-1)
            
            Cl = csr_matrix(coo_matrix((data,(rows,cols)), shape=(self.N**2, self.K)))
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
        
# Posterior Updates to Hyperparameters --------------------------------------------------------------------------------
    def post_Alpha(self):  # Posterior Hyperparams           
        # Add the counts from the test data
        for j in range(self.nclasses):
            Tj = self.E_t[:, j].reshape((self.Ntest, 1))
                   
            for j2 in range(self.nclasses):
                Tj2 = self.E_t[:, j2].reshape((self.Ntrain, 1))         
                Tpair = Tj[:, np.newaxis].dot(Tj2[np.newaxis, :]).flatten()[:, np.newaxis]
                        
                rowidx = j * self.nclasses + j2
                
                for l in range(self.nscores):
                    self.alpha[rowidx, l, :] = self.C[l].T.dot(Tpair).reshape(-1)                
                
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def lnjoint(self, alldata=False):
        '''
        For use with crowdsourced data in table format (should be converted on input)
        '''
        if self.uselowerbound or alldata:
            for j in range(self.nclasses):
                data = []
                for j2 in range(self.nclasses):
                    rowidx = j * self.nclasses + j2
                    
                    for l in range(self.nscores):
                        data_l = self.C[l].multiply(self.lnPi[rowidx, l, :][np.newaxis,:])                        
                        data = data_l if data==[] else data + data_l
                self.lnpCT[:, j] = np.array(data.sum(axis=1)).reshape(-1) + self.lnkappa[j]
        else:
            logging.error("Not implemented yet -- please use lower bound")
            
            
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)
    
    N = 100
    J = 3
    t = np.random.randint(0, J, N)
    
    K = 5
    alpha0 = np.array([[5, 3, 1], [1, 3, 1], [1, 3, 5]], dtype=float)
    from scipy.stats import dirichlet 
    pi = np.zeros((J**2, 3, K))
    
    
                            