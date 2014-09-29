'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix
from scipy.special import psi, gammaln
from ibccdata import DataHandler

class Ibcc(object):
    
    useLowerBound = True #may be quicker not to calculate this
    
    minNoIts = 1
    maxNoIts = 500
    convThreshold = 0.0001
    
    crowdTable = False #crowd labels as a full KxnObjs table? Otherwise, use 
    # a sparse 3-column list, where 1st column=classifier ID, 2nd column = 
    # obj ID, 3rd column = score.
    
    nClasses = None
    nScores = None
    alpha0 = None
    nu0 = None
    K = None
    
    #The model
    lnKappa = []
    nu = []
    lnPi = []
    alpha = []
    ET = []
    
    obsIdxs = []
    
    keeprunning = True # set to false causes the combineClassifications method to exit without completing

    def expecLnKappa(self):#Posterior Hyperparams
        sumET = np.sum(self.ET[self.obsIdxs,:], 0)
        for j in range(self.nClasses):
            self.nu[j] = self.nu0[j] + sumET[j]
        self.lnKappa = psi(self.nu) - psi(np.sum(self.nu))
       
    def postAlpha(self):#Posterior Hyperparams -- move back to static IBCC
        for j in range(self.nClasses):
            for l in range(self.nScores):
                Tj = self.ET[:,j].reshape((self.nObjs,1))
                counts = self.C[l].T.dot(Tj).reshape(-1)
                self.alpha[j,l,:] = self.alpha0[j,l,:] + counts
       
    def expecLnPi(self):#Posterior Hyperparams
        self.postAlpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for s in range(self.nScores):        
            self.lnPi[:,s,:] = psi(self.alpha[:,s,:]) - psiSumAlpha
        
    def lnjoint_table(self):
        lnjoint = np.zeros((self.nObjs, self.nClasses))
        agentIdx = np.tile(np.transpose(range(self.K)), (self.nObjs,1)) 
    
        for j in range(self.nClasses):
            lnjoint[:,j] = np.sum(self.lnPi[j,self.crowdTable,agentIdx],1) + self.lnKappa[j]
        return lnjoint
        
    def lnjoint_sparseList(self):
        lnjoint = np.zeros((self.nObjs, self.nClasses))
        for j in range(self.nClasses):
            data = self.lnPi[j,self.crowdlabels[:,2],self.crowdlabels[:,0]].reshape(-1)
            rows = self.crowdlabels[:,1].reshape(-1)
            cols = np.zeros(self.crowdlabels.shape[0])
            
            likelihood_j = coo_matrix((data, (rows,cols)), shape=(self.nObjs,1)).todense()
            lnjoint[:,j] = likelihood_j.reshape(-1) + self.lnKappa[j]      
        return lnjoint
        
    def lnjoint(self):
        if self.crowdTable != None:
            return self.lnjoint_table()
        else:
            return self.lnjoint_sparseList()

    def expecT(self):
        
        self.ET = np.zeros((self.nObjs, self.nClasses))
        pT = joint = np.zeros((self.nObjs, self.nClasses))
        lnjoint = self.lnjoint()
            
        #ensure that the values are not too small
        largest = np.max(lnjoint, 1)
        for j in range(self.nClasses):
            joint[:,j] = lnjoint[:,j] - largest
            
        joint = np.exp(joint)
        norma = np.sum(joint, axis=1)
        for j in range(self.nClasses):
            pT[:,j] = joint[:,j]/norma
            self.ET[:,j] = pT[:,j]
            
        for j in range(self.nClasses):            
            #training labels
            row = np.zeros((1,self.nClasses))
            row[0,j] = 1
            self.ET[self.trainT==j,:] = row    
            
        return lnjoint
      
    def postLnJoint(self, lnjoint):
        lnpCT = np.sum(np.sum( lnjoint*self.ET ))                        
        return lnpCT      
            
    def postLnKappa(self):
        lnpKappa = gammaln(np.sum(self.nu0))-np.sum(gammaln(self.nu0)) \
                    + sum((self.nu0-1)*self.lnKappa)
        return lnpKappa
        
    def qLnKappa(self):
        lnqKappa = gammaln(np.sum(self.nu))-np.sum(gammaln(self.nu)) \
                        + np.sum((self.nu-1)*self.lnKappa)
        return lnqKappa
    
    def qLnT(self):
        ET = self.ET[self.ET!=0]
        return np.sum( ET*np.log(ET) )
        
    def lowerBound(self, lnjoint):
                        
        #probability of these targets is 1 as they are training labels
        #lnjoint[self.trainT!=-1,:] -= np.reshape(self.lnKappa, (1,self.nClasses))
        lnpCT = self.postLnJoint(lnjoint)                    
                        
        #alpha0 = np.reshape(self.alpha0, (self.nClasses, self.nScores, self.K))
        lnpPi = gammaln(np.sum(self.alpha0, 1))-np.sum(gammaln(self.alpha0),1) \
                    + np.sum((self.alpha0-1)*self.lnPi, 1)
        lnpPi = np.sum(np.sum(lnpPi))
            
        lnpKappa = self.postLnKappa()
            
        EEnergy = lnpCT + lnpPi + lnpKappa
        
        lnqT = self.qLnT()

        lnqPi = gammaln(np.sum(self.alpha, 1))-np.sum(gammaln(self.alpha),1) + \
                    np.sum( (self.alpha-1)*self.lnPi, 1)
        lnqPi = np.sum(np.sum(lnqPi))        
            
        lnqKappa = self.qLnKappa()
            
        H = - lnqT - lnqPi - lnqKappa
        L = EEnergy + H
        
        #logging.debug('EEnergy ' + str(EEnergy) + ', H ' + str(H))
        return L
        
    def preprocessTraining(self, crowdlabels, trainT=None):
        if trainT==None:
            if (crowdlabels.shape[1]!=3 or self.crowdTable != None):
                trainT = np.zeros(crowdlabels.shape[0]) -1
            else:
                trainT = np.zeros( len(np.unique(crowdlabels[:,1])) ) -1
        
        self.trainT = trainT
        self.nObjs = trainT.shape[0]        
        
    def preprocessCrowdLabels(self, crowdlabels):
        #ensure we don't have a matrix by mistake
        if not isinstance(crowdlabels, np.ndarray):
            crowdlabels = np.array(crowdlabels)
        C = {}
        if crowdlabels.shape[1]!=3 or self.crowdTable != None:            
            for l in range(self.nScores):
                Cl = np.zeros(crowdlabels.shape)
                Cl[crowdlabels==l] = 1
                C[l] = Cl
            self.crowdTable = crowdlabels
            self.obsIdxs = np.argwhere(np.sum(crowdlabels,axis=1)>=0)             
        else:            
            for l in range(self.nScores):
                lIdxs = np.where(crowdlabels[:,2]==l)[0]     
                data = np.array(np.ones((len(lIdxs),1))).reshape(-1)
                rows = np.array(crowdlabels[lIdxs,1]).reshape(-1)
                cols = np.array(crowdlabels[lIdxs,0]).reshape(-1)     
                Cl = coo_matrix((data,(rows,cols)), shape=(self.nObjs, self.K))
                C[l] = Cl
            self.crowdlabels = crowdlabels
            self.obsIdxs = np.unique(crowdlabels[:,1])
        self.C = C
        
    def initK(self, crowdlabels):
        if self.crowdTable != None:
            newK = self.crowdTable.shape[1]
        else:
            newK = np.max(crowdlabels[:,0])
        if self.K<=newK:
            self.K = newK+1 #+1 since we start from 0
            self.initParams() 
    
    def combineClassifications(self, crowdlabels, trainT=None):
        
        self.preprocessTraining(crowdlabels, trainT)
        self.initT()
        
        logging.info('IBCC Combining...')
        oldL = -np.inf
        converged = False
        self.nIts = 0 #object state so we can check it later
        
        crowdlabels = crowdlabels.astype(int)
        self.initK(crowdlabels)
        self.preprocessCrowdLabels(crowdlabels)
        
        while not converged and self.keeprunning:
            oldET = self.ET
            #play around with the order you start these in:
            #Either update the params using the priors+training labels for t
            #Or update the targets using the priors for the params
            #Usually prefer the latter so that all data points contribute meaningful info,
            #since the targets for the test data will be inited only to the kappa-priors, 
            #and training data is often insufficient -> could lead to biased result 

            #update targets
            lnjoint = self.expecT() 

            #update params
            self.expecLnKappa()
            self.expecLnPi()
        
            #check convergence        
            if self.useLowerBound:
                L = self.lowerBound(lnjoint)
                logging.debug('Lower bound: ' + str(L) + ', increased by ' + str(L-oldL))
                change = L-oldL                
                oldL = L
            else:
                change = np.sum(np.sum(np.absolute(oldET - self.ET)))            
            if (self.nIts>=self.maxNoIts or change<self.convThreshold) and self.nIts>self.minNoIts:
                converged = True
            self.nIts+=1
            if change<0:
                logging.warning('Ibcc iteration ' + str(self.nIts) + ' absolute change was ' + str(change) + '. Possible bug or rounding error?')            
            else:
                logging.debug('Ibcc iteration ' + str(self.nIts) + ' absolute change was ' + str(change))
               
            import gc
            gc.collect()               
                
        logging.info('IBCC finished in ' + str(self.nIts) + ' iterations (max iterations allowed = ' + str(self.maxNoIts) + ').')
        return self.ET
        
    def initParams(self):
        logging.debug('Initialising parameters...') 
        logging.debug('Alpha0: ' + str(self.alpha0))
        self.initLnPi()
        
        logging.debug('Nu0: ' + str(self.nu0))
        self.initLnKappa()
        
    def initLnKappa(self):
        if self.nu!=[]:
            return
        self.nu = deepcopy(np.float64(self.nu0))
        sumNu = np.sum(self.nu)
        self.lnKappa = psi(self.nu) - psi(sumNu)
        
    def initLnPi(self):
        if self.alpha!=[] and self.alpha.shape[2]==self.K:
            return
        if len(self.alpha0.shape)<3:
            self.alpha0 = np.array(self.alpha0[:,:,np.newaxis], dtype=np.float64)
            self.alpha0 = np.repeat(self.alpha0, self.K, axis=2)
        oldK = self.alpha0.shape[2] 
        if oldK<self.K:
            nnew = self.K - oldK
            alpha0new = self.alpha0[:,:,0]
            alpha0new = alpha0new[:,:,np.newaxis]
            alpha0new = np.repeat(alpha0new, nnew, axis=2)
            self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
            
        self.alpha = deepcopy(np.float64(self.alpha0))#np.float64(self.alpha0[:,:,np.newaxis])

        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        self.lnPi = np.zeros((self.nClasses,self.nScores,self.K))
        for s in range(self.nScores):        
            self.lnPi[:,s,:] = psi(self.alpha[:,s,:]) - psiSumAlpha 
        
    def initT(self):        
        kappa = self.nu / np.sum(self.nu, axis=0)        
        self.ET = np.zeros((self.nObjs,self.nClasses)) + kappa  
        
    def __init__(self, nClasses=2, nScores=2, alpha0=None, nu0=None, K=1, tableFormat=False, dh=None):
        if dh != None:
            self.nClasses = dh.nClasses
            self.nScores = len(dh.scores)
            self.alpha0 = dh.alpha0
            self.nu0 = dh.nu0
            self.K = dh.K
            tableFormat = dh.table_format
        else:
            self.nClasses = nClasses
            self.nScores = nScores
            self.alpha0 = alpha0
            self.nu0 = nu0
            self.K = K
        
        self.initParams()
        if tableFormat:
            self.crowdTable = True
        else:
            self.crowdTable = None        
            
def loadCombiner(config_file, ibcc_class=None):
    dh = DataHandler()
    dh.loadData(config_file)
    if ibcc_class==None:
        combiner = Ibcc(dh=dh)
    else:
        combiner = ibcc_class(dh=dh)
    return combiner, dh

def runIbcc(configFile, ibcc_class=None):
    combiner, dh = loadCombiner(configFile, ibcc_class)
    #combine labels
    pT = combiner.combineClassifications(dh.crowdlabels, dh.goldlabels)

    if dh.output_file != None:
        dh.saveTargets(pT)

    dh.save_pi(combiner.alpha, combiner.nClasses, combiner.nScores)
    return pT, combiner
    
if __name__ == '__main__':
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    runIbcc(configFile)
    
