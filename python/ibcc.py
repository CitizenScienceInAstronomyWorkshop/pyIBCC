'''
@author: Edwin Simpson
'''
import sys
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix
from scipy.special import psi, gammaln
import pickle

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

    def expecLnKappa(self):#Posterior Hyperparams
        sumET = np.sum(self.ET[self.obsIdxs,:], 0)
        for j in range(self.nClasses):
            self.nu[j] = self.nu0[j] + sumET[j]
        self.lnKappa = psi(self.nu) - psi(np.sum(self.nu))
       
    def postAlpha(self):#Posterior Hyperparams -- move back to static IBCC
        for j in range(self.nClasses):
            for l in range(self.nScores):
                counts = np.matrix(self.ET[:,j]) * self.C[l]
                self.alpha[j,l,:] = self.alpha0[j,l] + counts
       
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
            data = self.lnPi[j,self.crowdLabels[:,2],self.crowdLabels[:,0]].reshape(-1)
            rows = self.crowdLabels[:,1].reshape(-1)
            cols = np.zeros(self.crowdLabels.shape[0])
            
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
            pT[:,j] = np.divide(joint[:,j], norma)
            self.ET[:,j] = pT[:,j]
            
        for j in range(self.nClasses):            
            #training labels
            row = np.zeros((1,self.nClasses))
            row[0,j] = 1
            self.ET[self.trainT==j,:] = row    
            
        return lnjoint
      
    def postLnJoint(self, lnjoint):
        lnpCT = np.sum(np.sum( np.multiply(lnjoint, self.ET) ))                        
        return lnpCT      
            
    def postLnKappa(self):
        lnpKappa = gammaln(np.sum(self.nu0))-np.sum(gammaln(self.nu0)) \
                    + sum(np.multiply(self.nu0-1,self.lnKappa))
        return lnpKappa
        
    def qLnKappa(self):
        lnqKappa = gammaln(np.sum(self.nu))-np.sum(gammaln(self.nu)) \
                        + np.sum(np.multiply(self.nu-1,self.lnKappa))
        return lnqKappa
    
    def qLnT(self):
        ET = self.ET[self.ET!=0]
        return np.sum( np.multiply( ET,np.log(ET) ) )
        
    def lowerBound(self, lnjoint):
                        
        #probability of these targets is 1 as they are training labels
        #lnjoint[self.trainT!=-1,:] -= np.reshape(self.lnKappa, (1,self.nClasses))
        lnpCT = self.postLnJoint(lnjoint)                    
                        
        #alpha0 = np.reshape(self.alpha0, (self.nClasses, self.nScores, self.K))
        lnpPi = gammaln(np.sum(self.alpha0, 1))-np.sum(gammaln(self.alpha0),1) \
                    + np.sum(np.multiply(self.alpha0-1, self.lnPi), 1)
        lnpPi = np.sum(np.sum(lnpPi))
            
        lnpKappa = self.postLnKappa()
            
        EEnergy = lnpCT + lnpPi + lnpKappa
        
        lnqT = self.qLnT()

        lnqPi = gammaln(np.sum(self.alpha, 1))-np.sum(gammaln(self.alpha),1) + \
                    np.sum( np.multiply(self.alpha-1,self.lnPi), 1)
        lnqPi = np.sum(np.sum(lnqPi))        
            
        lnqKappa = self.qLnKappa()
            
        H = - lnqT - lnqPi - lnqKappa
        L = EEnergy + H
        
        #print 'EEnergy ' + str(EEnergy) + ', H ' + str(H)
        return L
        
    def preprocessTraining(self, crowdLabels, trainT=None):
        if trainT==None:
            if (crowdLabels.shape[1]!=3 or self.crowdTable != None):
                trainT = np.zeros(crowdLabels.shape[0]) -1
            else:
                trainT = np.zeros(np.max(crowdLabels[:,1])) -1
        
        self.trainT = trainT
        self.nObjs = trainT.shape[0]        
        
    def preprocessCrowdLabels(self, crowdLabels):
        C = {}
        if crowdLabels.shape[1]!=3 or self.crowdTable != None:            
            for l in range(self.nScores):
                Cl = np.matrix(np.zeros(crowdLabels.shape))
                Cl[crowdLabels==l] = 1
                C[l] = Cl
            self.crowdTable = crowdLabels
            self.obsIdxs = np.argwhere(np.sum(crowdLabels,axis=1)>=0)
        else:            
            for l in range(self.nScores):
                lIdxs = np.where(crowdLabels[:,2]==l)[0]     
                data = np.array(np.ones((len(lIdxs),1))).reshape(-1)
                rows = np.array(crowdLabels[lIdxs,1]).reshape(-1)
                cols = np.array(crowdLabels[lIdxs,0]).reshape(-1)     
                Cl = coo_matrix((data,(rows,cols)), shape=(self.nObjs, self.K))
                C[l] = Cl
            self.crowdLabels = crowdLabels
            self.obsIdxs = np.unique(crowdLabels[:1])
        self.C = C

    
    def combineClassifications(self, crowdLabels, trainT=None):
        
        self.preprocessTraining(crowdLabels, trainT)
        self.initT()
        
        print 'Combining...'
        oldL = -np.inf
        converged = False
        self.nIts = 0 #object state so we can check it later
        
        crowdLabels = crowdLabels.astype(int)
        self.preprocessCrowdLabels(crowdLabels)
        
        while not converged:
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
                #print 'Lower bound: ' + str(L) + ', increased by ' + str(L-oldL)
                change = L-oldL                
                if L-oldL<0:
                    print 'Possible error -> lower bound went down. Maybe rounding error?'
                oldL = L
            else:
                change = np.sum(np.sum(np.absolute(oldET - self.ET)))            
            if (self.nIts>=self.maxNoIts or change<self.convThreshold) and self.nIts>self.minNoIts:
                converged = True
            self.nIts+=1
            print 'Ibcc iteration ' + str(self.nIts) + ' absolute change was ' + str(change)
                
        print 'IBCC finished in ' + str(self.nIts) + ' iterations (max iterations allowed = ' + str(self.maxNoIts) + ').'
        return self.ET
        
    def initParams(self):
        print 'Initialising parameters...' 
        print 'Alpha0: ' + str(self.alpha0)
        self.initLnPi()
        
        print 'Nu0: ' + str(self.nu0)
        self.initLnKappa()
        
    def initLnKappa(self):
        if self.nu!=[]:
            return
        self.nu = deepcopy(np.float64(self.nu0))
        sumNu = np.sum(self.nu)
        self.lnKappa = psi(self.nu) - psi(sumNu)
        
    def initLnPi(self):
        if self.alpha!=[]:
            return
        if len(self.alpha0.shape)<3:
            self.alpha0 = np.float64(self.alpha0[:,:,np.newaxis])
            self.alpha0 = np.repeat(self.alpha0, self.K, axis=2)
        self.alpha = deepcopy(np.float64(self.alpha0))#np.float64(self.alpha0[:,:,np.newaxis])

        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        self.lnPi = np.zeros((self.nClasses,self.nScores,self.K))
        for s in range(self.nScores):        
            self.lnPi[:,s,:] = psi(self.alpha[:,s,:]) - psiSumAlpha 
        
    def initT(self):        
        kappa = self.nu / np.sum(self.nu, axis=0)        
        self.ET = np.matrix(np.zeros((self.nObjs,self.nClasses))) + kappa  
        
    def __init__(self, nClasses, nScores, alpha0, nu0, K, tableFormat=False):
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

def loadCrowdLabels(inputFile, scores):   
    '''
    Loads labels from crowd in sparse list format, i.e. 3 columns, classifier ID,
    object ID, score.
    '''
    pyFileExists = False
    try:
        with open(inputFile+'.dat','r') as inFile:
            crowdLabels, tIdxs, K = pickle.load(inFile)
            pyFileExists = True
    except Exception:
        print 'Will try to load a CSV file...'
    
        crowdLabels = np.genfromtxt(inputFile, delimiter=',', \
                                skip_header=1,usecols=[0,1,2])

        tIdxs, crowdLabels[:,1] = np.unique(crowdLabels[:,1],return_inverse=True)
        kIdxs, crowdLabels[:,0] = np.unique(crowdLabels[:,0],return_inverse=True)
        K = len(kIdxs)
        
    unmappedScores = np.round(crowdLabels[:,2])
    for i,s in enumerate(scores):
        crowdLabels[(unmappedScores==s),2] = i
    
    maxT = np.max(tIdxs)
    blanks = np.zeros(len(tIdxs))
    idxList = range(len(tIdxs))
    
    tIdxMap = coo_matrix(( idxList, (tIdxs,blanks)), shape=(maxT+1,1) )
    tIdxMap = tIdxMap.tocsr()
    
    if not pyFileExists:
        try:
            with open(inputFile+'.dat', 'wb') as outFile:
                pickle.dump((crowdLabels,tIdxs,K), outFile)
        except Exception:
            print 'Could not save the input data as a Python object file.'
    
    return crowdLabels, tIdxMap, tIdxs, K, len(tIdxs)
    
def loadCrowdTable(inputFile, scores):
    '''
    Loads crowd labels in a table format
    '''
    unmappedScores = np.round(np.genfromtxt(inputFile, delimiter=','))
    
    #tIdxs, crowdLabels[:,1] = np.unique(crowdLabels[:,1],return_inverse=True)
    #kIdxs, crowdLabels[:,0] = np.unique(crowdLabels[:,0],return_inverse=True)
    K = unmappedScores.shape[1]
    N = unmappedScores.shape[0]
    tIdxs = range(N)
            
    crowdTable = np.zeros((N,K))
    for i,s in enumerate(scores):
        crowdTable[unmappedScores==s] = i
    
    maxT = np.max(tIdxs)
    blanks = np.zeros(len(tIdxs))
    
    tIdxMap = coo_matrix(( tIdxs, (tIdxs,blanks)), shape=(maxT+1,1) )
    tIdxMap = tIdxMap.tocsr()
    return (crowdTable, tIdxMap, tIdxs, K, len(tIdxs))  
    
def loadGold(goldFile, tIdxMap, nObjs, classLabels=None, secondaryTypeCol=-1):   
    
    import os.path
    if not os.path.isfile(goldFile):
        print 'No gold labels found.'
        gold = np.zeros(nObjs) -1
        return gold, None
    
    if secondaryTypeCol>-1:
        useCols=[0,1,secondaryTypeCol]
    else:
        useCols=[0,1]
        
    try:
        gold = np.genfromtxt(goldFile, delimiter=',', skip_header=0,usecols=useCols,invalid_raise=True)
    except Exception:
        gold = np.genfromtxt(goldFile, delimiter=',', skip_header=0)
        
    if np.any(np.isnan(gold[0])): #skip header if necessary
        gold = gold[1:,:]
    print "gold shape: " + str(gold.shape)
    
    if len(gold.shape)==1 or gold.shape[1]==1:
        goldLabels = gold
    else:
        goldLabels = gold[:,1]
        
        goldIdxs = gold[:,0]
        #map the original idxs to local idxs
        goldIdxs = tIdxMap[goldIdxs,0].todense()
        
        #create an array for gold for all the objects/data points in this test set
        goldSorted = np.zeros(nObjs) -1
        goldSorted[goldIdxs] = gold[:,1]
        goldLabels = goldSorted
        
        #if there is secondary type info, create a similar array for this
        if secondaryTypeCol>-1:
            goldTypes = np.zeros(nObjs)
            goldTypes[goldIdxs] = gold[:,2]
            goldTypes[np.isnan(goldTypes)] = 0 #some examples may have no type info
            goldTypes[goldLabels==-1] = -1 #negative examples have type -1
          
    if classLabels:
        #convert text to class IDs
        for i in range(gold.shape[0]):
            classIdx = np.where(classLabels==goldLabels[i])
            if classIdx:
                goldLabels[i] = classIdx
            else:
                goldLabels[i] = -1
    
    if secondaryTypeCol>-1:
        return goldLabels, goldTypes
    else: 
        return goldLabels, None

def initFromConfig(configFile, K, tableFormat=False):
    nClasses = 2
    scores = np.array([3, 4])
    nu0 = np.array([50.0, 50.0])
    alpha0 = np.array([[2, 1], [1, 2]])     
        
    #read configuration
    with open(configFile, 'r') as conf:
        configuration = conf.read()
        exec(configuration)        
        
    nScores = len(scores)
    #initialise combiner
    combiner = Ibcc(nClasses, nScores, alpha0, nu0, K, tableFormat)
    
    return combiner
    
def loadData(configFile):
    
    #Defaults that will usually be overwritten by project config
    inputFile = './data/input.csv'
    tableFormat = False
    #columns in input file:
    # 0 = agent/worker/volunteer ID
    # 1 = object ID
    # 2 = scores given to the object
    goldFile = './data/gold.csv'
    #columns in gold file
    # 0 = object ID
    # 1 = class label
    scores = np.array([3, 4])
    
    outputFile =  './output/output.csv'
    confMatFile = './output/confMat.csv'
    classLabels = None
  
    trainIds = None #IDs of targets that should be used as training data. Optional 
    
    #column index of secondary type information about the data points stored in the gold file. 
    #-1 means no such info
    goldTypeCol = -1
    
    def translateGold(gold):
        return gold
    
    #read configuration
    with open(configFile, 'r') as conf:
        configuration = conf.read()
        exec(configuration)    

    #load labels from crowd
    if tableFormat:
        (crowdLabels, tIdxMap, tIdxs, K, nObjs) = loadCrowdTable(inputFile, scores)
    else:
        (crowdLabels, tIdxMap, tIdxs, K, nObjs) = loadCrowdLabels(inputFile, scores)
    
    #load gold labels if present
    gold, goldTypes = loadGold(goldFile, tIdxMap, nObjs, classLabels, goldTypeCol)
        
    gold = translateGold(gold)
    
    #map the training IDs to our local indexes
    if trainIds != None:
        trainIds = tIdxMap[trainIds,0].todense()
    
    return K,tableFormat,crowdLabels,gold,tIdxs,trainIds,outputFile,confMatFile,goldTypes             

def loadCombiner(configFile):
    K,tableFormat,crowdLabels,gold,origCandIds,trIdxs,outputFile,confMatFile,goldTypes = loadData(configFile)
    combiner = initFromConfig(configFile, K, tableFormat)
    return combiner, crowdLabels, gold, origCandIds, trIdxs,outputFile,confMatFile,goldTypes

def saveTargets(pT, tIdxs, outputFile):
    #write predicted class labels to file
    print 'writing results to file'
    
    print 'Posterior matrix: ' + str(pT.shape)
    tIdxs = np.reshape(tIdxs, (len(tIdxs),1))
    print 'Target indexes: ' + str(tIdxs.shape)
    
    np.savetxt(outputFile, np.concatenate([tIdxs, pT], 1))

def saveAlpha(alpha, nClasses, nScores, K, confMatFile):
    #write confusion matrices to file if required
    if not confMatFile is None:
        print 'writing confusion matrices to file'
        pi = alpha
        for l in range(nScores):
            pi[:,l,:] = np.divide(alpha[:,l,:], np.sum(alpha,1) )
        
        flatPi = pi.reshape(1, nClasses*nScores, K)
        flatPi = np.swapaxes(flatPi, 0, 2)
        np.savetxt(confMatFile, flatPi.reshape(K, nClasses*nScores), fmt='%1.3f')    

def runIbcc(configFile):
    K,tableFormat,crowdLabels,gold,tIdxs,_,outputFile,confMatFile,_ = loadData(configFile)
    combiner = initFromConfig(configFile, K, tableFormat)
        
    #combine labels
    pT = combiner.combineClassifications(crowdLabels, gold)

    if outputFile != None:
        saveTargets(pT, tIdxs, outputFile)
    if confMatFile != None:
        saveAlpha(combiner.alpha, combiner.nClasses, combiner.nScores, combiner.K, confMatFile)
    
    return pT, combiner
    
if __name__ == '__main__':

    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    runIbcc(configFile)
    
