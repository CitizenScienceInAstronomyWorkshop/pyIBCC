'''
Created on 29 Sep 2014

@author: edwin
'''

import pickle, logging
import numpy as np
from scipy.sparse import coo_matrix

class DataHandler(object):
    '''
    classdocs
    '''

    crowdlabels = None
    table_format = False
    
    targetidxmap = None
    targetidxs = None
    max_targetid = 0
    trainids = None
    
    scores = None
    
    K = 0
    N = 0
    
    goldlabels = None
    goldsubtypes = None
    
    output_file = None
    confmat_file = None
    input_file = None
    gold_file = None

    nclasses = 2
    scores = None
    nu0 = None
    alpha0 = None         

    def __init__(self):
        '''
        Constructor
        '''
        
    def loadCrowdLabels(self, scores):   
        '''
        Loads labels from crowd in sparse list format, i.e. 3 columns, classifier ID,
        object ID, score.
        '''
        pyFileExists = False
        try:
            with open(self.input_file+'.dat','r') as inFile:
                crowdLabels, self.targetidxs, K = pickle.load(inFile)
                pyFileExists = True
        except Exception:
            logging.info('Will try to load a CSV file...')
        
            crowdLabels = np.genfromtxt(self.input_file, delimiter=',', \
                                    skip_header=1,usecols=[0,1,2])
    
            self.targetidxs, crowdLabels[:,1] = np.unique(crowdLabels[:,1],return_inverse=True)
            kIdxs, crowdLabels[:,0] = np.unique(crowdLabels[:,0],return_inverse=True)
            K = len(kIdxs)
            
        unmappedScores = np.round(crowdLabels[:,2])
        for i,s in enumerate(scores):
            crowdLabels[(unmappedScores==s),2] = i
        
        maxT = np.max(self.targetidxs)
        blanks = np.zeros(len(self.targetidxs))
        idxList = range(len(self.targetidxs))
        
        tIdxMap = coo_matrix(( idxList, (self.targetidxs,blanks)), shape=(maxT+1,1) )
        tIdxMap = tIdxMap.tocsr()
        
        if not pyFileExists:
            try:
                with open(self.input_file+'.dat', 'wb') as outFile:
                    pickle.dump((crowdLabels,self.targetidxs,K), outFile)
            except Exception:
                logging.error('Could not save the input data as a Python object file.')
        
        self.crowdlabels = crowdLabels
        self.targetidxmap = tIdxMap
        self.K = K
        self.N = len(self.targetidxs)
        self.max_targetid = np.max(self.targetidxs)
        
    def loadCrowdTable(self, scores):
        '''
        Loads crowd labels in a table format
        '''
        unmappedScores = np.round(np.genfromtxt(self.input_file, delimiter=','))
        
        #targetidxs, crowdlabels[:,1] = np.unique(crowdlabels[:,1],return_inverse=True)
        #kIdxs, crowdlabels[:,0] = np.unique(crowdlabels[:,0],return_inverse=True)
        self.K = unmappedScores.shape[1]
        self.N = unmappedScores.shape[0]
        self.targetidxs = range(self.N)
                
        self.table_format_flag = np.zeros((self.N,self.K))
        for i,s in enumerate(scores):
            self.table_format_flag[unmappedScores==s] = i
        
        maxT = np.max(self.targetidxs)
        blanks = np.zeros(len(self.targetidxs))
        
        tIdxMap = coo_matrix(( self.targetidxs, (self.targetidxs,blanks)), shape=(maxT+1,1) )
        self.targetidxmap = tIdxMap.tocsr()
        self.max_targetid = self.N
        
    def loadGold(self, classLabels=None, secondaryTypeCol=-1):   
        
        import os.path
        if not os.path.isfile(self.gold_file):
            logging.warning('No gold labels found.')
            self.goldlabels = np.zeros(self.N) -1
        
        if secondaryTypeCol>-1:
            useCols=[0,1,secondaryTypeCol]
        else:
            useCols=[0,1]
            
        try:
            gold = np.genfromtxt(self.gold_file, delimiter=',', skip_header=0,usecols=useCols,invalid_raise=True)
        except Exception:
            gold = np.genfromtxt(self.gold_file, delimiter=',', skip_header=0)
            
        if np.any(np.isnan(gold[0])): #skip header if necessary
            gold = gold[1:,:]
        logging.debug("gold shape: " + str(gold.shape))
        
        if len(gold.shape)==1 or gold.shape[1]==1: #position in this list --> id of data point
            goldLabels = gold
        else: # sparse format: first column is id of data point, second column is gold label value
            
            #map the original idxs to local idxs
            valid_gold_idxs = np.argwhere(gold[:,0]<=self.max_targetid)
            gold = gold[valid_gold_idxs.reshape(-1),:]
            goldIdxs = gold[:,0]
            goldIdxs = self.targetidxmap[goldIdxs,0].todense()
            
            #create an array for gold for all the objects/data points in this test set
            goldLabels = np.zeros(self.N) -1
            goldLabels[goldIdxs] = gold[:,1]
            
            #if there is secondary type info, create a similar array for this
            if secondaryTypeCol>-1:
                goldTypes = np.zeros(self.N)
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

        self.goldlabels = goldLabels        
        if secondaryTypeCol>-1:
            self.goldsubtypes = goldTypes
        
    def loadData(self, configFile):
        
        #Defaults that will usually be overwritten by project config
        tableFormat = False
        #columns in input file:
        # 0 = agent/worker/volunteer ID
        # 1 = object ID
        # 2 = scores given to the object
        
        #columns in gold file
        # 0 = object ID
        # 1 = class label
        scores = np.array([3, 4])
        
        classLabels = None
      
        trainIds = None #IDs of targets that should be used as training data. Optional 
        
        #column index of secondary type information about the data points stored in the gold file. 
        #-1 means no such info
        goldTypeCol = -1
        
        def translateGold(gold):
            return gold
        
        outputFile = './output/output.csv'
        confMatFile = None#'./output/confMat.csv'
        inputFile = './data/input.csv'
        goldFile = './data/gold.csv'
        
        nClasses = 2
        nu0 = np.array([50.0, 50.0])
        alpha0 = np.array([[2, 1], [1, 2]])     
        
        #read configuration
        with open(configFile, 'r') as conf:
            configuration = conf.read()
            exec(configuration)
            
        self.output_file = outputFile
        self.confmat_file = confMatFile
        self.input_file = inputFile
        self.gold_file = goldFile
        self.scores = scores
        self.nclasses = nClasses
        self.nu0 = nu0
        self.alpha0 = alpha0
    
        #load labels from crowd
        if tableFormat:
            self.loadCrowdTable(scores)
        else:
            self.loadCrowdLabels(scores)
        
        #load gold labels if present
        self.loadGold(classLabels, goldTypeCol)
            
        self.goldlabels = translateGold(self.goldlabels)
        
        #map the training IDs to our local indexes
        if trainIds != None:
            self.trainids = self.targetidxmap[trainIds,0].todense()
        
        self.table_format = tableFormat
            
    def saveTargets(self, pT):
        #write predicted class labels to file
        logging.info('writing results to file')
        logging.debug('Posterior matrix: ' + str(pT.shape))
        tIdxs = np.reshape(self.targetidxs, (len(self.targetidxs),1))
        logging.debug('Target indexes: ' + str(tIdxs.shape))    
        np.savetxt(self.output_file, np.concatenate([tIdxs, pT], 1))
    
    def save_pi(self, alpha, nClasses, nScores):
        #write confusion matrices to file if required
        if self.confmat_file is None:
            return
    
        logging.info('writing confusion matrices to file')
        pi = np.zeros(alpha.shape)
        for l in range(nScores):
            pi[:,l,:] = alpha[:,l,:]/np.sum(alpha,1)
        
        flatPi = pi.reshape(1, nClasses*nScores, alpha.shape[2])
        flatPi = np.swapaxes(flatPi, 0, 2)
        np.savetxt(self.confmat_file, flatPi.reshape(alpha.shape[2], nClasses*nScores), fmt='%1.3f')    
        
        