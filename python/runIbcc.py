
'''
@author: Edwin Simpson
'''

import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix
from scipy.special import psi
from astropy.io import ascii

'''
CHANGE THE LINE BELOW IF YOU WANT TO USE A DIFFERENT CONFIG FILE!
'''
configFile = './config/my_project.py'

class Ibcc:

    def expecLnKappa(self):
        #Posterior Hyperparams
        sumET = np.sum(self.ET, 0)
        
        for j in range(self.nClasses):
            self.nu[j] = self.nu0[j] + sumET[j]
                   
        #Expected Log         
        sumNu = np.sum(self.nu)
        self.lnKappa = psi(self.nu) - psi(sumNu)
       
    def expecLnPi(self, C):
        #Posterior Hyperparams
        for j in range(self.nClasses):
            for l in range(self.nScores):
                counts = C[l].T * self.ET[:,j]
                self.alpha[j,l,:] = self.alpha0[j,l] + counts
        
        #Expected Log
        sumAlpha = np.sum(self.alpha, 0)
        psiSumAlpha = psi(sumAlpha)
        psiSumAlpha = np.tile(psiSumAlpha, (self.nClasses, 1, 1))
        
        self.lnPi = psi(self.alpha) - psiSumAlpha        

    def expecT(self, crowdLabels):
        
        joint = np.zeros((self.nObjs, self.nClasses))
        
        for j in range(self.nClasses):
            lnPi_j = self.lnPi[j,:,:]
            likelihood_j = np.sum(lnPi_j[crowdLabels[:,2], crowdLabels[:,0]])
            joint[:,j] = likelihood_j + self.lnKappa[j]
            
        joint = np.exp(joint)
        norma = np.sum( joint, axis=1 )
        
        for j in range(self.nClasses):
            self.ET[:,j] = np.divide(joint[:,j], norma)

    def initParams(self):
        
        print '\n%s parameters...' % lang_dict(lang)['initial'].title()
        
        self.nu = deepcopy(self.nu0)
         
        #first copy alpha for all K citizen scientists
        #self.alpha = self.alpha0[:,:,np.newaxis]    
        print 'Alpha0:' 
        print self.alpha0
        self.alpha = self.alpha0[:,:,np.newaxis]
        self.alpha = np.repeat(self.alpha, self.K, axis=2)#(self.nClasses,1,self.K))
        
    def initT(self, trainT):
        
        print '\n%s results: ' % lang_dict(lang)['initial'].title()
        #print trainT
        self.nObjs = trainT.shape[0]
        self.ET = np.zeros((self.nObjs,self.nClasses))
    
        for i in range(len(trainT)):
            t = trainT[i,0]
            if t>-1:
                self.ET[i,t] = 1
    
    def getET(self):
        return self.ET
    
    def preprocessCrowdLabels(self, crowdLabels):
        
        C = {}
        
        for l in range(self.nScores):
            
            lIdxs = np.where(crowdLabels[:,2]==l)[0]
                        
            data = np.array(np.ones((len(lIdxs),1))).reshape(-1)
            rows = np.array(crowdLabels[lIdxs,1]).reshape(-1)
            cols = np.array(crowdLabels[lIdxs,0]).reshape(-1)
                        
            Cl = coo_matrix((data,(rows,cols)), shape=(self.nObjs, self.K))
            C[l] = Cl
    
        return C
    
    def combineClassifications(self, crowdLabels):
        
        print '\nCombining classifications ...'
        
        maxNoIts = 500
        convThreshold = 0.0001
        converged = False
        nIts = 0
        
        crowdLabels = crowdLabels.astype(int)
        C = self.preprocessCrowdLabels(crowdLabels)
        
        while not converged:
            oldET = self.ET
            
            #update params
            self.expecLnKappa()
            self.expecLnPi(C)
            
            #update targets
            self.expecT(crowdLabels)
            
            change = np.sum(np.sum(np.absolute(oldET - self.ET)))
            
            #check convergence
            if nIts>=maxNoIts or change<convThreshold:
                converged = True
            nIts+=1
            print 'IBCC iteration ' + str(nIts) + ' absolute change was ' + str(change)
                
        endaffix = 's' if nIts > 1 else ''
        print 'IBCC converged in ' + str(nIts) + ' iteration' + endaffix + '.'
        
    def __init__(self, nClasses, nScores, alpha0, nu0, K, trainT):
        self.nClasses = nClasses
        self.nScores = nScores
        self.alpha0 = alpha0
        self.nu0 = nu0
        self.K = K
        self.initT(trainT)        
        self.initParams()

def loadCrowdLabels(inputFile, scores):   

    print '\nLoading data for crowd labels from %s' % inputFile

    #crowdLabels = np.genfromtxt(inputFile, delimiter=',',comments='#')
    data = ascii.read(inputFile)

    # subject indices
    tIdxs = np.unique(data['subjectID'])

    # user indices
    kIdxs = np.unique(data['userID'])
    K = len(kIdxs)

    crowdLabels = np.zeros((len(data),len(data.columns)),dtype=int)

    for idx,row in enumerate(data):
        crowdLabels[idx,0] = np.where(kIdxs==row['userID'])[0][0]
        crowdLabels[idx,1] = np.where(tIdxs==row['subjectID'])[0][0]
        crowdLabels[idx,2] = np.where(scores==row['score'])[0][0]
    
    return (crowdLabels, tIdxs, K)
    
def loadGold(goldFile):   
    gold = np.genfromtxt(goldFile, delimiter=',', comments='#')
    return gold

def lang_dict(lang):
    ld = {}
    if lang == 'en-us':
        ld['initial'] = 'initializing'
    if lang == 'en-gb':
        ld['initial'] = 'initialising'

    return ld

############################################################
# Running program from command line begins here
############################################################

#read configuration
with open(configFile, 'r') as conf:
    configuration = conf.readlines()
    for line in configuration:
        exec(line)

#load labels from crowd
(crowdLabels, tIdxs, K) = loadCrowdLabels(inputFile, scores)

print 'Objects in test set: ' + str(tIdxs)

#load gold labels if present
import os.path
if os.path.isfile(goldFile):
    gold = loadGold(goldFile)
    goldIdxs = gold[:,0]
    for i in range(gold.shape[0]):
        tIdx = np.where(tIdxs==gold[i,0])
        if len(tIdx[0]) > 0:
            goldIdxs[i] = tIdx[0][0]
        else:
            print 'Gold labelled item was not in test set ' + str(gold[i,:])
        
    goldSorted = np.zeros((len(tIdxs),1)) -1
    
    for i in range(len(goldIdxs)):
        goldSorted[goldIdxs[i]] = gold[i,1] 
        
    gold = goldSorted
else:
    print '\nNo gold label data found'
    gold = np.zeros((len(tIdxs), 1)) -1

#initialise combiner
combiner = Ibcc(nClasses, nScores, alpha0, nu0, K, gold)

#combine labels
combiner.combineClassifications(crowdLabels)

#write predicted class labels to file
pT = combiner.getET()
print '\nWriting results to file (needs fixing for multi-class case).'

print 'pT shape:    ' + str(pT.shape)
tIdxs = np.reshape(tIdxs, (len(tIdxs),1))
print 'tIDxs shape: ' + str(tIdxs.shape)

np.savetxt(outputFile, np.concatenate([tIdxs, pT], 1))

#write confusion matrices to file if required
if not confMatFile is None:
    print '\nWriting confusion matrices to file.'
    pi = combiner.alpha
    for l in range(nScores):
        pi[:,l,:] = np.divide(combiner.alpha[:,l,:], np.sum(combiner.alpha,1) )
    
    flatPi = pi.reshape(1, nClasses*nScores, K)
    flatPi = np.swapaxes(flatPi, 0, 2)
    np.savetxt(confMatFile, flatPi.reshape(K, nClasses*nScores), fmt='%1.3f')
