'''
Created on 22 Apr 2014

@author: Edwin Simpson
'''

import ibcc
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, average_precision_score, auc
import matplotlib.pyplot as plt

def getAucs(testResults, labels, nClasses):
    '''
    Calculate the area under the ROC curve (called AUC) and 
    the area under the precision-recall curve, called the average precision (AP).
    '''
    auc_result = np.zeros(nClasses-1)
    ap = np.zeros(nClasses-1)
    for j in range(nClasses-1):
        y_true = labels[:,j]
        y_scores = testResults[:,j]
        #auc[j] = roc_auc_score(y_true, y_scores) #need scikit 0.14. 
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        auc_result[j] = auc(fpr, tpr)
        ap[j] = average_precision_score(y_true, y_scores)
                    
        diffs = tpr-fpr
        best = np.argmax(diffs)
        print 'The best threshold is ' + str(thresholds[best]) + ' at diff of ' + str(np.max(diffs))       
        print 'tpr: ' + str(tpr[best])
        print 'fpr: ' + str(fpr[best])
                    
    return (auc_result,ap)

def getAccMeasures(testResults, labels, nClasses, testIdxs=None):
    '''
    If testIdxs is not set, we compare against all data points. 
    If testIdxs is set, we ignore other data points, assuming they were part of the training set.
    '''
    
    testResults = testResults[testIdxs,1:]
    labels = labels[testIdxs].reshape(-1)
    
    labelMatrix = np.zeros((len(labels),nClasses-1)) 
    for j in range(1,nClasses):
        labelMatrix[labels==j,j-1] = 1
        
    print 'Evaluating the results using curves with classification thresholds.'
        
    auc, ap = getAucs(testResults, labelMatrix, nClasses)
    
    print 'Evaluating the results using the greedy classifications.'
    testResults = np.round(testResults)
    
    acc = 1 - np.sum(abs(labelMatrix-testResults), 0) / len(testIdxs)
    print 'acc: ' + str(acc)
    acc = acc[0] #each class column will be the same. Related to a weighted average of precision across all classes
    
    disagreement = abs(labelMatrix-testResults)
    fn = np.sum(disagreement[labels>=1,:],0)
    fp = np.sum(disagreement[labels==0,:],0)
    
    agreement = 1-disagreement
    tp = np.sum(agreement[labels>=1,:],0) 
    tn = np.sum(agreement[labels==0,:],0)
        
    recall = tp / (tp+fn)
    spec = tn/(fp+tn)
    prec = tp / (tp+fp)
        
    return (acc,recall,spec,prec,auc,ap)

def plotCumDist(pT,nClasses,foldTestIdxs,gold):    
    #sort probabilities in order
    #x values are the probabilities
    #y values are indices
    # have to select the highest y value for any duplicate x values
    
    for j in range(1,nClasses):#skip first class as this is used for uninteresting examples
        X = np.sort(pT[:,j]).reshape(-1)
        X = 1-X
        Y = np.array(range(len(X)))
        plt.figure()
        plt.plot(X,Y)
        plt.xlabel('Probability of Planet')
        plt.ylabel('No. Candidates')
        plt.title('Cumulative Distribution of All Candidates')
        
        plt.figure()
        X = X[gold==j]
        Y = np.array(range(len(X)))
        plt.plot(X,Y)
        plt.xlabel('Probability of Planet')
        plt.ylabel('No. Candidates')
        plt.title('Cumulative Distribution of Candidates with Confirmed Class Label ' + str(j))

def testIbccPerformance(combiner, crowdLabels, goldTraining, goldAll, testIdxs=None):
    #An alternative to the runIbcc function in the Ibcc module, which does not save the resulting 
    #classifications, but prints a performance analysis
    if testIdxs==None:
        trIdxs = np.argwhere(goldAll>-1)
        testIdxs = trIdxs.reshape(-1)
 
    pT = combiner.combineClassifications(crowdLabels, goldTraining)
    print 'Nu: ' + str(combiner.nu)  
    plotCumDist(pT,combiner.nClasses,testIdxs,goldAll)        
    
    #analyse the accuracy of the results
    acc,recall,spec,prec,auc,ap = getAccMeasures(pT,goldAll,combiner.nClasses,testIdxs) 
    return (acc,recall,spec,prec,auc,ap)

def printResults(meanAcc,meanRecall,meanSpecificity,meanPrecision,meanAuc,meanAp):
    print '---For each class separately---'
    
    print 'Mean Recall: ' + str(meanRecall)
    print 'Mean Specificity: ' + str(meanSpecificity)
    print 'Mean Precision: ' + str(meanPrecision)
    print 'Mean AUC: ' + str(meanAuc)
    print 'Mean AP: ' + str(meanAp)
    
    print '--- Means across all classes ---'
    
    #unweighted average across classes
    nClasses = len(meanAuc)#combiner.nClasses-1
    meanAuc = np.sum(meanAuc)/nClasses
    meanAp = np.sum(meanAp)/nClasses
    meanRecall = np.sum(meanRecall)/nClasses
    meanSpecificity = np.sum(meanSpecificity)/nClasses
    meanPrecision = np.sum(meanPrecision)/nClasses  
    
    print 'Mean Accuracy: ' + str(meanAcc)
    print 'Mean Recall: ' + str(meanRecall)
    print 'Mean Specificity: ' + str(meanSpecificity)
    print 'Mean Precision: ' + str(meanPrecision)
    print 'Mean AUC: ' + str(meanAuc)
    print 'Mean AP: ' + str(meanAp)

def testUnsupervised(configFile):
    # no training data, test all points we have true labels for
    combiner, crowdLabels, gold, _,_,_ = ibcc.loadData(configFile)
    
    goldTr = np.zeros(len(gold)) -1 
    
    (acc,recall,spec,prec,auc,ap) = testIbccPerformance(combiner, crowdLabels, goldTr, gold)
    printResults(acc,recall,spec,prec,auc,ap)
    
def testSupervised(configFile):
    #supply all training data. The metrics will be unfair
    combiner, crowdLabels, gold, _,_,_ = ibcc.loadData(configFile)
    
    (acc,recall,spec,prec,auc,ap) = testIbccPerformance(combiner, crowdLabels, gold, gold)
    printResults(acc,recall,spec,prec,auc,ap)

def testXValidation(nFolds, configFile):
    '''
    Run n-fold cross validation
    '''
    if nFolds==0:
        testSupervised(configFile)
        return
    elif nFolds==1:
        testUnsupervised(configFile)
        return
    
    #load the data
    combiner, crowdLabels, gold, _,_,_ = ibcc.loadData(configFile)
    
    meanAcc = 0
    meanRecall = np.zeros((1,combiner.nClasses-1))
    meanSpecificity = np.zeros((1,combiner.nClasses-1))
    meanPrecision = np.zeros((1,combiner.nClasses-1))
    meanAuc = np.zeros(combiner.nClasses-1)
    meanAp = np.zeros(combiner.nClasses-1)
    
    #split the data into nFolds different partitions
    trIdxs = np.argwhere(gold>-1)
    
    kf = KFold(len(trIdxs), n_folds=nFolds, indices=False)      
    
    #for each partition, run IBCC
    #any unlabelled data is included and is not split
    for trMask, testMask in kf:
        goldPartition = np.zeros(len(gold)) -1 
        goldPartition[trIdxs[trMask]] = gold[trIdxs[trMask]]
        foldTestIdxs = trIdxs[testMask].reshape(-1)
        
        acc,recall,spec,prec,auc,ap = testIbccPerformance(combiner, crowdLabels, goldPartition, gold, foldTestIdxs)  
        
        #save to overall summary
        meanAcc += acc
        meanRecall += recall,
        meanSpecificity += spec
        meanPrecision += prec
        meanAuc += auc
        meanAp += ap
                
    meanAcc /= nFolds
    meanRecall /= nFolds
    meanSpecificity /= nFolds
    meanPrecision /= nFolds
    meanAuc /= nFolds
    meanAp /= nFolds
    #display summary of results across all folds
    printResults(meanAcc,meanRecall,meanSpecificity,meanPrecision,meanAuc,meanAp)
    
testUnsupervised('./python/config/ph_2.py')    
#testXValidation('./config/my_project.py')
#testXValidation('./config/thesis_synth.py')