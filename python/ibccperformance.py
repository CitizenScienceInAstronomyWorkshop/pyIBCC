'''
Created on 22 Apr 2014

@author: Edwin Simpson
'''

import ibcc
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, average_precision_score, auc
import matplotlib.pyplot as plt
import matplotlib.legend as leg

figure1 = 1
figure2 = 2
figure3 = 3
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
        print  'The best threshold is ' + str(thresholds[best]) + ' at diff of ' + str(np.max(diffs))       
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
        
    if tp+fn==0:
        print "recall unknown"
        recall = 0
    else:
        recall = tp / (tp+fn)
        
    if fp+tn==0:
        print "specificity unknown"
        spec = 0
    else:
        spec = tn/(fp+tn)
        
    if tp+fp==0:
        "precision unknown"
        prec = 0
    else:
        prec = tp / (tp+fp)
        
    print 'Evaluating the results using curves with classification thresholds.'
    if fp+tn==0 or tp+fn==0:
        print 'Incomplete test labels, cannot evaluate AUC'
        auc = np.zeros(nClasses-1)
        ap = np.zeros(nClasses-1)
    else:
        auc, ap = getAucs(testResults, labelMatrix, nClasses)        
        
    return (acc,recall,spec,prec,auc,ap)

def plotCumDist(pT,nClasses,testIdxs,gold,label=""):    
    #sort probabilities in order
    #x values are the probabilities
    #y values are indices
    # have to select the highest y value for any duplicate x values
    #skip first class as this is used for uninteresting examples
    for j in range(1,nClasses):
        X = np.sort(pT[testIdxs,j]).reshape(-1)
        Y = - np.array(range(len(X))) + len(X) 
        plt.figure(figure1)
        plt.plot(X,Y, label=label)
        plt.legend(loc='lower center')
        plt.xlabel('Probability of Class ' + str(j))
        plt.ylabel('No. Candidates')
        plt.title('Cumulative Distribution of All Candidates')
        plt.hold(True)
        
        testIdxs_j = testIdxs[gold[testIdxs]==j]
        X = np.sort(pT[testIdxs_j,j]).reshape(-1)
        Y = - np.array(range(len(X))) + len(X) 
        plt.figure(figure2)
        plt.plot(X,Y, label=label)
        plt.legend(loc='lower center')
        plt.xlabel('Probability of Class ' + str(j))
        plt.ylabel('No. Candidates')
        plt.title('Cumulative Distribution of Candidates of Class ' + str(j) )
        plt.hold(True)

        testIdxs_j = testIdxs[gold[testIdxs]==j]
        X = np.sort(pT[testIdxs_j,j]).reshape(-1)
        Y = (- np.array( range(len(X)) ) + float(len(X)) )
        Y = np.float32(Y)/float(len(X))
        plt.figure(figure3)
        plt.plot(X,Y, label=label)
        plt.legend(loc='lower center')
        plt.xlabel('Probability of Class ' + str(j))
        plt.ylabel('No. Candidates')
        plt.title('Normalised Cumulative Distribution of Candidates of Class ' + str(j) )
        plt.hold(True)
        
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

def testIbccPerformance(combiner, crowdLabels, goldTraining, goldAll, \
                        testIdxs=None, label="", eval=True):
    #An alternative to the runIbcc function in the Ibcc module, which does not save the resulting 
    #classifications, but prints a performance analysis
    if testIdxs==None:
        testIdxs = np.argwhere(np.bitwise_and(goldAll>-1,goldTraining==-1))
        testIdxs = testIdxs.reshape(-1)
 
    print ' No. test indexes = ' + str(len(testIdxs))
 
    pT = combiner.combineClassifications(crowdLabels, goldTraining)
    print 'Nu: ' + str(combiner.nu)  
    
    #analyse the accuracy of the results
    if not eval:
        return pT, testIdxs
    
    plotCumDist(pT,combiner.nClasses,testIdxs,goldAll, label)        
    acc,recall,spec,prec,auc,ap = \
        getAccMeasures(pT,goldAll,combiner.nClasses,testIdxs)
  
    return (pT,testIdxs,acc,recall,spec,prec,auc,ap) 

def testUnsupervised(configFile, label, eval=True):
    # no training data, test all points we have true labels for
    combiner, crowdLabels, gold, _,_,_,_,goldTypes = ibcc.loadCombiner(configFile)
    
    goldTr = np.zeros(len(gold)) -1 
    
    if eval:
        pT,testIdxs,acc,recall,spec,prec,auc,ap = testIbccPerformance(combiner, crowdLabels, goldTr,\
                                        gold, label=label, eval=eval)
        printResults(acc,recall,spec,prec,auc,ap)
    else:
        pT,testIdxs = testIbccPerformance(combiner, crowdLabels, goldTr,\
                                        gold, label=label, eval=eval)
    return pT,goldTypes,testIdxs,gold
    
def testSupervised(configFile, label, trIds=None, goldTr=None, eval=True):
    #supply all training data. The metrics will be unfair
    combiner, crowdLabels, gold, origCandIds, trIdxs,_,_,goldTypes = ibcc.loadCombiner(configFile)

    if trIdxs != None and goldTr==None:
        goldTr = np.zeros(len(gold)) -1
        #trIds = np.sort(np.array(trIds)) #original Ids -- find their indexes in tIdxs
        #trIdxs = np.searchsorted(trIds, tIdxs)
        goldTr[trIdxs] = gold[trIdxs]
    elif goldTr==None:
        goldTr = gold
    
    if eval:
        pT,testIdxs,acc,recall,spec,prec,auc,ap = testIbccPerformance(combiner, crowdLabels, \
                                        goldTr, gold, label=label, eval=eval)
        printResults(acc,recall,spec,prec,auc,ap)
    else:
        pT,testIdxs = testIbccPerformance(combiner, crowdLabels, \
                                        goldTr, gold, label=label, eval=eval)
    return pT,goldTypes,testIdxs,gold,crowdLabels,origCandIds

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
    combiner, crowdLabels, gold, _,_,_,_ = ibcc.loadCombiner(configFile)
    
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
        
        _,_,acc,recall,spec,prec,auc,ap = testIbccPerformance(combiner, crowdLabels, goldPartition, gold, foldTestIdxs, label=configFile)  
        
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
 
#testXValidation('./config/my_project.py')
