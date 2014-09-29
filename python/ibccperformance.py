'''
Created on 22 Apr 2014

@author: Edwin Simpson
'''

import ibcc
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, average_precision_score, auc
import matplotlib.pyplot as plt
import os, logging
from copy import deepcopy

class Evaluator(object):

    figure1 = 1
    figure2 = 2
    figure3 = 3
    
    figure7 = 7
    figure8 = 8
    
    skill_fig = 11
    
    outputdir = "/home/edwin/git/pyIBCC/output/ph/plots/"
    
    combiner = None
    crowdLabels = []
    pT = []
    
    testIdxs = []
    trIdxs = []
    
    gold = [] #all gold labels, including test data
    goldTr = [] #gold labels for training
    goldTypes = [] # sub-types of the classes
    disc_gold_types = []
    
    secondary_type_cats = [2, 3, 4, 5, 6, 7, 10, 16]
    
    orig_cand_ids = []
    
    configfile = ""
    algolabel = ""
    datalabel = ""
    
    # if there are multiple positive classes, e.g. for each difficulty level, merge into one
    merge_all_pos = True 
    pT_premerge = [] #the raw data before merging is stored here
    
    def __init__(self, configfile, algolabel, datalabel, f1=1, f2=2, f3=3, f7=7, f8=8, skill_fig=11):
        self.configfile = configfile
        self.algolabel = algolabel
        self.datalabel = datalabel
        logging.info("Starting EXPT: " + datalabel + ": " + algolabel)
        
        self.figure1 = f1
        self.figure2 = f2
        self.figure3 = f3
        self.figure7 = f7
        self.figure8 = f8
        self.skill_fig = skill_fig
    
    def write_img(self, label, figureobj):
        #set current figure
        plt.figure(figureobj)
        
        if not os.path.isdir(self.outputdir):
            os.mkdir(self.outputdir)
        
        plt.savefig(self.outputdir+label+'.png', bbox_inches='tight', \
                    pad_inches=0, transparent=True, dpi=96)
    
    def getAucs(self, testResults, goldmatrix, nClasses):
        '''
        Calculate the area under the ROC curve (called AUC) and 
        the area under the precision-recall curve, called the average precision (AP).
        '''
        auc_result = np.zeros(nClasses-1)
        ap = np.zeros(nClasses-1)
        for j in range(nClasses-1):
            y_true = goldmatrix[:,j]
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
    
    def getAccMeasures(self, nClasses):
        '''
        If testIdxs is not set, we compare against all data points. 
        If testIdxs is set, we ignore other data points, assuming they were part of the training set.
        '''
        
        pT_test_pos = self.pT[self.testIdxs,1:]
        labels = self.gold[self.testIdxs].reshape(-1)
        
        labelMatrix = np.zeros((len(labels),nClasses-1)) 
        for j in range(1,nClasses):
            labelMatrix[labels==j,j-1] = 1
                
        print 'Evaluating the results using the greedy classifications.'
        testResults = np.round(pT_test_pos)
        
        acc = 1 - np.sum(abs(labelMatrix-testResults), 0) / len(self.testIdxs)
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
            
        nfiltered = tp + fp
        filter_rate = nfiltered/(nfiltered+tn+fn)
            
        print 'Evaluating the results using curves with classification thresholds.'
        if fp+tn==0 or tp+fn==0:
            print 'Incomplete test labels, cannot evaluate AUC'
            auc = np.zeros(nClasses-1)
            ap = np.zeros(nClasses-1)
        else:
            auc, ap = self.getAucs(testResults, labelMatrix, nClasses)        
            
        return acc,recall,spec,prec,auc,ap,nfiltered,filter_rate
    
    def plotCumDist(self, nClasses):    
        #sort probabilities in order
        #x values are the probabilities
        #y values are indices
        # have to select the highest y value for any duplicate x values
        #skip first class as this is used for uninteresting examples
        
        gold_test = self.gold[self.testIdxs]
        
        for j in range(1,nClasses):
            X = np.sort(self.pT[self.testIdxs,j]).reshape(-1)
            Y = - np.array(range(len(X))) + len(X) 
            plt.figure(self.figure1)
            plt.plot(X,Y, label=self.algolabel)
            plt.legend() # loc='lower center'
            plt.xlabel('Probability of Class ' + str(j))
            plt.ylabel('No. Candidates')
            plt.title('Cumulative Distribution of All Candidates')
            plt.hold(True)
            plt.grid(which="minor")
            
            self.write_img("cumdist_all_"+self.datalabel, self.figure1)
            
            testIdxs_j = self.testIdxs[gold_test==j]
            
            X = np.sort(self.pT[testIdxs_j,j]).reshape(-1)
            Y = - np.array(range(len(X))) + len(X) 
            plt.figure(self.figure2)
            plt.plot(X,Y, label=self.algolabel)
            plt.legend()
            plt.xlabel('Probability of Class ' + str(j))
            plt.ylabel('No. Candidates')
            plt.title('Cumulative Distribution of ' + self.datalabel)#Candidates of Class ' + str(j) )
            plt.hold(True)
            plt.grid(which="minor")
            
            self.write_img("cumdist_" + str(j) + "_" + self.datalabel, self.figure2)
    
            X = np.sort(self.pT[testIdxs_j,j]).reshape(-1)
            Y = (- np.array( range(len(X)) ) + float(len(X)) )
            Y = np.float32(Y)/float(len(X))
            plt.figure(self.figure3)
            plt.plot(X,Y, label=self.algolabel)
            plt.legend()
            plt.xlabel('Probability of Class ' + str(j))
            plt.ylabel('No. Candidates')
            plt.title('Normalised Cumulative Distribution of ' + self.datalabel)#Candidates of Class ' + str(j) )
            plt.hold(True)
            plt.grid(which="minor")
            
            self.write_img("normcumdist_" + str(j) + "_" + self.datalabel, self.figure3)
            
    def printResults(self, meanAcc,meanRecall,meanSpecificity,meanPrecision,meanAuc,meanAp,nfiltered,filter_rate):
        
        nClasses = len(meanAuc)#combiner.nClasses-1
        
        if nClasses > 1:
            print '---For each class separately---'
            
            print 'Recall: ' + str(meanRecall)
            print 'Specificity: ' + str(meanSpecificity)
            print 'Precision: ' + str(meanPrecision)
            print 'AUC: ' + str(meanAuc)
            print 'AP: ' + str(meanAp)
            print "No. data points marked positive: " + str(nfiltered)
            print "Fraction marked as positive: " + str(filter_rate)
        
            print '--- Means across all classes ---'
        
        #unweighted average across classes
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
        print "Mean no. data points marked positive: " + str(nfiltered)
        print "Mean fraction marked as positive: " + str(filter_rate)
        
    def plotRecallByRadius(self, seqno):
    
        cols = ['r','y','b','m','g']
        tvals = self.secondary_type_cats[1:]
        width = np.zeros(len(tvals), dtype=np.float)
        start = 2
        startseq = np.concatenate((np.array([2], dtype=np.float),np.array(tvals,dtype=np.float)))[0:len(tvals)]
        
        totalNs = np.zeros(len(tvals))
        recall = np.zeros(len(tvals))
        cumRecall = np.zeros(len(tvals))
        
        for i,t in enumerate(tvals):
            #find the boolean array of test indexes indicating this type
            thisType = np.bitwise_and(self.goldTypes<t, self.goldTypes>=start)
            thisType = thisType[self.testIdxs]
            #Find the test indexes of the current type 
            idxs = self.testIdxs[thisType]
            if len(idxs)==0:
                logging.warning("No test idxs found with radius between " + str(start) + ' and ' + str(t))
                continue
             
            width[i] = t-start       
            if i+1<len(tvals) and tvals[i+1]>t:
                start = t
            
            pT_t = self.pT[idxs,:]
            greedyLabels = np.round(pT_t[:,1])
            gold_t = self.gold[idxs]
            pos_t = gold_t==1
            
            tp = float(np.sum(greedyLabels[pos_t]==1))
            fn = float(np.sum(greedyLabels[pos_t]<1))
            recall[i] = tp/(tp+fn)
            
            totalNs[i] = tp+fn
            
            print "Recall for type " + str(t) + " is " + str(recall[i])
            cumRecall[i] = np.sum(np.multiply(recall[0:i+1], totalNs[0:i+1]))/np.sum(totalNs[0:i+1])
            #print "Cum Recall for type " + str(t) + " is " + str(cumRecall[i])
                    
        startseq = startseq + seqno*width/6.0
                    
        plt.figure(self.figure7)
        plt.bar(startseq, recall, label=self.algolabel, width=width/5, color=cols[seqno])
        plt.legend(loc="lower right")
        plt.xlabel('Radius/Earth Radii')
        plt.ylabel('Recall')
        plt.title('Recall of ' + self.datalabel + ' by Radius')
        plt.hold(True)
        plt.grid(which="minor")
        
        self.write_img("recallbyradius_" + self.datalabel, self.figure7)
        
        plt.figure(self.figure8)
        plt.xlabel('Radius/Earth Radii')
        plt.ylabel('Recall')
        plt.title('Recall of ' + self.datalabel + ' with Minimum Radius')
        plt.plot(tvals ,cumRecall, label=self.algolabel)
        plt.legend(loc="lower right")
        plt.hold(True)
        plt.grid(which="minor")
        
        self.write_img("recallbyradiusgreater_" + self.datalabel, self.figure8)
    
    def plot_skill_distribution(self):
        '''
        Plot Skill distribution by class, if more than one positive class.
        '''
        plt.figure(self.skill_fig)
        plt.xlabel("Volunteers sorted by detection rate")
        plt.ylabel("Detection Rate p(c=1|t=1)")
        plt.title("Distribution of Volunteers' Detection Rate of " + self.datalabel)
        
        #calculate skill levels
        alphasum = np.sum(self.combiner.alpha, 1).reshape((self.combiner.nClasses,1,self.combiner.K))
        pi = self.combiner.alpha / alphasum
        skill = pi[1:,0,:] #assumes score 0 is a positive classification
        skill = skill.reshape((skill.shape[0],self.combiner.K))
        sorted_skill = np.sort(skill, 1)
        
        xidx = np.arange(self.combiner.K, dtype=np.float)
        norm_xidx = xidx/np.float(self.combiner.K)
        
        if self.combiner.nClasses>2:
            cats = self.secondary_type_cats
        else:
            cats = [self.secondary_type_cats[0],self.secondary_type_cats[-1]]
        
        for i in range(sorted_skill.shape[0]):
            logging.info("Max skill: " + str(sorted_skill[i,-1]))
            plt.plot(norm_xidx, sorted_skill[i,:], label=str(cats[i])+"-"+str(cats[i+1])+", " +self.algolabel)
            plt.hold(True)
        plt.legend(loc="lower right")
        plt.grid(which="minor")
        
        self.write_img("volunteeracc_"+self.datalabel, self.skill_fig)
        
        #lnpi = self.combiner.lnPi
        #lnproportions = self.combiner.lnNu.reshape((self.combiner.nClasses,0,1))
        #pt_given_c = np.exp(lnpi[1:,0,:]+lnproportions[1:,:,:]) / np.exp(np.sum(lnpi[:,0,:]+lnproportions, axis=0))
        
    def printMajorityVoteByType(self):
        tVals = self.secondary_type_cats[1:]
        start = 2      
           
        for i,t in enumerate(tVals):
            
            thisType = np.bitwise_and(self.goldTypes<t, self.goldTypes>=start)
            idxs = self.testIdxs[thisType[self.testIdxs]]
            pT_t = self.pT[idxs,:]
            greedyLabels = np.round(pT_t[:,1])
            start = t
            hits = np.zeros(len(thisType))
            seenBy = np.zeros(len(thisType))
            for l in range(self.crowdLabels.shape[0]):
                idx = int(self.crowdLabels[l,1])
                if self.crowdLabels[l,2]==0:
                    hits[idx] += float(1)
                seenBy[idx] += float(1)
                
            nThisType = float(len(hits[thisType]))
            nWithPos = np.sum(hits[thisType]>0) 
            print nWithPos
            print nThisType
            print nWithPos/nThisType    
            majorities = np.divide(hits[thisType], seenBy[thisType])>0.5
            nMajorities = np.sum(majorities)
            print nMajorities  
            print nMajorities / nThisType
            if i==5:
                bigPlanetIds = idxs[greedyLabels==0]
                 
        return bigPlanetIds   
    
    def weightedVoteSchwamb(self):
        #this is incomplete as we don't do the second stage to label users who have not seen any 
        #simulations. When running this to filter simluations only, it is unfair to compare 
        #the paper's results with IBCC, since this method uses the test data to do training....
        #The iterative, semi-supervised nature of this method mean it is more like IBCC than it first seems.
        kIdxs = np.unique(self.crowdLabels[:,0])
        
        #counts per agent
        nCorrect = np.zeros(len(kIdxs))
        
        #values per data point
        detectionrate = np.zeros(len(self.gold))
        nClassifiers = np.zeros(len(self.gold))
        
        for l in range(self.crowdLabels.shape[0]):
            #for each label provided by crowd
            if self.crowdLabels[l,1] not in self.testIdxs:
                #training labels
                k = int(self.crowdLabels[l,0])
                i = int(self.crowdLabels[l,1])
                        
                #positive examples in training set
                if self.gold[i]==1:
                    #assumes 0 is positive!!!
                    nCorrect[k] += self.crowdLabels[l,2]==0 
                    detectionrate[i] += self.crowdLabels[l,2]==0 
                    nClassifiers[i] += 1
                else:
                    nClassifiers[i] += 1
            else:
                i = int(self.crowdLabels[l,1])
                nClassifiers[i] +=1
                    
        print 'Fraction of candidates with >=5 classifications: ' + \
                str(float(np.sum(nClassifiers>=5))/float(len(self.gold)))
                    
        detectionrate = np.divide(detectionrate, nClassifiers)
        errors = np.zeros(len(kIdxs)) 
        for l in range(self.crowdLabels.shape[0]):
            #for all labels with corresponding training
            if self.crowdLabels[l,1] not in self.testIdxs:
                k = int(self.crowdLabels[l,0])
                i = int(self.crowdLabels[l,1])
                if self.gold[i]==1:  #positive training examples
                    errors[k] += 0.2*detectionrate[i]* (self.crowdLabels[l,2]==1)
        
        weights = np.ones(len(kIdxs)) + nCorrect - errors 
        weights = weights - np.min(weights)
        weights = np.divide(weights, np.max(weights))
        
        pTVote = np.ones((len(self.gold),2))
        for l in range(self.crowdLabels.shape[0]):
            i = int(self.crowdLabels[l,1])
            k = int(self.crowdLabels[l,0])
            v = self.crowdLabels[l,2]==0
            if v:
                pTVote[i,1]+=weights[k]
            else:
                pTVote[i,0]+=weights[k]
        pTVote = np.divide(pTVote, np.sum(pTVote,axis=1).reshape(pTVote.shape[0],1))
        self.pT = pTVote
    
    def testIbccPerformance(self, runEvaluation=True):
        #An alternative to the runIbcc function in the Ibcc module, which does not save the resulting 
        #classifications, but prints a performance analysis
        if self.testIdxs==None or self.testIdxs==[]:
            self.testIdxs = np.argwhere(np.bitwise_and(self.gold>-1,self.goldTr==-1))
            self.testIdxs = self.testIdxs.reshape(-1)
     
        print ' No. test indexes = ' + str(len(self.testIdxs)) + ", with +ve examples " + str(len(np.argwhere(self.gold[self.testIdxs]>0)))
     
        self.pT = self.combiner.combineClassifications(self.crowdLabels, self.goldTr)
        print 'Nu: ' + str(self.combiner.nu)  
        
        if self.merge_all_pos:
            self.pT_premerge = self.pT
            self.pT = np.concatenate( (self.pT[:,0].reshape(self.pT.shape[0],1),\
                          np.sum(self.pT[:,1:],1).reshape(self.pT.shape[0],1)), axis=1)
        nclasses = self.pT.shape[1] 
        #analyse the accuracy of the results
        if not runEvaluation:
            return
        
        self.plotCumDist(nclasses)        
        acc,recall,spec,prec,auc,ap,nfiltered,filter_rate = self.getAccMeasures(nclasses)
      
        return acc,recall,spec,prec,auc,ap,nfiltered,filter_rate 
    
    def testSchwamb(self):
        self.combiner, self.crowdLabels, self.gold, self.orig_cand_ids,_,_,_,self.goldTypes = \
                                                                ibcc.loadCombiner(self.configfile)
        self.weightedVoteSchwamb()   
        nclasses = self.pT.shape[1]                                                        
        self.plotCumDist(nclasses)
        acc,recall,spec,prec,auc,ap,nfiltered,filter_rate = self.getAccMeasures(nclasses)
        self.printResults(acc,recall,spec,prec,auc,ap,nfiltered,filter_rate)
        result_array = [acc,recall,spec,prec,auc,ap,nfiltered,filter_rate]
        return result_array        
        
    def testUnsupervised(self, runEvaluation=True):
        # no training data, test all points we have true labels for
        self.combiner, self.crowdLabels, self.gold, self.orig_cand_ids,_,_,_,self.goldTypes = \
                                                                ibcc.loadCombiner(self.configfile)        
        self.goldTr = np.zeros(len(self.gold)) -1 
        
        if runEvaluation:
            acc,recall,spec,prec,auc,ap,nfiltered,filter_rate \
                = self.testIbccPerformance(runEvaluation=runEvaluation)
            self.printResults(acc,recall,spec,prec,auc,ap,nfiltered,filter_rate)
            result_array = [acc,recall,spec,prec,auc,ap,nfiltered,filter_rate]
            return result_array
        else:
            self.testIbccPerformance(runEvaluation=runEvaluation)
        
    def discretize_secondary_gold(self):
        '''
        Turn continuous feature values into discrete types that can be used as classes for training
        '''
        self.disc_gold_types = deepcopy(self.gold)
        for i in range(1, len(self.secondary_type_cats)):
            start = self.secondary_type_cats[i-1]
            end = self.secondary_type_cats[i]
            
            this_type = np.bitwise_and(self.goldTypes<end, self.goldTypes>=start)
            self.disc_gold_types[this_type] = i
        
    def testSupervised(self, runEvaluation=True):
        #supply all training data. The metrics will be unfair
        self.combiner, self.crowdLabels, self.gold, self.orig_cand_ids, self.trIdxs,_,_, self.goldTypes \
                                                                = ibcc.loadCombiner(self.configfile)
        
        if self.goldTypes != None and len(self.goldTypes)>0 and self.combiner.nClasses>2:
            self.discretize_secondary_gold()
            self.goldTr = np.zeros(len(self.gold)) -1
            self.goldTr[self.trIdxs] = self.disc_gold_types[self.trIdxs]
        elif self.trIdxs != None:
            self.goldTr = np.zeros(len(self.gold)) -1
            self.goldTr[self.trIdxs] = self.gold[self.trIdxs]
        else:
            self.goldTr = self.gold
        
        if runEvaluation:
            acc,recall,spec,prec,auc,ap,nfiltered,filter_rate = self.testIbccPerformance(runEvaluation=runEvaluation)
            self.printResults(acc,recall,spec,prec,auc,ap,nfiltered,filter_rate)
            result_array = [acc,recall,spec,prec,auc,ap,nfiltered,filter_rate]
            return result_array            
        else:
            self.testIbccPerformance(runEvaluation=runEvaluation)
    
    def testXValidation(self, nFolds, configFile):
        '''
        Run n-fold cross validation
        '''
        if nFolds==0:
            self.testSupervised(configFile)
            return
        elif nFolds==1:
            self.testUnsupervised(configFile)
            return
        
        #load the data
        self.combiner, self.crowdLabels, self.gold, _,_,_,_ = ibcc.loadCombiner(configFile)        
        meanAcc = 0
        meanRecall = np.zeros((1,self.combiner.nClasses-1))
        meanSpecificity = np.zeros((1, self.combiner.nClasses-1))
        meanPrecision = np.zeros((1, self.combiner.nClasses-1))
        meanAuc = np.zeros(self.combiner.nClasses-1)
        meanAp = np.zeros(self.combiner.nClasses-1)
        
        #split the data into nFolds different partitions
        trIdxs = np.argwhere(self.gold>-1)
        
        kf = KFold(len(trIdxs), n_folds=nFolds, indices=False)      
        
        #for each partition, run IBCC
        #any unlabelled data is included and is not split
        for trMask, testMask in kf:
            goldPartition = np.zeros(len(self.gold)) -1 
            goldPartition[trIdxs[trMask]] = self.gold[trIdxs[trMask]]
            self.goldTr = goldPartition
            self.testIdxs = trIdxs[testMask].reshape(-1)
            
            _,_,acc,recall,spec,prec,auc,ap = self.testIbccPerformance()  
            
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
        self.printResults(meanAcc,meanRecall,meanSpecificity,meanPrecision,meanAuc,meanAp,0,0)
        
    def extract_new_discoveries(self):
        '''
        Find possible new discoveries by extracting positive predictions from unlabelled points,
        and translate back to original IDs for further review
        '''
        unlab_idxs = np.ones(self.combiner.nObjs)
        unlab_idxs[np.argwhere(self.gold>-1)] = 0
        unlab_idxs = np.argwhere(unlab_idxs)
        
        pT_unlab = self.pT[unlab_idxs,1]
        discoveries = np.argwhere(pT_unlab>0.5)
        disco_idxs = unlab_idxs[discoveries]
        
        disco_orig_idxs = self.orig_cand_ids[disco_idxs]
        np.savetxt(self.outputdir+"/possible_discoveries.csv", disco_orig_idxs)
            
    #testXValidation('./config/my_project.py')
