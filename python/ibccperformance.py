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

    nclasses = 2
    
    figure1 = 1
    figure2 = 2
    figure3 = 3
    
    figure7 = 7
    figure8 = 8
    
    skill_fig = 11
    
    outputdir = "./output/plots/"
    
    combiner = None
    pT = []
    
    dh = None #data handler for IBCC
    
    testIdxs = []
    trIdxs = []
    
    gold_tr = [] #gold labels for training
    goldsubtypes = [] # sub-types of the classes
    disc_gold_types = []
    
    secondary_type_cats = [2, 3, 4, 5, 6, 7, 10, 16]
        
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
    
    def eval_auc(self, testresults=[], labels=[]):
        '''
        Calculate the area under the ROC curve (called AUC) and 
        the area under the precision-recall curve, called the average precision (AP).
        '''
        if testresults==[]:
            testresults = self.pT[self.testIdxs,1:]
        if labels==[]:
            labels = self.dh.goldlabels[self.testIdxs].reshape(-1)
        labelmatrix = np.zeros((len(labels),self.nclasses-1)) 
        for j in range(1,self.nclasses):
            labelmatrix[labels==j,j-1] = 1
        
        auc_result = np.zeros(self.nclasses-1)
        ap = np.zeros(self.nclasses-1)
        for j in range(self.nclasses-1):
            y_true = labelmatrix[:,j]
            y_scores = testresults[:,j]
            #auc[j] = roc_auc_score(y_true, y_scores) #need scikit 0.14. 
            FPR, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
            auc_result[j] = auc(FPR, tpr)
            ap[j] = average_precision_score(y_true, y_scores)
                        
            diffs = tpr-FPR
            best = np.argmax(diffs)
            print  'The best threshold is ' + str(thresholds[best]) + ' at diff of ' + str(np.max(diffs))       
            print 'tpr: ' + str(tpr[best])
            print 'FPR: ' + str(FPR[best])
                        
        if self.nclasses==2:
            auc_result = auc_result[0]
            ap = ap[0]
                        
        return auc_result, ap
    
    def eval_crossentropy(self, testresults=[], labels=[]):
        if testresults==[]:
            testresults = self.pT[self.testIdxs,:]
        elif testresults.ndim==1:
            testresults.shape = (testresults.size,1)
            testresults = np.concatenate((testresults,1-testresults), axis=1)             
        if labels==[]:
            labels = self.dh.goldlabels[self.testIdxs].reshape(-1)
                
        crossentropy = -np.log(testresults[:,labels])
        infidxs = np.isinf(crossentropy)
        print "number of infinite entropy data points: " + str(np.sum(infidxs==True))
        crossentropy[infidxs] = -np.log(0.0000001)
        print "number of cross entropy points: " + str(crossentropy.size)
        crossentropy = np.sum(crossentropy)/labels.size
        return crossentropy    
    
    def eval_results(self):
        '''
        If testIdxs is not set, we compare against all data points. 
        If testIdxs is set, we ignore other data points, assuming they were part of the training set.
        '''
        
        labels = self.dh.goldlabels[self.testIdxs].reshape(-1)
                        
        print 'Evaluating the results using the greedy classifications.'
        greedyresults = np.argmax(self.pT[self.testIdxs,:])#np.round(pT_test_pos)
        
        acc = 1 - np.sum(labels!=greedyresults, 0) / len(self.testIdxs)
        print 'acc: ' + str(acc)
        #each class column will be the same. Related to a weighted average of precision across all classes
        
        disagreement = labels!=greedyresults
        fn = np.sum(disagreement[labels>=1],0)[0]
        fp = np.sum(disagreement[labels==0],0)[0]
        
        agreement = labels==greedyresults
        tp = np.sum(agreement[labels>=1],0)[0] 
        tn = np.sum(agreement[labels==0],0)[0]
            
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
            auc = np.zeros(self.nclasses-1)
            ap = np.zeros(self.nclasses-1)
        else:
            auc, ap = self.eval_auc()        
            
        return acc,recall,spec,prec,auc,ap,nfiltered,filter_rate
    
    def plot_cum_dist(self):    
        #sort probabilities in order
        #x values are the probabilities
        #y values are indices
        # have to select the highest y value for any duplicate x values
        #skip first class as this is used for uninteresting examples
        
        gold_test = self.dh.goldlabels[self.testIdxs]
        
        for j in range(1,self.nclasses):
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
            
    def print_results(self, meanAcc,meanRecall,meanSpecificity,meanPrecision,meanAuc,meanAp,nfiltered,filter_rate):
        if self.nclasses > 1:
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
        meanAuc = np.sum(meanAuc)/self.nclasses
        meanAp = np.sum(meanAp)/self.nclasses
        meanRecall = np.sum(meanRecall)/self.nclasses
        meanSpecificity = np.sum(meanSpecificity)/self.nclasses
        meanPrecision = np.sum(meanPrecision)/self.nclasses  
        
        print 'Mean Accuracy: ' + str(meanAcc)
        print 'Mean Recall: ' + str(meanRecall)
        print 'Mean Specificity: ' + str(meanSpecificity)
        print 'Mean Precision: ' + str(meanPrecision)
        print 'Mean AUC: ' + str(meanAuc)
        print 'Mean AP: ' + str(meanAp)
        print "Mean no. data points marked positive: " + str(nfiltered)
        print "Mean fraction marked as positive: " + str(filter_rate)
        
    def plot_recall_by_type(self, seqno):
    
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
            thisType = np.bitwise_and(self.dh.goldsubtypes<t, self.dh.goldsubtypes>=start)
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
            gold_t = self.dh.goldlabels[idxs]
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
        original_nclasses = self.combiner.alpha.shape[0] #before we merge subtypes
        alphasum = np.sum(self.combiner.alpha, 1).reshape((original_nclasses,1,self.combiner.K))
        pi = self.combiner.alpha / alphasum
        skill = pi[1:,0,:] #assumes score 0 is a positive classification
        skill = skill.reshape((skill.shape[0],self.combiner.K))
        sorted_skill = np.sort(skill, 1)
        
        xidx = np.arange(self.combiner.K, dtype=np.float)
        norm_xidx = xidx/np.float(self.combiner.K)
        
        if original_nclasses>2:
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
        #lnproportions = self.combiner.lnNu.reshape((self.nclasses,0,1))
        #pt_given_c = np.exp(lnpi[1:,0,:]+lnproportions[1:,:,:]) / np.exp(np.sum(lnpi[:,0,:]+lnproportions, axis=0))
        
    def print_mv_by_type(self):
        tVals = self.secondary_type_cats[1:]
        start = 2      
           
        for i,t in enumerate(tVals):
            
            thisType = np.bitwise_and(self.dh.goldsubtypes<t, self.dh.goldsubtypes>=start)
            idxs = self.testIdxs[thisType[self.testIdxs]]
            pT_t = self.pT[idxs,:]
            greedyLabels = np.round(pT_t[:,1])
            start = t
            hits = np.zeros(len(thisType))
            seenBy = np.zeros(len(thisType))
            for l in range(self.dh.crowdlabels.shape[0]):
                idx = int(self.dh.crowdlabels[l,1])
                if self.dh.crowdlabels[l,2]==0:
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
        
    def run_test(self, evaluate=True):
        #An alternative to the runIbcc function in the Ibcc module, which does not save the resulting 
        #classifications, but prints a performance analysis
        if self.testIdxs==None or self.testIdxs==[]:
            self.testIdxs = np.argwhere(np.bitwise_and(self.dh.goldlabels>-1,self.gold_tr==-1))
            self.testIdxs = self.testIdxs.reshape(-1)
     
        print ' No. test indexes = ' + str(len(self.testIdxs)) + ", with +ve examples " + str(len(np.argwhere(self.dh.goldlabels[self.testIdxs]>0)))
     
        self.pT = self.combiner.combine_classifications(self.dh.crowdlabels, self.gold_tr)
        self.dh.save_hyperparams(self.combiner.alpha, self.combiner.nu, self.combiner.noptiter)
        
        if self.merge_all_pos:
            self.pT_premerge = self.pT
            self.pT = np.concatenate( (self.pT[:,0].reshape(self.pT.shape[0],1),\
                          np.sum(self.pT[:,1:],1).reshape(self.pT.shape[0],1)), axis=1)
        self.nclasses = self.pT.shape[1] 
        #analyse the accuracy of the results
        if not evaluate:
            return
        
        self.plot_cum_dist()        
        acc,recall,spec,prec,auc,ap,nfiltered,filter_rate = self.eval_results()
      
        return acc,recall,spec,prec,auc,ap,nfiltered,filter_rate    
        
    def test_unsupervised(self, evaluate=True):
        # no training data, test all points we have true labels for
        self.combiner, self.dh = ibcc.load_combiner(self.configfile)        
        self.gold_tr = np.zeros(len(self.dh.goldlabels)) -1 
        self.nclasses = self.combiner.nclasses
        
        if evaluate:
            acc,recall,spec,prec,auc,ap,nfiltered,filter_rate \
                = self.run_test(evaluate=evaluate)
            self.print_results(acc,recall,spec,prec,auc,ap,nfiltered,filter_rate)
        else:
            acc,recall,spec,prec,auc,ap,nfiltered,filter_rate = self.run_test(evaluate=evaluate)
        
        result_array = self.make_result_list(acc, recall, spec, prec, auc, ap, nfiltered, filter_rate)
        return result_array
        
    def test_supervised(self, evaluate=True):
        #supply all training data. The metrics will be unfair
        if self.combiner==None:
            self.load_supervised()
        
        if evaluate:
            acc,recall,spec,prec,auc,ap,nfiltered,filter_rate = self.run_test(evaluate=evaluate)
            self.print_results(acc,recall,spec,prec,auc,ap,nfiltered,filter_rate)
        else:
            acc,recall,spec,prec,auc,ap,nfiltered,filter_rate = self.run_test(evaluate=evaluate)

        result_array = self.make_result_list(acc,recall,spec,prec,auc,ap,nfiltered,filter_rate)
        return result_array            
    
    def load_supervised(self):
        self.combiner, self.dh = ibcc.load_combiner(self.configfile)
        self.nclasses = self.combiner.nclasses
        
        if self.dh.goldsubtypes != None and len(self.dh.goldsubtypes)>0 and self.nclasses>2:
            self.discretize_secondary_gold()
            self.gold_tr = np.zeros(len(self.dh.goldlabels)) -1
            self.gold_tr[self.dh.trainids] = self.disc_gold_types[self.dh.trainids]
        elif self.dh.trainids != None:
            self.gold_tr = np.zeros(len(self.dh.goldlabels)) -1
            self.gold_tr[self.dh.trainids] = self.dh.goldlabels[self.dh.trainids]
        else:
            self.gold_tr = self.dh.goldlabels        
    
    def test_x_validation(self, nfolds):
        '''
        Run n-fold cross validation
        '''
        if nfolds==0:
            self.test_supervised()
            return
        elif nfolds==1:
            self.test_unsupervised()
            return
        
        #load the data
        self.load_supervised()
        all_trainids = np.argwhere(self.gold_tr!=-1) #save the complete set of training labels
        kf = KFold(len(all_trainids), n_folds=nfolds, indices=False)#split the data into nfolds       

        result_array = None
        
        #for each partition, run IBCC
        #any unlabelled data is included and is not split
        for trMask, _ in kf:
            gold_tr_k = np.zeros(len(self.dh.goldlabels)) -1 
            gold_tr_k[all_trainids[trMask]] = self.dh.goldlabels[all_trainids[trMask]]
            self.gold_tr = gold_tr_k
            self.testIdxs = None
            
            acc,recall,spec,prec,auc,ap,nfiltered,filter_rate = self.run_test()  
            
            #save to overall summary
            result_array_k = np.array(self.make_result_list(acc,recall,spec,prec,auc,ap,nfiltered,filter_rate))
            result_array_k = result_array_k.reshape(1,result_array_k.size)
            if result_array==None:
                result_array = result_array_k
            else:
                result_array = np.concatenate((result_array,result_array_k), axis=0)
                  
        meanresults = np.sum(result_array,axis=0)/nfolds
                    
        meanacc = meanresults[0]
        meanrecall = meanresults[1]
        meanspec = meanresults[2]
        meanprec = meanresults[3]
        meanAuc = meanresults[4]
        meanAp = meanresults[5]
        meannfil = meanresults[6]
        meanfilrate = meanresults[7]
        
        #display summary of results across all folds
        self.print_results(meanacc,meanrecall,meanspec,meanprec,meanAuc,meanAp,meannfil,meanfilrate)

    def discretize_secondary_gold(self):
        '''
        Turn continuous feature values into discrete types that can be used as classes for training
        '''
        self.disc_gold_types = deepcopy(self.dh.goldlabels)
        for i in range(1, len(self.secondary_type_cats)):
            start = self.secondary_type_cats[i-1]
            end = self.secondary_type_cats[i]
            
            this_type = np.bitwise_and(self.dh.goldsubtypes<end, self.dh.goldsubtypes>=start)
            self.disc_gold_types[this_type] = i
    
    def make_result_list(self, acc,recall,spec,prec,auc,ap,nfiltered,filter_rate):
                
        results = [acc,recall,spec,prec,auc,ap,nfiltered,filter_rate]
                
        for i, item in enumerate(results):
            if type(item)==np.ndarray:
                if len(item)<=1:
                    results[i] = item[0]
                else:
                    results[i] = item.tolist()
        return results
        
    def extract_new_discoveries(self):
        '''
        Find possible new discoveries by extracting positive predictions from unlabelled points,
        and translate back to original IDs for further review
        '''
        unlab_idxs = np.ones(self.combiner.N)
        unlab_idxs[np.argwhere(self.dh.goldlabels>-1)] = 0
        unlab_idxs = np.argwhere(unlab_idxs)
        
        pT_unlab = self.pT[unlab_idxs,1]
        discoveries = np.argwhere(pT_unlab>0.5)
        disco_idxs = unlab_idxs[discoveries]
        
        disco_orig_idxs = self.dh.targetidxs[disco_idxs]
        np.savetxt(self.outputdir+"/possible_discoveries.csv", disco_orig_idxs)
            
    #test_x_validation('./config/my_project.py')
