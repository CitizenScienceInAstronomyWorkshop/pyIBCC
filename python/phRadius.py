'''
Analysis of IBCC performance on  Planet Hunters, breaking the data down by radius.

Created on 12 May 2014

@author: edwin
'''

import matplotlib.pyplot as plt
from ibccperformance import Evaluator
import logging, json

logging.basicConfig(level=logging.INFO)
plt.close('all')

results = []

# #TEST SET 1: Q1 Real Data Performance -- put that in the title
# #only the real q1 data
 
datalabel = "Confirmed Planets"

label = "unsup. learning on real"
ip = Evaluator('./python/config/ph_q1_uns.py', label, datalabel)
results.append(ip.testUnsupervised())
ip.plotRecallByRadius(0)
    
# #real data plus sims. Evaluate on real
label = "unsup. learning with real+sim"
ip = Evaluator('./python/config/ph_q1_uns_plussims.py', label, datalabel)
results.append(ip.testUnsupervised())
ip.plotRecallByRadius(1)
   
#real data plus sims. Eval on real. Sims as training.
label = "sims as training"
ip = Evaluator('./python/config/ph_q1_supe_plussims.py', label, datalabel)
results.append(ip.testSupervised())
ip.plotRecallByRadius(2)
ip.plot_skill_distribution()

#TEST SET 1+: Q1 Real Data Performance, IBCC divides peoples' reliability by radius group
   
# ADD IN ANOTHER METHOD FOR COMPARISON.
# REPEAT THE 50% SIMS METHOD MULTIPLE TIMES IF NOT DONE ALREADY?
   
label = "simulations as training, radius split"
ip = Evaluator('./python/config/ph_q1_supe_plussims_rad.py', label, datalabel)
results.append(ip.testSupervised())
ip.plotRecallByRadius(3)
ip.plot_skill_distribution()
ip.extract_new_discoveries()

#########################################################################

#TEST SET 2: Q1 Simulated Data Performance, all models learned using simulated and real crowd data.
#real data plus sims, evaluate only on sims 
datalabel = "Simulations"

label = "unsup. learning with real+sim"
ip = Evaluator('./python/config/ph_q1_uns_sims.py', label, datalabel, 4, 5, 6, 9, 10)
results.append(ip.testUnsupervised())
ip.plotRecallByRadius(0)

#sims only
label = "unsup. learning with sims only"
ip = Evaluator('./python/config/ph_q1_uns_simsonly.py', label, datalabel, 4, 5, 6, 9, 10)
results.append(ip.testUnsupervised())
ip.plotRecallByRadius(1)

#simulated data only, using 50% as training
label = "sims only, 50% sims as training"
ip = Evaluator('./python/config/ph_q1_supe_simsonly.py', label, datalabel, 4, 5, 6, 9, 10)
results.append(ip.testSupervised())
ip.plotRecallByRadius(3)

#real data plus sims. Eval on half of sims, other half as training.
label = "real+sim, 50% sims as training"
ip = Evaluator('./python/config/ph_q1_supe_sims.py', label, datalabel, 4, 5, 6, 9, 10, 12)
results.append(ip.testSupervised())
ip.plotRecallByRadius(2)
ip.plot_skill_distribution()
  
#TEST SET 2+: Simulated Data Performance, IBCC divides reliability by radius
# #real data plus sims. Eval on half of sims, other half as training.
label = "real+sim, 50% sims as training, radius split"
ip = Evaluator('./python/config/ph_q1_supe_sims_rad.py', label, datalabel, 4, 5, 6, 9, 10, 12)
results.append(ip.testSupervised())
ip.plotRecallByRadius(4)
ip.plot_skill_distribution()
#########################################################################

with open(ip.outputdir+"/results.json", 'w') as outfile:
    json.dump(results, outfile, indent=2)

###########################################################################

#From test set 1+, removed because we don't test radius split in unsupervised mode 
#real data plus sims. Evaluate on real -- unsupervised doesn't make sense with radius split
# label = "unsup. learning with real+sim, radius split"
# ip.testUnsupervised('./python/config/ph_q1_uns_plussims_rad.py', label)

#real data plus sims. Eval on real. Sims as training.
  
#From test set 2+, see above 
#real data plus sims, evaluate only on sims 
# label = "unsup. learning with real+sim, radius split"
# pT,goldTypes,testIdxs,goldAll = ip.testUnsupervised(\
#                                     './python/config/ph_q1_uns_sims_rad.py', label)
# plotRecallByRadius(pT, goldTypes, testIdxs, goldAll, label)  
  
#########################################################################    
  
# ip.plotCumDist(pTsum,2,testIdxsReal,goldAll, label)        
# acc,recall,spec,prec,auc,ap = \
#     ip.getAccMeasures(pTsum,goldAll,2,testIdxsReal)

#remove the real ones from the testIdxs
# goldAll[testIdxsReal]  = -1
#  
# testIdxs = np.setdiff1d(testIdxs, testIdxsReal, True)
#  
# goldAll[goldAll>1] = 1
# ip.plotCumDist(pTsum,2,testIdxs,goldAll, label)        
# acc,recall,spec,prec,auc,ap = \
#     ip.getAccMeasures(pTsum,goldAll,2,testIdxs)
# plotRecallByRadius(pT, goldTypes, testIdxs, goldAll, label)
