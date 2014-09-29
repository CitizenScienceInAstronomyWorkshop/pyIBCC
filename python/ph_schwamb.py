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
ip.outputdir = ip.outputdir + "/schwamb"
results.append(ip.testSchwamb())
ip.plotRecallByRadius(0)

#########################################################################

#TEST SET 2: Q1 Simulated Data Performance, all models learned using simulated and real crowd data.
#real data plus sims, evaluate only on sims 
datalabel = "Simulations"

#sims only
label = "unsup. learning with sims only"
ip = Evaluator('./python/config/ph_q1_uns_simsonly.py', label, datalabel, 4, 5, 6, 9, 10)
ip.outputdir = ip.outputdir + "/schwamb"
results.append(ip.testSchwamb())
ip.plotRecallByRadius(1)
#########################################################################

with open(ip.outputdir+"/results.json") as outfile:
    json.dump(results, outfile, indent=2)