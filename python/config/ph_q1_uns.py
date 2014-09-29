import numpy as np
import ph_common

print 'Configuring IBCC'

def translateGold(gold):
    #EBs are just unlabelled data - not labelled training, not used for evaluation
    gold[gold==2] = -1
    #simulations are treated as planets 
    gold[gold==3] = 1
    return gold

scores = np.array([9, 10])
nScores = len(scores)
nClasses = 2
inputFile =   './data/PH data/paper_crowd/PlanetHunters_3-26-14_Q1_annnotations_IBCC.csv'
goldFile =    './data/PH data/paper_gold/PH3-26-14_Q1_real.csv'
outputFile =  './output/ph/output.csv'
confMatFile = './output/ph/confMat.csv'
classLabels=None#do this conversion in a spreadsheet due to bugs['candidate','planet','eb','simulation']
alpha0 = ph_common.alpha0
nu0 = ph_common.nu0

goldTypeCol = 3

print 'Planet hunters 2-class config done.'