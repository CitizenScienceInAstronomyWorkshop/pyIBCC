import numpy as np

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
inputFile =   './data/PH data/paper_crowd/PlanetHunters_3-26-14_Q1_ann_realAndSim.csv'
goldFile =    './data/PH data/paper_gold/PH6-3-14_Q1_sims.csv'
outputFile =  './output/ph/output.csv'
confMatFile = './output/ph/confMat.csv'
classLabels=None#do this conversion in a spreadsheet due to bugs['candidate','planet','eb','simulation']
alpha0 = np.array([[199, 200], [200, 199]]) #for PH data
nu0 = np.array([10.0, 10.0])

goldTypeCol = 3

print 'Planet hunters 2-class config done.'