import numpy as np

print 'Configuring IBCC'

#Include the simulations as training data.

def translateGold(gold):
    #turn the EBs and simulations into instances of "planet"
    gold[gold==2] = 1
    gold[gold==3] = 1
    gold[gold==-1] = 0
    return gold

scores = np.array([9, 10])
nScores = len(scores)
nClasses = 8
inputFile =   './data/PH data/paper_crowd/PlanetHunters_3-26-14_Q1_ann_realAndSim.csv'
goldFile =    './data/PH data/paper_gold/PH6-3-14_Q1_sims_with_real.csv'#_rad?
outputFile =  './output/ph/output.csv'
confMatFile = './output/ph/confMat.csv'
classLabels=None#do this conversion in a spreadsheet due to bugs['candidate','planet','eb','simulation']
alpha0 = np.array([[1, 2], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1]]) #for PH data
nu0 = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
trainIdFile = './data/PH data/paper_gold/PH6-3-14_Q1_sims.csv'
try:
    trainIds = np.genfromtxt(trainIdFile, delimiter=',', skip_header=1,usecols=[0],invalid_raise=True)
except Exception:
    trainIds = np.genfromtxt(trainIdFile, delimiter=',', skip_header=1)

goldTypeCol = 3

print 'Planet hunters 2-class config done.'