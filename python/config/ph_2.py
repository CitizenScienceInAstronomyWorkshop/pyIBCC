import numpy as np

print 'Configuring IBCC'

def translateGold(gold):
    #turn the EBs and simulations into instances of "planet"
    gold[gold==2] = 1
    gold[gold==3] = 1
    gold[gold==-1] = 0
    return gold

scores = np.array([9, 10])
nScores = len(scores)
nClasses = 2
inputFile =   './data/PH data/PlanetHunters_3-26-14_Q1_annnotations_IBCC.csv'
goldFile =    './data/PH data/PlanetHunters_3-26-14_Q1_light_curves_IBCC.csv'
outputFile =  './output/ph/output.csv'
confMatFile = './output/ph/confMat.csv'
classLabels=None#do this conversion in a spreadsheet due to bugs['candidate','planet','eb','simulation']
alpha0 = np.array([[1, 2], [2, 1]]) #for PH data
nu0 = np.array([10.0, 10.0])

print 'Planet hunters 2-class config done.'