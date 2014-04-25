import numpy as np

print 'Configuring IBCC'

scores = np.array([9, 10])
nScores = len(scores)
nClasses = 4
inputFile =   '../data/PH data/PlanetHunters_3-26-14_Q1_annnotations_IBCC.csv'
goldFile =    '../data/PH data/PlanetHunters_3-26-14_Q1_light_curves_IBCC.csv'
outputFile =  '../output/ph/output.csv'
confMatFile = '../output/ph/confMat.csv'
classLabels=None#do this conversion in a spreadsheet due to bugs['candidate','planet','eb','simulation']
alpha0 = np.array([[1, 2], [2, 1],[1.8,1.2], [2.5,1]]) #for PH data
nu0 = np.array([50.0, 50.0, 50.0, 50.0])