import numpy as np

print 'Configuring IBCC'

scores = np.array([3, 4])
nScores = len(scores)
nClasses = 2
nu0 = np.array([50.0, 50.0])
alpha0 = np.array([[2, 1], [1, 2]])  
inputFile =   './data/test1/input.csv'
goldFile =    './data/test1/gold.csv'
outputFile =  './output/output.csv'
confMatFile = './output/confMat.csv'