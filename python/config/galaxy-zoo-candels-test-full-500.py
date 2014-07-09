import numpy as np

print 'Configuring IBCC'

folder = 'galaxy-zoo-candels-test'
label = '_full_500'

scores = [0,1,2]
nScores = len(scores)
nClasses = nScores
nu0 = np.array((60, 30, 10))
alpha0 = np.array([[3, 2, 1],
                   [2, 3, 1],
                   [1, 1, 4]])

inputFile =   '../data/{}/input{}.csv'.format(folder, label)
goldFile =    '../data/{}/gold{}.csv'.format(folder, label)
outputFile =  '../output/{}/output{}.csv'.format(folder, label)
confMatFile = '../output/{}/confMat{}.csv'.format(folder, label)
