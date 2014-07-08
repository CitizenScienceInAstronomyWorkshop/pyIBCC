import numpy as np

print 'Configuring IBCC'

folder = 'galaxy-zoo-candels-test'
label = '_quick_100_mod10'

scores = [0,1,2]
nScores = len(scores)
nClasses = nScores
nu0 = np.array((50, 25, 25))
alpha0 = np.array([[2, 1, 1],
                   [1, 2, 1],
                   [1, 1, 2]])

inputFile =   '../data/{}/input{}.csv'.format(folder, label)
goldFile =    '../data/{}/gold{}.csv'.format(folder, label)
outputFile =  '../output/{}/output{}.csv'.format(folder, label)
confMatFile = '../output/{}/confMat{}.csv'.format(folder, label)
