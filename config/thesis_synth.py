import numpy as np

print 'Configuring IBCC'

def translate_gold(gold):
    #turn the EBs and simulations into instances of "planet"
    from copy import deepcopy
    translated = deepcopy(gold)
    translated[gold==1] = 0
    translated[gold==2] = 1
    return translated

scores = np.array([0, 1])
nScores = len(scores)
nClasses = 2
inputFile =   '/homes/49/edwin/data/thesis_synth_sharing/exp1/data/base_var1_d1.csv'
goldFile =    '/homes/49/edwin/data/thesis_synth_sharing/exp1/data/labels_var1_d1.csv'
outputFile =  '../output/thesis_synth/output.csv'
confMatFile = '../output/thesis_synth/confMat.csv'
classLabels=None#do this conversion in a spreadsheet due to bugs['candidate','planet','eb','simulation']
alpha0 = np.array([[8, 1], [1, 8]]) #for PH data
nu0 = np.array([1.0, 1.0])

tableFormat = True

print 'Thesis synthetic experiments config done.'