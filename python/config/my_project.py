#!/usr/bin/env python

''' Configuration file for pyIBCC. Currently set for Galaxy Zoo CANDELS data. '''

__author__ = "Kyle Willett"

print 'Running my_project.py --- configuring IBCC'

scores = np.array([0,1,2])
nScores = len(scores)
nClasses = 3
inputFile =   './data/gz_candels_input_1000.csv'
goldFile =    './data/gold.csv'
outputFile =  './output/output.csv'
confMatFile = './output/confMat.csv'

nu0 = np.array([25,25,50])
alpha0 = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])

lang = 'en-us'
