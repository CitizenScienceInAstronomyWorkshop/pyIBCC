#!/usr/bin/env python

''' Configuration file for pyIBCC. Currently set for Galaxy Zoo (which version?) data. '''

__author__ = "Kyle Willett"

print 'Running my_project.py --- configuring IBCC'

scores = np.array([3, 4])
nScores = len(scores)
nClasses = 2
inputFile =   './data/input.csv'
goldFile =    './data/gold.csv'
outputFile =  './output/output.csv'
confMatFile = './output/confMat.csv'

nu0 = np.array([50,50])
alpha0 = np.array([[2, 1], 
                   [1, 2]])

lang = 'en-us'
