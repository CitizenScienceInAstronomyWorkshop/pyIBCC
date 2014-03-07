'''
Created on 10 Jan 2013

@author: Kieran Finn
'''
import settings
from copy import deepcopy
import numpy as np
from functions import *
from collections import defaultdict

class object_details():
    def __init__(self):
        self.spotted=0.0
        self.not_spotted=0.0
        self.sc=False
    def score(self):
        if not self.sc:#this way the calculation only has to be performed once
            N=self.spotted+self.not_spotted
            if N==0:
                self.sc=0.0
            else:
                self.sc= 0.2*self.spotted/N
        return self.sc

def get_indices(labels):
    out=[]#may need to make this a longlist if I run into memory errors
    print 'Finding known planets'
    for i in xrange(len(labels)):
        if i%1000==0:
            overprint('Processing line %s of %s' %(add_comma(i),add_comma(len(labels))))
        if labels[i]==1:
            out.append(i)
    print '\n'
    return out

def initiate_weights(objects,people,scores,indices):
    object_dict=defaultdict(lambda:object_details())
    weights=defaultdict(lambda:[1.0])
    print 'initialising weights'
    count=0
    for a in indices:
        if count%1000==0:
            overprint('Processing line %s of %s' %(add_comma(count),add_comma(len(indices))))
        count+=1
        i=objects[a]
        k=people[a]
        l=scores[a]
        if l==1:
            weights[k][0]+=1
            object_dict[i].spotted+=1
        else:
            weights[k].append(object_dict[i])
            object_dict[i].not_spotted+=1
    print '\nSumming up scores'
    for k in weights.keys():
        temp=weights[k][0]
        N=(weights[k][0]+len(weights[k])-1)/2#initial weights are normlised so that maximum score is 2
        for i in range(1,len(weights[k])):
            temp-=weights[k][i].score()
        weights[k]=temp/N
    weights.default_factory=lambda:1.0#changes the default weight to 1 instead of a list
    return weights

def get_scores(objects,people,scores,weights):
    print 'calculating scores'
    results=defaultdict(lambda:[0.0,0.0])
    for a in xrange(len(objects)):
        if a%10000==0:
            overprint('Processing line %s of %s' %(add_comma(a),add_comma(len(objects)))) 
        i=objects[a]
        k=people[a]
        l=scores[a]
        results[i][1]+=weights[k]
        if l==1:
            results[i][0]+=weights[k]            
    for i in results.keys():
        try:
            results[i]=results[i][0]/results[i][1]
        except ZeroDivisionError:#shouldn't happen but just in case
            results[i]=0
    print '\n'
    return results

def get_weights(objects,people,scores,results):
    print 'calculating weights'
    weights=defaultdict(lambda:[0.0,0.0])
    for a in xrange(len(objects)):
        if a%10000==0:
            overprint('Processing line %s of %s' %(add_comma(a),add_comma(len(objects)))) 
        i=objects[a]
        k=people[a]
        l=scores[a]
        weights[k][1]+=1
        if l==1:
            weights[k][0]+=results[i]
        else:
            weights[k][0]+=1.0-results[i]
    N=0.0            
    for k in weights.keys():
        try:
            temp=weights[k][0]/weights[k][1]
        except ZeroDivisionError:#shouldn't happen but just in case
            temp=0
        N+=temp
        weights[k]=temp
    N/=len(weights.keys())
    for k in weights.keys():
        weights[k]/=N
    print '\n'
    return weights
        
def check_convergence(pre_weights,weights):
    to_check=[]
    for i in weights.keys():
        to_check.append(abs(pre_weights[i]-weights[i]))
    out=np.median(to_check)
    if out>1e-4:
        print 'median difference is %.3e. Continuing algorithm' %out
        return False
    else:
        return True

def main(data):
    objects,people,scores,labels=data
    planet_indices=get_indices(labels)
    weights=initiate_weights(objects,people,scores,planet_indices)
    
    '''main loop'''
    iteration=0
    while True:
        iteration+=1
        print 'iteration %d' %iteration
        pre_weights=deepcopy(weights)
        results=get_scores(objects,people,scores,weights)
        weights=get_weights(objects,people,scores,results)
        if check_convergence(pre_weights,weights):
            return dict(results)#converts it to a dict so it can be pickled
        
    
    