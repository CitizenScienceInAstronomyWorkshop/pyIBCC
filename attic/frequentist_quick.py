'''
Created on 10 Jan 2013

@author: Kieran Finn
'''
import settings
from copy import deepcopy
import numpy as np
from functions import *
from collections import defaultdict

def refine_data(dat):
    print 'refining data'
    objects,people,scores,labels=dat
    planets=longlist([])
    candidates=longlist([])
    data=defaultdict(lambda: defaultdict(lambda:0))
    rev_data=defaultdict(lambda: defaultdict(lambda:0))
    for i in range(len(objects)):
        if i%1000==0:
            overprint('Processing line %s of %s' %(add_comma(i),add_comma(len(objects))))
        p=people[i]
        o=objects[i]
        s=scores[i]
        l=labels[i]
        if l==1:
            planets.append(o)
        else:
            candidates.append(o)
        set_item(data,o,p,max(s,data[o][p]))
        set_item(rev_data,p,o,data[o][p])
    tpeople=set(people)
    planets=set(planets)
    candidates=set(candidates)
    print '\n%s objects and %s users' %(add_comma(len(planets)+len(candidates)),add_comma(len(tpeople)))    
    return (planets,candidates,tpeople,data,rev_data)

def initiate_weights(planets,people,data):
    print 'initialising weights'
    weights={k:1.0 for k in people}
    changed=[]
    for i in planets:
        spotted=[]
        not_spotted=[]
        for k in data[i].keys():
            changed.append(k)
            if data[i][k]==1:
                spotted.append(k)
                weights[k]+=1
            else:
                not_spotted.append(k)
        try:
            score=0.2*float(len(spotted))/len(not_spotted)
            for k in not_spotted:
                weights[k]-=score
        except ZeroDivisionError: #happens if everyone spotted it
            pass
    N=max(weights.values())/2
    for k in changed:
        weights[k]/=N
    return weights

def get_scores(objects,weights,data):
    scores={}
    for i in objects:
        s=0.0
        N=0.0
        for k in data[i].keys():
            s+=data[i][k]*weights[k]
            N+=weights[k]
        try:
            scores[i]=s/N
        except ZeroDivisionError:
            scores[i]=0.0
    return scores

def get_weights(people,scores,rev_data):
    weights={}
    for k in people:
        w=0.0
        N=0.0
        for i in rev_data[k].keys():
            if rev_data[k][i]==1:
                w+=scores[i]
            else:
                w+=1.0-scores[i]
            N+=1
        try:
            weights[k]=w/N
        except ZeroDivisionError:
            weights[k]=0.0
    N=float(sum(weights.values()))/len(weights.values())
    for k in weights.keys():
        weights[k]/=N
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
    planets,candidates,people,data,rev_data=refine_data(data)
    weights=initiate_weights(planets,people,data)
    objects=planets.union(candidates)
    
    '''main loop'''
    iteration=0
    while True:
        iteration+=1
        print 'iteration %d' %iteration
        pre_weights=deepcopy(weights)
        scores=get_scores(objects,weights,data)
        weights=get_weights(people,scores,rev_data)
        if check_convergence(pre_weights,weights):
            return scores
        
    
    