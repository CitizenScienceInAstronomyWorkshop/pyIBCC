'''
Created on Dec 27, 2012

@author: kieranfinn
'''
import settings
from functions import *
from collections import defaultdict

def general_mean(objects,people,scores,weights): #really only works for two outputs,may need to adjust
    results=defaultdict(lambda:[0.0,0.0])
    for i in range(len(objects)):
        if i%10000==0:
            overprint('Updating results for object %s of %s' %(add_comma(i),add_comma(len(objects))))
        o=objects[i]
        p=people[i]
        s=scores[i]*weights[p] #shorthand
        results[o][0]+=s
        results[o][1]+=weights[p]
    for i in results.keys():
        results[i]=float(results[i][0])/results[i][1]
    print '\n'
    return dict(results)

def mean(data):
    objects,people,scores,labels=data
    weights={}
    for i in people:
        weights[i]=1
    return general_mean(objects,people,scores,weights)

def weighted_mean(data): 
    '''this is a very simple weighted mean which gives users +1 if they get a classification right and -1 if they get one wrong'''
    split=int(round(len(data[0])*settings.split_frac))
    objects,people,scores,labels=data
    tobjects,tpeople,tscores,tlabels=[i[:split] for i in data]
    weights=defaultdict(lambda:1)
    for i in range(len(tpeople)):
        if tscores[i]==tlabels[i]:
            to_add=1
        elif tlabels[i] != settings.unsure_value:
            to_add=-1
        else:
            to_add=0
        weights[tpeople[i]]+=to_add
    try:
        mn=min(weights.values())-1
        for w in weights.keys():
            weights[w] -= mn #ensures the minimum weight is 1
    except:
        pass
    return general_mean(objects,people,scores,weights)

def consensus(data):#score is simply the number of people
    objects,people,scores,labels=data
    out=defaultdict(int)
    for i in range(len(objects)):
        if i%10000==0:
            overprint('Updating results for object %s of %s' %(add_comma(i),add_comma(len(objects))))
        out[objects[i]]+=scores[i]
    print '\n'
    return dict(out)