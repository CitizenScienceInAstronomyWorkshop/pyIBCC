'''
Created on 5 Mar 2013

@author: Kieran Finn
'''
import numpy as np
from functions import *
from math import sqrt
from scipy import interpolate

folder='D:/Documents/Planet_hunters/results/01_Feb_2013_11_14/'

methods=['weighted mean','mean','frequentist']

def get_threshold_plot(data,normalise=True,resolution=0.0001):
    ''' may be a quicker way. also may need to amend to calculate auc'''
    threshold=np.arange(0,1+resolution,resolution)
    fraction=np.zeros_like(threshold)
    for datum in data:
        index=int(datum/resolution)
        fraction[index]+=1
    if normalise:
        N=sum(fraction)
        if N==0:
            print 'not enough known labels to produce curve.'
            return [0,0]
        for i in range(len(fraction)):
            fraction[i]/=N
    for i in range(1,len(fraction)):#this should cumulate the results without many sums
        fraction[i]+=fraction[i-1]
    return (np.array(threshold),np.array(fraction))

labeldict={'candidate':0,'planet':1,'synthetic':1,'eb':0,'simulation':1}
def get_source_labels():
    print 'reading source labels'
    f=open('source_labels.csv')
    out={}
    for line in f:
        source,label=line.split(',')
        out[source]=labeldict[label.strip().strip('"')]
    f.close()
    return out
labels=get_source_labels()

def combine(results):
    out={0:[],1:[]}
    for i in results.keys():
        out[labels[i]].append(results[i])
    return out

def get_cumulative(fname):
    results=pload(folder+fname+'.dat')
    n=len(results)
    threshold,out=get_threshold_plot(results.values())
    '''
    results=combine(results)
    n=len(results[1])
    threshold,x=get_threshold_plot(results[0])
    threshold,y=get_threshold_plot(results[1])
    f=interpolate.interp1d(x,y)
    out=[0 for i in threshold]
    for i in range(len(threshold)):
        try:
            out[i]=f(threshold[i])
        except ValueError:
            continue'''
    return np.array(out),n

def prob(x):
    post=0
    first=0
    k=1.0
    while True:
        to_add=(-1)**(k-1)
        to_add*=np.exp(0.0-2*(k*x)**2)
        post+=to_add
        if abs(first/to_add)>1000 or k>100:
            break
        else:
            if k==1.0:
                first=to_add
        k+=1
    return 1-2*post

def get_KS_stat(f0,fname):
    f1,n=get_cumulative(fname)
    D=max(abs(f1-f0))
    return D*sqrt(n)


ibcc,n=get_cumulative('ibcc')
print '\n'
for method in methods:
    stat=get_KS_stat(ibcc,method)
    print method,stat,np.log10(2)-2*(stat**2)*np.log10(np.e)
    
print '\ndone'



