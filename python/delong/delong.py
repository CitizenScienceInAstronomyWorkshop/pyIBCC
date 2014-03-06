'''
Created on 5 Mar 2013

@author: Kieran Finn
'''

import numpy as np
from functions import *
from scipy.special import erf, erfc
from math import sqrt
import pickle
from copy import copy

folder='D:/Documents/Planet_hunters/results/05_Mar_2013_18_27/'
#folder='D:/Documents/Planet_hunters/results/06_Mar_2013_14_43_01/'
#folder='csv_results/'

methods=['ibcc','weighted mean','mean','frequentist']

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
        try:
            out[labels[i]].append(results[i])
        except KeyError:
            pass
    return out

def prob(x,sigma,mu=0):
    #return 0.5*(1+erf(abs(x-mu)/(sqrt(2)*sigma)))
    return 0.5*erfc(abs(x-mu)/(sqrt(2)*sigma))

data={method:combine(pload(folder+method+'.dat')) for method in methods}
#data={method:combine(csvload(folder+method+'.csv')) for method in methods}
k=len(methods)
m=len(data[methods[0]][1])
n=len(data[methods[0]][0])

theta=[0 for i in range(k)]
print 'calculating theta'
for r in range(k):
    method=methods[r]
    print 'under %s method' %method
    try:
        post=pload(method+'_theta.dat')
    except:
        post=0.0
        for i in range(m):
            progress_bar(i,m)
            for j in range(n):
                post+=int(data[method][1][i]>data[method][0][j])
        pdump(post,method+'_theta.dat')
    theta[r]=float(post)/(m*n)
    print '\n'
theta=np.matrix(theta)
    
print 'calculating V_10'
V_10=[[0 for i in range(m)] for r in range(k)]
for r in range(k):
    method=methods[r]
    print 'under %s method' %method
    try:
        V_10[r]=pload(method+'_V10.dat')
    except:
        for i in range(m):
            progress_bar(i,m)
            post=0.0
            for j in range(n):
                post+=int(data[method][1][i]>data[method][0][j])
            V_10[r][i]=float(post)/n
        pdump(V_10[r],method+'_V10.dat')
    print '\n'
V_10=np.matrix(V_10)
    
print 'calculating V_01'
V_01=[[0 for j in range(n)] for r in range(k)]
for r in range(k):
    method=methods[r]
    print 'under %s method' %method
    try:
        V_01[r]=pload(method+'_V01.dat')
    except:
        for j in range(n):
            progress_bar(j,n)
            post=0.0
            for i in range(m):
                post+=int(data[method][1][i]>data[method][0][j])
            V_01[r][j]=float(post)/m
        pdump(V_01[r],method+'_V01.dat')
    print '\n'
V_01=np.matrix(V_01)

S_10=(1.0/(m-1))*(V_10*V_10.T+m*(theta.T*theta))
S_01=(1.0/(n-1))*(V_01*V_01.T+n*(theta.T*theta))

S=(1.0/m)*S_10+(1.0/n)*S_01

print S

l=[1,0,0,0]
print '\n'
for r in range(1,len(methods)):
    L=copy(l)
    L[r]=-1
    L=np.matrix(L)
    var=L*S*L.T
    sd=var[0,0]**0.5
    stat=(L*theta.T)[0,0]
    print methods[r],stat,var[0,0],stat/sd,prob(stat,sd)
    
print'\ndone'



