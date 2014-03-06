'''
Created on 11 Feb 2013

@author: Kieran Finn
'''
import numpy as np
import sys
from random import random
import pylab as p
from mpl_toolkits.mplot3d import axes3d #used for 3d axese
from math import pi, sqrt
from copy import deepcopy

I=2

def get3dax(): #returns a pylab 3d axis instance
    f=p.figure()
    out=f.gca(projection='3d')
    return out

def sum(l):
    out=np.zeros_like(l[0])
    for i in l:
        out+=i
    return out

def random_pi():
    a=random()
    b=random()
    return np.array([[a,1-a],[b,1-b]])

def plot_communities(communities):
    colours=['b','r','g','y']    
    for pi in communities:
        dx=[]
        dy=[]
        dz=[]
        x=[]
        y=[]
        z=[]
        c=[]
        for i in range(len(pi)):
            for j in range(len(pi[0])):
                dz.append(pi[i][j])
                dx.append(1)
                dy.append(1)
                x.append(i-0.5)
                y.append(j-0.5)
                z.append(0)
                c.append(colours[j])
        ax=get3dax()
        ax.set_xlabel('True Label')
        ax.set_ylabel('Score')
        ax.set_zlabel('Probability')
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.bar3d(x,y,z,dx,dy,dz,color=c)
        ax.set_zlim(0,1)

def hd(pi1,pi2):
    return 1-np.sum([np.sqrt(pi2[i]*pi1[i]) for i in range(len(pi1))])

def square(_x):
    out=0
    for i in range(len(_x)):
        for j in range(len(_x[0])):
            out+=_x[i][j]**2
    out=_x[0][0]**2+_x[1][1]**2
    return out

def d(pi1,pi2):
    out=0
    for i in range(len(pi1)):
        out+=hd(pi1[i],pi2[i])**2
    return sqrt(out)

def get_r(_w,_sigma,_m,_x):
    expo=0.0-(1.0/_sigma)*d(_m,_x)
    pre=(_w/sqrt(2*pi*_sigma))**I
    return pre*np.exp(expo)

def check_convergence(old,new):
    print old
    print new
    if raw_input('>>')=='y':
        sys.exit()
    out=[]
    for i in range(len(old)):        
        temp=(old[i]-new[i])
        out.append(square(temp))
    med=np.median(out)
    print med
    return (med<1e-4)

def main(x):
    x=[np.array(i) for i in x]
    N=len(x)
    K=3
    m=[random_pi() for k in  range(K)]
    w=[1.0 for k in range(K)]
    sigma=[1e-2 for k in range(K)]
    
    count=1
    converged=False
    while not converged:
        print 'Iteration %d' %count
        count+=1
        old=deepcopy(m)
        r=[[get_r(w[k],sigma[k],m[k],x[n]) for n in range(N)] for k in range(K)]
        for k in range(K):
            norm=sum([r[k][n] for n in range(N)])
            for n in range(N):
                r[k][n]/=norm
                
        R=[sum([r[k][n] for n in range(N)]) for k in range(K)]
        m=[sum([r[k][n]*x[n] for n in range(N)])/R[k] for k in range(K)]
        converged=check_convergence(old,m)
        sigma=[sum([r[k][n]*square(x[n]-m[k]) for n in range(N)])/(I*R[k]) for k in range(K)]
        norm=sum([R[k] for k in range(K)])
        w=[R[k]/norm for k in range(K)]
        
    plot_communities(m)
    return m
        
    
