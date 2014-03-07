'''
Created on 23 Jan 2013

@author: Kieran Finn
'''
import pickle
from functions import *
import numpy as np
import catagorize
import pylab as p
from mpl_toolkits.mplot3d import axes3d #used for 3d axese

def get3dax(): #returns a pylab 3d axis instance labelled with x, y and z
    f=p.figure()
    out=f.gca(projection='3d')
    return out

in_name='alpha.pkl'
user_fname='users.pkl'
N=500 #number to catagorize

def score(a): #currently doesn't take account of size of alphas, just confusion matrix
    a=[[float(a[j][l])/float(sum(a[j])) for l in range(2)] for j in range(2)]
    sc=a[0][0]+a[1][1]-a[1][0]-a[0][1]
    return sc

f=open(in_name,'r')
alpha=pickle.load(f)
f.close()
f=open(user_fname,'r')
users=pickle.load(f)
f.close()
print '%s users' %add_comma(len(alpha))

scores={i:score(alpha[i]) for i in alpha.keys()}

people=sorted(scores, key=scores.get)#ordered List
cpeople=people[::len(people)/N]#takes the best N people, may want to change that

pi={i:[[float(alpha[i][j][l])/float(sum(alpha[i][j])) for l in range(2)] for j in range(2)] for i in people}

#communities,index2id=catagorize.main(pi)


def plot_alpha(alpha):
    colours=['b','r','g','y']
    pi=[[float(alpha[j][l])/float(sum(alpha[j])) for l in range(2)] for j in range(2)] 
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
    
p.close('all')
f=open('user_check.csv','r')
users=[]
for line in f:
    try:
        user=line.strip()
        a=alpha[user]
        if user not in users:
            users.append(user)
            plot_alpha(a)
            p.title(user)
    except:
        print 'no user named %s' %line
f.close()