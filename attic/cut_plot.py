'''
Created on 1 Mar 2013

@author: Kieran Finn
'''
import pylab as p
import numpy as np
from collections import defaultdict

def barchart(dictionary,colour='b'):
    '''creates a barchart from a dictionary dictionary={label:frac}'''
    labels=[]
    fracs=[]
    indices=[]
    i=0
    width=0.35
    for key in dictionary:
        indices.append(i)
        labels.append(key)
        fracs.append(dictionary[key])
        i+=1
    indices=np.array(indices)
    p.bar(indices,fracs,width,color=colour)
    p.xticks(indices+width/2., labels )
    return True

old=defaultdict(int)
new=defaultdict(int)

f=open('pre_cut.csv','r')
for line in f:
    old[line.strip().strip('"')]+=1
f.close()

f=open('D:/Documents/Planet_hunters/results/01_Mar_2013_11_56/candidate_list.csv','r')
for line in f:
    kind=line.split(',')[-1].strip().strip('"')
    new[kind]+=1
f.close()

p.title('result of cuts')
p.ylabel('number of new candidates')
barchart(old)
barchart(new,'r')
p.show()
