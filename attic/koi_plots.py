'''
Created on 17 Apr 2013

@author: Kieran Finn
'''
import pickle
from functions import *
import pylab as p
import numpy as np
from collections import defaultdict

p.rcParams.update({'font.size': 18}) # makes the default text size larger
p.rc('text', usetex=True) #allows the use of latex expressions for labels

folder='final_run'
fname='koi_details.csv'
headings=[[1,'Semi Major Axis',0.3,'$R_\odot$'],[2,'Radius',20,'$R_\oplus$']]
threshold=0.8
    
def getplot(data,mx,bins=25):
    resolution=float(mx)/bins
    bins=np.arange(0,mx+resolution,resolution)
    hist,bins=np.histogram(data,bins=bins)
    centre=(bins[:-1]+bins[1:])/2
    return [centre,hist.astype(np.float)]
                 
scores=pload(folder+'/ibcc.dat')
f=open(fname,'r')
r=f.readlines()
f.close()
koi=defaultdict(list)
found=defaultdict(list)
errors=0
total=0
for line in r:
    total+=1
    words=line.split(',')
    source=words[0].strip()
    try:
        score=scores[source]
    except KeyError:
        print 'Error, no score for source %s' %source
        errors+=1
        continue
    for i in range(1,len(words)):
        try:
            x=float(words[i])
        except:
            continue
        koi[i].append(x)
        if score>threshold:
            found[i].append(x)
print 'no score for %d of %d' %(errors,total)

for header in headings:
    index,title,mx,units=header
    p.figure()
    x,y1=getplot(koi[index],mx)
    x,y2=getplot(found[index],mx)
    y=y2/y1
    width=x[1]-x[0]
    p.bar(x,y,align='center',width=width,yerr=1.0/np.sqrt(y1),ecolor='r')
    p.xlim(0,mx)
    p.xlabel(title+'/'+units)
    p.ylabel('Fraction of planets found')
p.show()