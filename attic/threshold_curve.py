'''
Created on 12 Feb 2013

@author: Kieran Finn
'''
import pickle
import pylab as p
from collections import defaultdict
import numpy as np
from functions import overprint,add_comma

def read_file(folder,name):
    fname=folder+name+'.dat'
    f=open(fname,'r')
    out=pickle.load(f)
    f.close()
    return out

folder='D:/Documents/Planet_hunters/results/12_Feb_2013_13_12/'
comb_methods=['mean','weighted mean','ibcc','frequentist']
folder='D:/Documents/Planet_hunters/results/12_Feb_2013_13_12/'
comb_methods={method:read_file(folder,method) for method in comb_methods}

def get_threshold_plot(data,normalise=True):
    ''' may be a quicker way. also may need to amend to calculate auc'''
    resolution=0.0001
    threshold=np.arange(0,1.0+resolution,resolution)
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
    for i in range(len(fraction)-1)[::-1]:#this should cumulate the results without many sums
        fraction[i]+=fraction[i+1]
    return (threshold,fraction)

def get_plot(candidates,koi):
    threshold,candidates=get_threshold_plot(candidates,normalise=False)
    _,koi=get_threshold_plot(koi,normalise=False)
    out=[koi[i]/candidates[i] for i in range(len(threshold))]
    return (threshold,out)

def make_curve(name,data):
    p.figure()
    p.title(name)
    for key in data.keys():
        x,y=get_plot(data[key]['candidates'],data[key]['koi'])
        p.plot(x,y,label=key)
    p.legend()
    p.xlabel('Threshold')
    p.ylabel('Fraction')

keys=['candidates','koi']
    
data=defaultdict(lambda:{i:[] for i in keys})
for key in keys:
    print '\n'+key
    fname=key+'.csv'
    f=open(fname,'r')
    r=f.readlines()
    f.close()
    count=0
    for line in r:
        if count%100==0:
            overprint('processing line %s of %s' %(add_comma(count),add_comma(len(r))))
        count+=1
        source=line.strip()
        for method in comb_methods.keys():
            try:
                data[method][key].append(comb_methods[method][source])
            except KeyError:
                pass
                
make_curve('Fraction of new candidates which are KOI',data)
p.show()
    
    