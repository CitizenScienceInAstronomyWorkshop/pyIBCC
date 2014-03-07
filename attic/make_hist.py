'''
Created on 12 Feb 2013

@author: Kieran Finn
'''
import pickle
import pylab as p
from collections import defaultdict
import numpy as np

fname='koi_3.csv'
results='D:/Documents/Planet_hunters/results/12_Feb_2013_13_12/ibcc.dat'

f=open(results,'r')
results=pickle.load(f)
f.close()

def make_hist(name,data):
    sigma=3
    n=len(data)
    f,ax=p.subplots(n,sharex=True)
    i=0
    for key in data.keys():
        dat=data[key]
        mean=np.mean(dat)
        sd=abs(np.std(dat))
        temp=[]
        for datum in dat:#remove outliers for cleaner plot
            if datum>mean-sigma*sd and datum<mean+sigma*sd:
                temp.append(datum)
        ax[i].title.set_text(key)
        ax[i].hist(temp,bins=1000)
        i+=1
    p.xlabel(name)
    
def get_threshold_plot(data,normalise=True,threshold=np.array([])):
    ''' may be a quicker way. also may need to amend to calculate auc'''
    if threshold.any():
        resolution=threshold[1]-threshold[0]
    else:
        sd=abs(np.std(data))
        resolution=1.0*sd/len(data)
        threshold=np.arange(0,max(data),resolution)
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
    return (threshold,fraction)

def get_plot(All,found):
    threshold,All=get_threshold_plot(All,normalise=False)
    _,found=get_threshold_plot(found,normalise=False,threshold=threshold)
    out=[found[i]/All[i] for i in range(len(threshold))]
    return (threshold,out)
    
def make_curve(name,data):
    p.figure()
    p.title(name)
    x,y=get_plot(data['All'],data['Found'])
    p.plot(x,y,label=key)
    p.legend()
    p.xlabel(name)
    p.ylabel('Fraction')
        
def get_headers(s):
    names=[i.strip().strip('"') for i in s.split(',')]
    out={i:names.index(i) for i  in names}
    return out

def get_keys(words,headers):
    source=words[headers['id']]
    try:
        res=results[source]
    except KeyError:
        res=0
    if res>0.8:
        return ['All','Found']
    else:
        return ['All']

f=open(fname,'r')
r=f.readlines()
f.close()
headers=get_headers(r[0])
to_plot=headers.keys()
for i in ['id','kind']:
    try:
        to_plot.remove(i)
    except ValueError:
        pass
    
data={i:defaultdict(list) for i in to_plot}
for line in r[1:]:
    words=[i.strip().strip('"') for i in line.split(',')]
    keys=get_keys(words,headers)
    for header in to_plot:
        for key in keys:
            try:
                data[header][key].append(float(words[headers[header]]))
            except ValueError:
                pass
                
            
for header in to_plot:
    make_curve(header,data[header])
p.show()
    
    