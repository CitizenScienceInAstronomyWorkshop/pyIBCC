'''
Created on 27 Feb 2013

@author: Kieran Finn
'''
import pylab as p
from functions import *
import numpy as np
from collections import defaultdict
import urllib2
import os
import json
data_folder='D:/Documents/Planet_hunters/ph-stars/'
fname='check_periodicity.csv'

def open_url(url):
    N=3
    n=0
    while n<N:
        try:
            return urllib2.urlopen(url)
        except:
            n+=1
    return False
    
def read_url(url,meta=False):
    fname=data_folder+url.split('/')[-1]
    if not os.path.isfile(fname) or os.path.getsize(fname)==0:
        print '\n%s not downloaded' %fname
        try:
            f=open_url(url)
            g=open(fname,'w')
            g.write(f.read())
            g.close()
            f.close()
        except:
            print 'Problem with url %s' %url
            return []
    f=open(fname,'r')
    r=f.read()
    f.close()
    if r[0]=='l':
        r=r[17:].rstrip('\n;)') #there is a problem with some of the json files for the data. need to strip some of the header  
    try:
        out=json.loads(r)
    except:
        print 'error reading json file'
        print r[:100]+'...'+r[-100:]
        return {}
    try:
        if meta:
            out=out['meta_data']
        else:
            out=out['data']#gets rid of meta data if there
    except:
        pass
    return out

out=defaultdict(list)
periods={}
to_plot=[]
f=open(fname,'r')
r=f.readlines()
f.close()
count=0
for line in r:
    if count%10==0:
        overprint('processing line %s of %s' %(add_comma(count),add_comma(len(r))))
    count+=1
    lc_id,url,period=line.split(',')
    try:
        period=float(period)
    except:
        print '\nperiod is %s for %s' %(period.strip(),lc_id)
        continue
    data=read_url(url.strip('"'))
    try:
        data[0]['tr']
    except:
        print '\nno transits in %s' %lc_id
        continue
    periods[lc_id]=period
    old=False
    startend={1:-1,0:-1}
    flag=0
    i=0
    while i<len(data):
        tr=int(float(data[i]['tr']))
        if tr!=flag:
            startend[tr]=float(data[i-1+tr]['x'])
            flag=tr
            if startend[0]>0:
                new=(startend[0]+startend[1])/2
                startend={1:-1,0:-1}
                if old:
                    expected=np.round((new-old)/period)*period
                    difference=abs(new-old-expected)/period
                    out[lc_id].append(difference)
                    to_plot.append(difference)
                else:
                    old=new
        i+=1
        
p.hist(to_plot,bins=np.arange(0,1.01,0.01))
p.show()
        
    