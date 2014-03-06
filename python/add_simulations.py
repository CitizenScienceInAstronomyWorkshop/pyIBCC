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
import shelve

zero_dict=shelve.open('zeros.dat')
transit_dict=shelve.open('simulation_transit_details.dat')
data_folder='D:/Documents/Planet_hunters/ph-stars/'
fname='check_periodicity_2.csv'

temp_trans=defaultdict(list)
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

def zero(release_id,url):
    offset=131.511
    if release_id=='1.0':
        return offset
    major,minor=release_id.split('.')
    if major=='2':
        return 30*float(minor)+offset
    data=read_url(url,meta=True)
    try:
        return data['start_time']
    except:
        return False
    
def get_zero(lc_id,release_id,url):
    try:
        return zero_dict[lc_id]
    except KeyError:
        out=zero(release_id,url)
        if out:
            zero_dict[lc_id]=out
            return out
        else:
            print 'error calculating zero for light curve '+lc_id
            return 0
            
            
f=open(fname,'r')
r=f.readlines()
f.close()
count=0
for line in r:
    if count%10==0:
        overprint('processing line %s of %s' %(add_comma(count),add_comma(len(r))))
    count+=1
    lc_id,url,release_id,period=line.split(',')
    try:
        period=float(period)
    except:
        print 'period is %s for %s' %(period.strip(),lc_id)
        continue
    data=read_url(url.strip('"'))
    try:
        data[0]['tr']
    except:
        print 'no transits in %s' %lc_id
        continue
    startend={1:-1,0:-1}
    flag=0
    i=0
    while i<len(data):
        tr=int(float(data[i]['tr']))
        if tr!=flag:
            startend[tr]=float(data[i-1+tr]['x'])
            flag=tr
            if startend[0]>0:
                centre=(startend[0]+startend[1])/2
                startend={1:-1,0:-1}
                temp_trans[lc_id].append(centre)
        i+=1
for i in temp_trans.keys():
    transit_dict[i]=temp_trans[i]
transit_dict.close()