'''
Created on 18 Jan 2013

@author: Kieran Finn
'''
import time
beginning=time.time()
import urllib2
import sys
import json
from functions import *
import time
import pickle
from math import pi
from scipy.integrate import quad
import numpy as np
from collections import defaultdict
import os

in_fname='interesting_detailed.csv'
out_fname='planet_specific.csv'
rad_fname='radius_data.csv'
data_folder='/Users/kieranfinn/Documents/ph-stars/'
data_folder='D:/Documents/Planet_hunters/ph-stars/'

required_radius=[0.05,1]
required_mass=[0.0004,1]

dont_use='DO NOT USE'

def open_url(url):
    N=3
    n=0
    while n<N:
        try:
            return urllib2.urlopen(url)
        except:
            n+=1
    return False

def read_url(url):
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
    try:
        if r[0]=='l':
            r=r[17:].rstrip('\n;)') #there is a problem with some of the json files for the data. need to strip some of the header
    except:
        print url
        print fname
        print r
        
    try:
        out=json.loads(r)
    except:
        print 'error reading json file'
        print r[:100]+'...'+r[-100:]
        sys.exit()
    try:
        out=out['data']#gets rid of meta data if there
    except:
        pass
    return out
    
def get_sources(fname):
    radii={}
    urls={}
    f=open(fname,'r')
    r=f.readlines()
    f.close()
    for line in r:
        i,url,rad=line.split(',')
        rad=float(rad)
        urls[i]=url.strip('"')
        radii[i]=rad
    out=(radii,urls)
    return out

def get_data_points(x,y,height,width,data):
    out=[]
    indices=[]
    for i in range(len(data)):
        if x<data[i]['x']<(x+width) and y<data[i]['y']<(y+height):
            indices.append(i)
            out.append(data[i])
    if len(indices)!=0:
        indices=range(min(indices),max(indices)+1)
    return [out,indices]

def refine_indices(indices,length):
    min_length=200#minimum number of points required to set ambient light
    if len(indices)>min_length:
        return indices
    low=min(indices)
    high=max(indices)+1
    to_add=min_length-len(indices)
    front=range(low-(int(to_add/2)+1),low)
    back=range(high,high+int(to_add/2)+1)
    while len(front)!=0 and front[0]<0:#ensures data doesn't go out of range
        del front[0]
        back.append(back[-1]+1)
    while len(back)!=0 and back[-1]>=length:
        del back[-1]
        front=[front[0]-1]+front
    return front+indices+back
    

def get_rad(rad,x,y,height,width,data):
    points,indices=get_data_points(x,y,height,width,data)
    if len(points)==0:
        return 0
    mn=np.inf
    for point in points:
        if point['y']<mn:
            transit_x,transit_y=point['x'],point['y']#x and y coords of bottom transit      
    indices=refine_indices(indices,len(data))
    x=[]
    y=[]
    for i in indices:
        x.append(data[i]['x'])
        y.append(data[i]['y'])
    m,c=np.polyfit(x,y,1)#fit a line to the ambient data
    depth=m*transit_x+c-transit_y
    if depth>0:
        return rad*(depth**0.5)
    else:
        return 0

'''def get_score(depth,dy,rad):#this will need to be changed, probably to represent some kind of bayseian probability
    f=lambda x: (1.0/(((2*pi)**0.5)*dy))*np.exp(-((x-depth)**2)/(2*dy**2))#gaussian
    a,b=[(i/rad)**2 for i in required_radius]
    score=quad(f,a,b)
    if score>0.5:
        out='2'
    elif score>0.05:
        out='1'
    else:
        out='0'
    return out'''

def get_label(mass,label):
    mass=mass.strip('"')
    if mass=='Null':
        return label
    try:
        mass=float(mass)
    except ValueError:
        return label
    if required_mass[0]<mass<required_mass[1]:
        return 'jupiter'
    else:
        return label
    
        

data=defaultdict(lambda: defaultdict(list))
f=open(in_fname,'r')
source_radii={}
nlines=0
for line in f:
    nlines+=1
    words=line.split(',')#this line may need to be changed if the data is in another format (extension, make more general)
    source,user,rad=words[:3]
    data[source][user].append(words[3:])
    source_radii[source]=float(rad)  
f.close()
f=open(out_fname,'w')
g=open(rad_fname,'w')

count=0
for source in data.keys():
    radius=source_radii[source]
    for user in data[source].keys():
        if source_radii[source]=='"NULL"' or source_radii[source]==0:#we don't have enough data for this star so we won't use it
            count+=len(data[source][user])
            continue
        rad=[]
        for line in data[source][user]:
            if count%10==0:
                overprint('Processing line %s of %s' %(add_comma(count),add_comma(nlines)))
            count+=1
            answer,label,planet_mass,url,x,y,height,width=line
            if answer=='11':
                source_data=read_url(url.strip('"'))
                temp_rad=get_rad(source_radii[source],float(x),float(y),float(height),float(width),source_data)
                if temp_rad!=0:
                    rad.append(temp_rad)
        label=get_label(planet_mass,label)
        if len(rad)!=1:
            score=0
            rad=0
            source_rad=0
        else:
            rad=sum(rad)/len(rad)
            score=int(required_radius[0]<rad<required_radius[1])
            source_rad=source_radii[source]
        g.write('%s,%s,%s,%f,%f\n' %(user,source,label,rad,source_rad))
        f.write('%s,%s,%d,%s\n' %(user,source,score,label))
        
f.close()
g.close()
print 'Total running time: %s' %hms(time.time()-beginning)
print 'done'
                    

