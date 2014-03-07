'''
Created on 18 Jan 2013

@author: Kieran Finn
'''
import urllib2
import sys
import json
from functions import *
import time
import pickle
from math import pi
from scipy.integrate import quad
import numpy as np

source_fname='q1_sources.csv'
in_fname='q1_detailed.csv'
out_fname='planet_specific.csv'

required_radius=[0.05,0.5]

dont_use='DO NOT USE'

def read_url(url):
    f=urllib2.urlopen(url)
    r=f.read()
    f.close()
    r=r[17:].rstrip('\n;)') #there is a problem with the json files for the data. need to strip some of the header
    try:
        out=json.loads(r)
    except:
        print 'error reading json file'
        print r[100:]+'...'+r[-100:]
        sys.exit()
    return out
    
def get_sources(fname):
    pickle_name=fname.split('.')[0]+'.pkl'
    try:
        f=open(pickle_name,'r')
        print 'reading data from pickle file %s' %pickle_name
        radii,data_points=pickle.load(f)
        f.close()
        return (radii,data_points)
    except:
        pass
    radii={}
    data_points={}
    f=open(fname,'r')
    r=f.readlines()
    f.close()
    count=0
    for line in r:
        if count%10==0:
            overprint('Processing line %s of %s' %(add_comma(count),add_comma(len(r))))
        count+=1
        i,url,rad=line.split(',')
        rad=float(rad)
        dat=read_url(url.strip('"'))
        radii[i]=rad
        data_points[i]=dat
    print '\n'
    out=(radii,data_points)
    f=open(pickle_name,'w')
    pickle.dump(out,f)
    f.close()
    return out

def get_data_points(x,y,height,width,data):
    out=[]
    for point in data:
        if x<point['x']<(x+width) and y<point['y']<(y+height):
            out.append(point)
    return out

def get_rad(x,y,height,width,data):
    points=get_data_points(x,y,height,width,data)
    if len(points)==0:
        return (0,0)
    depth=np.inf
    for i in points:
        if 1-i['y']<depth:
            depth=1-i['y']
            dy=i['dy']
    if depth<=0:
        return (0,0)
    else:
        return (depth,dy)

def get_score(depth,dy,rad):#this will need to be changed, probably to represent some kind of bayseian probability
    f=lambda x: (1.0/(((2*pi)**0.5)*dy))*np.exp(-((x-depth)**2)/(2*dy**2))#gaussian
    a,b=[(i/rad)**2 for i in required_radius]
    score=quad(f,a,b)
    if score>0.5:
        out='2'
    elif score>0.05:
        out='1'
    else:
        out='0'
    return out

print 'getting source data'
source_radii,source_data=get_sources(source_fname)
f=open(in_fname,'r')
r=f.readlines()
f.close()
f=open(out_fname,'w')

i=0
while i<len(r):
    class_id,_,user,light_curve,label=r[i].split(',')[:5]
    depth=[]
    dy=[]
    score=False
    if source_radii[light_curve]=='NULL' or source_radii[light_curve]==0:#we don't have enough data for this star so we won't use it
        score=dont_use
    while i<len(r) and class_id==r[i].split(',')[0]:
        if i%1000==0:
            overprint('Processing line %s of %s' %(add_comma(i),add_comma(len(r))))
        if score != dont_use:
            _,answer,_,_,_,x,y,height,width=r[i].split(',')
            if answer=='11':#this is a transit, calculate radius
                temp_depth,temp_dy=get_rad(float(x),float(y),float(height),float(width),source_data[light_curve])
                if temp_depth!=0:
                    depth.append(temp_depth)
                    dy.append(temp_dy)
        i+=1
    if score!= dont_use:
        if len(depth)==0:
            score='0'
        else:
            depth=sum(depth)/len(depth)
            dy=sum(dy)/len(dy)
            score=get_score(depth,dy,source_radii[light_curve])
        f.write('%s,%s,%s,%s\n' %(user,light_curve,score,label))
f.close()

print 'done'
                    

