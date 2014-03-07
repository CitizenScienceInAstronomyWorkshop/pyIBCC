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
from collections import defaultdict

source_fname='q1_sources.csv'
in_fname='q1_detailed.csv'
out_fname='planet_specific.csv'
rad_fname='radius_data.csv'

required_radius=[0.05,0.5]

dont_use='DO NOT USE'

def read_url(url):
    try:
        f=urllib2.urlopen(url)
    except:
        print 'Problem with url %s' %url
        return []
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
source_radii,source_urls=get_sources(source_fname)


data=defaultdict(lambda: defaultdict(list))
f=open(in_fname,'r')
nlines=0
for line in f:
    nlines+=1
    words=line.split(',')#this line may need to be changed if the data is in another format (extension, make more general)
    light_curve,class_id=words[:2]
    data[light_curve][class_id].append(words[2:])   
f.close()
f=open(out_fname,'w')
g=open(rad_fname,'w')

count=0
for light_curve in data.keys():
    radius=source_radii[light_curve]
    source_data=read_url(source_urls[light_curve])
    for class_id in data[light_curve].keys():
        if source_radii[light_curve]=='"NULL"' or source_radii[light_curve]==0:#we don't have enough data for this star so we won't use it
            count+=len(data[light_curve][class_id])
            continue
        depth=[]
        dy=[]
        for line in data[light_curve][class_id]:
            if count%10==0:
                overprint('Processing line %s of %s' %(add_comma(count),add_comma(nlines)))
            count+=1
            answer,user,label,x,y,height,width=line
            if answer=='11':
                temp_depth,temp_dy=get_rad(float(x),float(y),float(height),float(width),source_data)
                if temp_depth>0:#will cause errors if negative depth
                    depth.append(temp_depth)
                    dy.append(temp_dy)
        if len(depth)==0:
            score='0'
        else:
            depth=sum(depth)/len(depth)
            dy=sum(dy)/len(dy)
            score=get_score(depth,dy,source_radii[light_curve])
            g.write('%s,%s,%s,%f,%f,%f\n' %(user,light_curve,label,depth,dy,source_radii[light_curve]))
        f.write('%s,%s,%s,%s\n' %(user,light_curve,score,label))
        
f.close()
g.close()
print 'done'
                    

