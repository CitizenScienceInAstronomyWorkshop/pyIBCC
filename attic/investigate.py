'''
Created on 1 Feb 2013

@author: Kieran Finn
'''
import pylab as p
import urllib2
import sys
import json
from glob import glob
import pickle
from functions import *
from random import random
from collections import defaultdict
import webbrowser
import os

#folder='D:/Documents/Planet_hunters/results/01_Feb_2013_11_14/'
#folder='full_data_28_01_13'

folder='final_run/'
fname='investigate_full.csv'


data_folder='D:/Documents/Planet_hunters/ph-stars/'
if not os.path.isdir(data_folder):
    data_folder='/Users/kieranfinn/Documents/ph-stars/'


out=fname.split('.')[0]+'.pkl'

colormap=p.get_cmap()
colours=pload('user_scores.dat')
colours['NULL']=1.0

#colours=defaultdict(lambda:0.5)
zooniverse_dict=pload('source_zooniverse.dat')

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
        return {}
    try:
        out=out['data']#gets rid of meta data if there
    except:
        pass
    return out

def get_scores(folder):
    out={}
    files=glob(folder+'*.dat')
    #files=['weighted mean.csv','mean.csv','frequentist.csv','ibcc.csv']
    for fname in files:
        name=fname.split('\\')[-1].split('.')[0]
        out[name]=pload(fname)
        #out[name]=csvload(fname)
    return out
        

class box():
    def __init__(self,x,y,width,height,colour):
        self.x1=x
        self.x2=x+width
        self.y1=y
        self.y2=y+height
        self.colour=colour
        
    def points(self):
        x=[self.x1,self.x1,self.x2,self.x2,self.x1]
        y=[self.y1,self.y2,self.y2,self.y1,self.y1]
        return [x,y]
    
    def plot(self):
        x,y=self.points()
        p.plot(x,y,color=self.colour)
        
class light_curve():
    def __init__(self,light_curve_id,url,release_id):
        self.id=light_curve_id
        self.data=False
        self.release_id=release_id
        self.boxes=[]
        self.url=url
        
    def add_box(self,x,y,width,height,user):
        try:
            x=float(x)
            y=float(y)
            width=float(width)
            height=float(height)
            self.boxes.append(box(x,y,width,height,colormap(colours[user])))
        except:
            pass
        
    def get_data(self):
        if not self.data:
            self.data=read_url(self.url)
        return self.data
    
    def plot(self,clean=False):
        fig=p.figure(int(float(self.release_id)*10))
        x=[]
        y=[]
        dy=[]
        for point in self.get_data():
            try:
                x.append(float(point['x']))
                y.append(float(point['y']))
                dy.append(float(point['dy']))
            except:
                print point
                print 'error with point'
                continue
        p.errorbar(x,y,yerr=dy,fmt='ro')
        p.title('light curve id: %s\nrelease id: %s' %(self.id,self.release_id))
        if not clean:
            for box in self.boxes:
                box.plot()
            ax=fig.axes[0]
            ax=p.mpl.colorbar.make_axes(ax)[0]
            p.mpl.colorbar.ColorbarBase(ax,cmap=colormap)
        
            
class source():
    def __init__(self,source_id,label):
        self.id=source_id
        self.light_curves={}
        self.label=label
        
    def get_light_curve(self,light_curve_id,url,release_id):
        if light_curve_id not in self.light_curves.keys():
            self.light_curves[light_curve_id]=light_curve(light_curve_id,url,release_id)
        return self.light_curves[light_curve_id]
    
    def plot(self,clean=False):
        print 'Details for source number %s' %self.id
        print 'Label: %s' %self.label
        for method in scores.keys():
            try:
                print '%s: %.4f' %(method, scores[method][self.id])
            except:
                pass
        for light_curve_id in self.light_curves.keys():
            self.light_curves[light_curve_id].plot(clean=clean)
            
        
def read_data(fname):
    out={}
    f=open(fname,'r')
    r=f.readlines()
    f.close()
    count=0
    for line in r:
        if count%100==0:
            overprint('processing line %s of %s' %(add_comma(count),add_comma(len(r))))
        count +=1
        source_id,light_curve_id,user_id,label,url,release_id,x,y,width,height=line.strip().split(',')
        url=url.strip('"')
        if source_id not in out.keys():
            out[source_id]=source(source_id,label)
        lc=out[source_id].get_light_curve(light_curve_id,url,release_id)
        lc.add_box(x,y,width,height,user_id)
    print '\n'
    return out

def openpage(s):
    url='http://www.planethunters.org/sources/%s' %zooniverse_dict[s]
    webbrowser.open(url)

def user_plots():
    while True:
        source_id=raw_input('Enter the source id: ')
        if source_id=='0':
            break
        p.close('all')
        try:
            sources[source_id].plot()
            openpage(source_id)
        except KeyError:
            print 'not a valid source id'
            
def transit_plot(source,release):
    p.close('all')
    source=str(source)
    sources[source].plot(clean=True)
    light=''
    for lc in sources[source].light_curves.keys():
        if sources[source].light_curves[lc].release_id==str(release):
            light=lc
    p.figure(int(release*10))
    p.title('Source id: %s\nLight Curve id: %s    Release id: %.1f' %(source,light,release))
    p.xlabel('Day')
    p.ylabel('Relative Luminosity')

scores=get_scores(folder)
#scores=defaultdict(lambda:defaultdict(lambda:1))
sources=read_data(fname)
'''for source_id in sources.keys():
    for light_curve_id in sources[source_id].light_curves.keys():
        sources[source_id].light_curves[light_curve_id].colours=dict(sources[source_id].light_curves[light_curve_id].colours)
f=open(out,'w')
pickle.dump(sources,f)
f.close()'''
print 'done'
            
        
        
    
        