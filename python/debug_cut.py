'''
Created on 19 Feb 2013

@author: Kieran Finn
'''
import time
beginning=time.time()
import pylab as p
import urllib2
import sys
import numpy as np
import json
from glob import glob
import cPickle as pickle
import mean
from functions import *
from random import random
import ibcc
from collections import defaultdict
import os
import itertools
import settings

folder='D:/Documents/Planet_hunters/results/01_Feb_2013_11_14/'
#folder='full_data_28_01_13'
fname='investigate.csv'
fname=settings.data_file
data_folder='D:/Documents/Planet_hunters/ph-stars/'
#data_folder='/Users/kieranfinn/Documents/ph-stars/'
out_folder="D:/Documents/Planet_hunters/results/"
out=fname.split('.')[0]+'_transits.dat'
linking_length=0.2
max_allowed_depth=0.001

all_transits={}#a dictionary which contains the details of  every transit
source_labels={}
light_curve_details={}

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
    if r[0]=='l':
        r=r[17:].rstrip('\n;)') #there is a problem with some of the json files for the data. need to strip some of the header  
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
    for fname in files:
        name=fname.split('\\')[-1].split('.')[0]
        f=open(fname,'r')
        out[name]=pickle.load(f)
        f.close()
    return out
        

class box():
    def __init__(self,x,y,width,height,user):
        self.x1=x
        self.x2=x+width
        self.y1=y
        self.y2=y+height
        self.user=user
        self.centre=x+0.5*width
        self.id=os.urandom(32)
        
    def points(self):
        x=[self.x1,self.x1,self.x2,self.x2,self.x1]
        y=[self.y1,self.y2,self.y2,self.y1,self.y1]
        return [x,y]
    
    def contains(self,x,y):
        x=self.x1<=x<=self.x2
        y=self.y1<=y<=self.y2
        if (x and y):
            return [self.user]
        else:
            return []
    
    def plot(self):
        x,y=self.points()
        p.plot(x,y)
        
    def get_data_points(self,data):
        out=[]
        indices=[]
        for i in range(len(data)):
            if self.x1<float(data[i]['x'])<self.x2:
                indices.append(i)
                out.append(data[i])
        if len(indices)!=0:
            indices=range(min(indices),max(indices)+1)
        return [out,indices]
    
    def refine_indices(self,indices,length):
        min_length=100#minimum number of points required to set ambient light
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
        
    
    def get_depth(self,data):
        points,indices=self.get_data_points(data)
        if len(points)==0:
            return 0
        transit_x=0
        transit_y=0
        mn=np.inf
        for point in points:
            if float(point['y'])<mn:
                transit_x,transit_y=float(point['x']),float(point['y'])#x and y coords of bottom transit      
        indices=self.refine_indices(indices,len(data))
        x=[]
        y=[]
        for i in indices:
            x.append(float(data[i]['x']))
            y.append(float(data[i]['y']))
        m,c=np.polyfit(x,y,1)#fit a line to the ambient data
        expected_y=m*transit_x+c
        return (expected_y-transit_y)/expected_y
    
    def pickleable(self):
        return {'id':self.id,
                'x1':self.x1,
                'x2':self.x2,
                'y1':self.y1,
                'y2':self.y2,
                'user':self.user}

        
class transit(box):
    def __init__(self,group):
        centre=sum([box.centre for box in group])/len(group)
        width=sum([box.x2-box.x1 for box in group])/len(group)
        self.x1=centre-0.5*width
        self.x2=centre+0.5*width
        self.y1=0
        self.y2=2 #these are probably unessesery but include them so I can use all the box methods
        self.users=set([box.user for box in group])
        self.id='%.3f' %centre#need a unique id which can identify the transit so use x coordinate
    
    def pickleable(self):
        return {'id':self.id,
                'x1':self.x1,
                'x2':self.x2,
                'y1':self.y1,
                'y2':self.y2,
                'users':self.users}
        
       
        
class light_curve():
    def __init__(self,light_curve_id,url,release_id):
        self.id=light_curve_id
        self.url=url
        self.release_id=release_id
        self.boxes=[]
        self.users=[]
        self.box_tree=defaultdict(list)
        self.box_resolution=2*linking_length
        self.data=False
        self.scores=False
        self.transits=False
        light_curve_details[self.id]=[url,release_id]
        
    def add_box(self,x,y,width,height,user):
        try:
            x=float(x)
            y=float(y)
            width=float(width)
            height=float(height)
            if width<2:#ignore any boxes that are too wide
                new_box=box(x,y,width,height,user)
                self.boxes.append(new_box)
                index=int(np.round(new_box.centre/self.box_resolution))
                self.box_tree[index].append(new_box)
        except:
            pass
        self.users.append(user)
        
    def get_data(self):
        if not self.data:
            self.data=read_url(self.url)
        return self.data
    
    def score(self,x,y):
        out=[]
        for box in self.boxes:
            out+=box.contains(x,y)
        return len(set(out))
    
    def get_scores(self):
        if not self.scores:
            self.scores=[]
            for point in self.get_data():
                try:
                    self.scores.append(self.score(float(point['x']),float(point['y'])))
                except:
                    print point
                    print 'error with point'
                    continue
        return self.scores        
    
    def find_transits(self):
        '''uses friends of friends to catagorize the marked boxes into transits'''
        if not self.transits:
            to_add=copy(self.box_tree)
            out=[]
            while True:
                try:
                    a=itertools.chain(*to_add.values()).next()
                    group=[a]
                except StopIteration:
                    break
                for box in group:
                    index=int(box.centre/self.box_resolution)
                    for i in [index,index+np.sign(box.centre)]:#int rounds positive numbers down and positive numbers up
                        temp=copy(to_add[i])
                        for new in to_add[i]:
                            if abs(new.centre-box.centre)<linking_length:
                                temp.remove(new)
                                group.append(new)
                        to_add[i]=temp
                tran=transit(set(group))
                out.append(tran)
                all_transits[self.id+'_'+tran.id]=[tran.x1,tran.x2-tran.x1]
            self.transits=out
        return self.transits
    
    def output(self):
        objects=[]
        people=[]
        scores=[]
        labels=[]
        users=set(self.users)
        for tran in self.find_transits():
            people+=list(tran.users)
            people+=list(users.difference(tran.users))
            scores+=[1 for i in range(len(tran.users))]+[0 for i in range(len(users)-len(tran.users))]
            objects+=[self.id+'_'+tran.id for i in range(len(users))]
            labels+=[0 for i in range(len(users))]
        return (objects,people,scores,labels)
    
    def cleanup(self):
        out=False
        temp=[]
        for tran in self.boxes:
            depth=tran.get_depth(self.get_data())
            if depth<0 or depth>max_allowed_depth:#a binary transit
                pass
            else:
                temp.append(tran)
                out=True
        self.boxes=temp#shouldn't remove objects for a list being iterated over
        self.data=False#this should free up some memory. Hopefully.
        return out
                
    def pickleable(self):
        out={'id':self.id,
             'url':self.url,
             'release_id':self.release_id,
             'users':self.users}
        boxes=[bx.pickleable() for bx in self.boxes]
        out['boxes']=boxes
        return out
            
    def plot(self,clean=False,transits=False):
        p.figure()
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
            to_plot= self.find_transits() if transits else self.boxes
            for box in to_plot:
                box.plot()
        
            
class source():
    def __init__(self,source_id,label):
        self.id=source_id
        self.light_curves={}
        self.label=label
        
    def get_light_curve(self,light_curve_id,url,release_id):
        if light_curve_id not in set(self.light_curves.keys()):
            self.light_curves[light_curve_id]=light_curve(light_curve_id,url,release_id)
        return self.light_curves[light_curve_id]
    
    def output(self):
        objects=[]
        people=[]
        labels=[]
        scores=[]
        for lc in self.light_curves.values():
            new_objects,new_people,new_scores,new_labels=lc.output()
            objects+=new_objects
            people+=new_people
            scores+=new_scores
            labels+=new_labels
        #labels=[self.label for i in range(len(objects))]
        objects=[self.id+'_'+obj for obj in objects]
        return (objects,people,scores,labels)    
            
    def cleanup(self):
        out=False
        for lc in self.light_curves.values():
            out=out or lc.cleanup()#source is real candidate if ANY light curves pass the test
        return out
    
    def pickleable(self):
        out={'id':self.id,
             'label':self.label}
        lcs=[lc.pickleable() for lc in self.light_curves.values()]
        out['light_curves']=lcs
        return out
        
    def plot(self,clean=False,transits=False):
        print 'Details for source number %s' %self.id
        print 'Label: %s' %self.label
        for method in scores.keys():
            print '%s: %.4f' %(method, scores[method][self.id])
        for light_curve_id in self.light_curves.keys():
            self.light_curves[light_curve_id].plot(clean=clean,transits=transits)
            
        
def read_data(fname):
    out={}
    f=open(fname,'r')
    r=f.readlines()
    f.close()
    count=0
    for line in r:
        if count%1000==0:
            overprint('processing line %s of %s' %(add_comma(count),add_comma(len(r))))
        count +=1
        source_id,light_curve_id,user_id,label,url,release_id,x,y,width,height=line.strip().split(',')
        source_labels[source_id]=label
        url=url.strip('"')
        if source_id not in set(out.keys()):
            out[source_id]=source(source_id,label)
        lc=out[source_id].get_light_curve(light_curve_id,url,release_id)
        lc.add_box(x,y,width,height,user_id)
    print '\n'
    return out
'''main program'''

dir_name=out_folder+date_string()#have this as a new folder to avoid overwriting old files
orig_dir=dir_name
end=1
while True:
    try:
        os.mkdir(dir_name)
        break
    except WindowsError:
        dir_name=orig_dir+'_%02d' %end
        end+=1
settings.dir_name=dir_name
print 'Storing data in %s' %dir_name

print 'finding previous scores from %s' %folder
scores=get_scores(folder)
print 'Reading data from %s' %fname
sources=read_data(fname)
print 'sources before cuts: %s' %add_comma(len(sources))

out_objects=longlist([],500000)
out_people=longlist([],500000)
out_scores=longlist([],500000)
out_labels=longlist([],500000)
count=1
for current_source in sources.values():
    overprint('finding transits for source %s of %s. id=%s' %(add_comma(count),add_comma(len(sources)),current_source.id))
    count+=1
    new_objects,new_people,new_scores,new_labels=current_source.output()
    out_objects.add(new_objects)
    out_people.add(new_people)
    out_scores.add(new_scores)
    out_labels.add(new_labels)
  
f=open(dir_name+'/source_transits_lc_details.dat','w')
pickle.dump([source_labels,all_transits,light_curve_details],f)
f.close()

del sources #free up some memory
print '\n'
print 'total transits found: %s' %add_comma(len(out_objects))
f=open(dir_name+'/'+out,'w')
for i in range(len(out_objects)):
    f.write('%s,%s,%d,%s\n' %(out_objects[i],out_people[i],out_scores[i],out_labels[i]))
f.close()
    
print 'running mean algorithm on transits to find most promising ones'
results=mean.mean([out_objects,out_people,out_scores,out_labels])

f=open(dir_name+'/transit_results.dat','w')
pickle.dump(results,f)
f.close()

sources={}
for tran in results.keys():
    if results[tran]>0:#this line may need to be edited
        source_id,light_curve_id,tran_id=tran.split('_')
        x,width=all_transits[light_curve_id+'_'+tran_id]
        url,release_id=light_curve_details[light_curve_id]
        if source_id not in set(sources.keys()):
            sources[source_id]=source(source_id,source_labels[source_id])
        lc=sources[source_id].get_light_curve(light_curve_id,url,release_id)
        lc.add_box(x,0,width,2,tran_id)#again, using 0-2 for y axis to include everything. May want to make this more sophisticated at some point

print 'sources after first round of cuts: %s' %add_comma(len(sources))

f=open(dir_name+'/pickled_sources.dat','w')
pickle.dump({i:sources[i].pickleable() for i in sources.keys()},f)
f.close()

print 'Total running time: %s' %hms(time.time()-beginning)
settings.close_all()