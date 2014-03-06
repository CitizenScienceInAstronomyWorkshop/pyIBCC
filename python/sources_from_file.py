'''
Created on 24 Feb 2013

@author: Kieran Finn
'''
import pylab as p
import urllib2
import sys
import numpy as np
import json
from glob import glob
import cPickle as pickle
from functions import *
from random import random
import ibcc
from collections import defaultdict
import os
import itertools
import settings
import shelve

p.rcParams.update({'font.size': 18}) # makes the default text size larger

in_file='D:/Documents/Planet_hunters/results/06_Mar_2013_16_42/pickled_sources.dat'

data_folder='D:/Documents/Planet_hunters/ph-stars/'
#data_folder='/Users/kieranfinn/Documents/ph-stars/'
folder='D:/Documents/Planet_hunters/results/01_Feb_2013_11_14/'
zero_dict=shelve.open('zeros.dat')
transit_dict=shelve.open('transit_details.dat')
simulation_dict=shelve.open('simulation_transit_details.dat')

linking_length=0.2
max_allowed_depth=0.001
picture_number=0

source_details={}
f=open('source_details.csv','r')
for line in f:
    source_id,label,kepler,zooniverse=line.split(',')
    source_id=source_id.strip()
    label=label.strip().strip('"')
    kepler=kepler.strip().strip('"')
    zooniverse=zooniverse.strip().strip('"')
    source_details[source_id]=(label,kepler,zooniverse)
f.close()

def open_url(url):
    N=3
    n=0
    while n<N:
        try:
            return urllib2.urlopen(url)
        except:
            n+=1
    return False

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
    except (KeyError,TypeError):
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
        return (x and y)
    
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
        indices=self.refine_indices(indices,len(data))
        x=[]
        y=[]
        for i in indices:
            x.append(float(data[i]['x']))
            y.append(float(data[i]['y']))
        m,c=np.polyfit(x,y,1)#fit a line to the ambient data
        x=np.array(x)
        y=np.array(y)
        expected=m*x+c
        residuals=(expected-y)/expected
        mx=-np.inf
        for i in range(len(residuals)-1):
            current=min([residuals[i],residuals[i+1]])
            if current>mx:
                mx=current
        return mx
    
    def pickleable(self):
        return {'id':self.id,
                'x1':self.x1,
                'x2':self.x2,
                'y1':self.y1,
                'y2':self.y2,
                'user':self.user}
        
    def zoom_plot(self,data):
        points,_=self.get_data_points(data)
        if len(points)==0:
            return False
        y=[float(point['y']) for point in points]
        top=2*max(y)-min(y)
        bottom=2*min(y)-max(y)
        y=[]
        x=[]
        dy=[]
        for point in data:
            try:
                x.append(float(point['x']))
                y.append(float(point['y']))
                dy.append(float(point['dy']))
            except:
                print point
                print 'error with point'
                continue
        p.figure()
        p.errorbar(x,y,yerr=dy,fmt='ro')
        p.xlim(self.x1,self.x2)
        p.ylim(bottom,top)
        return True
       
            

        
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
        self.zero=self.get_zero()
        self.boxes=[]
        self.users=[]
        self.box_tree=defaultdict(list)
        self.box_resolution=2*linking_length
        self.data=False
        self.scores=False
        self.transits=False
        self.kind=False
        
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
            offset=float(self.data[0]['x'])
            for i in range(len(self.data)):
                self.data[i]['x']=float(self.data[i]['x'])-offset#takes account of some files which are written in BJD
        return self.data
    
    def get_zero(self):
        try:
            return zero_dict[self.id]
        except KeyError:
            out=zero(self.release_id,self.url)
            if out:
                zero_dict[self.id]=out
                return out
            else:
                print 'error calculating zero for light curve '+self.id
                return 0
            
    def get_label(self,tran):
        if not self.kind:
            self.kind='candidate'#may do something with kind in the future
            try:
                transit_dict[self.id]
                self.kind='planet'
            except:
                pass
            try:
                simulation_dict[self.id]
                self.kind='simulation'
            except:
                pass
        if self.kind=='candidate':
            out='0'
        elif self.kind=='planet':
            out=False
            for planet in transit_dict[self.id]:
                epoch,period=planet
                t=self.zero+tran.centre
                t=epoch + np.round((t-epoch)/period)*period -self.zero
                out=out or tran.contains(t,1)
            out=str(int(out))
        elif self.kind=='simulation':
            out=False
            for centre in simulation_dict[self.id]:
                out=out or tran.contains(centre,1)
            out=str(int(out))
        else:
            print 'error finding label'
            out='0'
        return out
    
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
        p.ylim(0.99,1.01)
        
    def zoom_plot(self,source_id):
        global picture_number
        label,_,_=source_details[source_id]
        out=False
        for tran in self.boxes:
            if not tran.zoom_plot(self.get_data()):
                continue
            out=True
            p.title('source id: %s\nlight curve id: %s    release id: %s' %(source_id,self.id,self.release_id))
            pname='%04d.png' %picture_number
            picture_number+=1
            p.savefig(picture_directory+'/'+pname)
            if label=='candidate':
                candidate_htmlfile.write('<img src="transits/%s" width="400" height="300">\n' %pname)
            else:
                eb_htmlfile.write('<img src="transits/%s" width="400" height="300">\n' %pname)
            p.close()
        self.data=False
        return out
        
            
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
        labels=[self.label for i in range(len(objects))]
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
            
    def zoom_plot(self):
        out=False
        for lc in self.light_curves.values():
            out=out or lc.zoom_plot(self.id)
        return out
            
def read_pickled_sources(fname):
    out={}
    sources=pload(fname)
    for i in sources.keys():
        source_dict=sources[i]
        current_source=source(source_dict['id'],source_dict['label'])
        for lc in source_dict['light_curves']:
            current_lc=current_source.get_light_curve(lc['id'], lc['url'], lc['release_id'])
            for bx in lc['boxes']:
                width=float(bx['x2'])-float(bx['x1'])
                height=float(bx['y2'])-float(bx['y1'])
                current_lc.add_box(bx['x1'],bx['y1'],width,height,bx['user'])
        out[i]=current_source
    return out

def test_depths(sources):
    r_sun=109.2
    real=[]
    expected=[]
    difference=[]
    out=defaultdict(lambda:defaultdict(list))
    test_sources=[]
    test_depths=[]
    f=open('simulations_test.csv','r')
    for line in f:
        source_id,stellar_rad,planet_rad=line.split(',')
        try:
            current_source=sources[source_id]
            expected_depth=float((float(planet_rad)/(float(stellar_rad)*r_sun))**2)/1000000
            test_sources.append(current_source)
            test_depths.append(expected_depth)
        except:
            continue
    f.close()
    f=open('kepler_test.csv','r')
    for line in f:
        source_id,depth=line.split(',')
        try:
            current_source=sources[source_id]
            expected_depth=float(depth)/1000000
            test_sources.append(current_source)
            test_depths.append(expected_depth)
        except:
            continue
    f.close()
    for i in range(len(test_sources)):
        progress_bar(i,len(test_sources))
        current_source=test_sources[i]
        expected_depth=test_depths[i]
        for lc in current_source.light_curves.values():
            for tran in lc.boxes:
                if lc.get_label(tran)=='0':
                    continue
                real_depth=tran.get_depth(lc.get_data())
                diff=abs(expected_depth-real_depth)/expected_depth
                expected.append(expected_depth)
                real.append(real_depth)
                difference.append(diff)
                if diff>0.5:
                    out[source_id][lc.id].append(tran)
            lc.data=False
    print '\n'
    return (expected,real,difference,out)

def new_kepler_transits(sources):
    total=0
    new=0
    out=defaultdict(lambda:defaultdict(list))
    f=open('kepler_transits.csv','r')
    for line in f:
        source_id,epoch,period=line.split(',')
        epoch=float(epoch)
        period=float(period)
        try:
            test_source=sources[source_id]
        except:
            continue
        for lc in test_source.light_curves.values():
            temp=[]
            for tran in lc.boxes:
                total+=1
                t=lc.zero+tran.centre
                t=epoch + np.round((t-epoch)/period)*period -lc.zero
                if not tran.contains(t,1):
                    out[source_id][lc.id].append(tran)
                    new+=1
                temp.append(t)
            for t in temp:
                lc.add_box(t,0,0.1,2,'expected')
    f.close()
    print '\ntotal new transits found: %d' %new
    print 'total number of transits: %d' %total
    return out

def make_zoom_plots(sources):
    global picture_directory,picture_number,candidate_htmlfile,eb_htmlfile
    picture_number=0
    picture_directory='D:/Documents/Planet_hunters/candidate_transits/'+date_string()
    orig_dir=picture_directory
    end=1
    while True:
        try:
            os.mkdir(picture_directory)
            break
        except WindowsError:
            picture_directory=orig_dir+'_%02d' %end
            end+=1
    candidate_htmlfile=open(picture_directory+'/candidate_transits.html','w')
    candidate_htmlfile.write('<!DOCTYPE html>\n<html>\n<body>\n')
    eb_htmlfile=open(picture_directory+'/eb_transits.html','w')
    eb_htmlfile.write('<!DOCTYPE html>\n<html>\n<body>\n')       
    count=0
    for s in sources.values():
        progress_bar(count,len(sources))
        count+=1
        _,kepler,zooniverse=source_details[s.id]
        if s.label.strip('"')=='candidate':
            candidate_htmlfile.write('<p>Source: %s    Kepler id: %s    Zooniverse id: %s</p>\n' %(s.id,kepler,zooniverse))
            s.zoom_plot()
        elif s.label.strip('"')=='eb':
            eb_htmlfile.write('<p>Source: %s    Kepler id: %s    Zooniverse id: %s</p>\n' %(s.id,kepler,zooniverse))
            s.zoom_plot()
    candidate_htmlfile.write('</body>\n</html>')
    eb_htmlfile.write('</body>\n</html>')
                
    
scores=get_scores(folder)
sources=read_pickled_sources(in_file)
make_zoom_plots(sources)
print '\ndone'