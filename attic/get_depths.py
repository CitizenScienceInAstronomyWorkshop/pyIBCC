'''
Created on 20 Apr 2013

@author: Kieran Finn
'''
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
from functions import *
from random import random
import mean
import ibcc
from collections import defaultdict
import os
import itertools
import settings
import shelve

folder='D:/Documents/Planet_hunters/results/05_Mar_2013_18_27/'
#folder='full_data_28_01_13'
fname='investigate_full.csv'
data_folder='D:/Documents/Planet_hunters/ph-stars/'
#data_folder='/Users/kieranfinn/Documents/ph-stars/'
out_folder="D:/Documents/Planet_hunters/results/"
out=fname.split('.')[0]+'_transits.dat'
zero_dict=shelve.open('zeros.dat')
transit_dict=shelve.open('transit_details.dat')
simulation_dict=shelve.open('simulation_transit_details.dat')

repetition_file=open('repetition.csv','w')

linking_length=0.2
max_allowed_depth=0.05

all_transits={}#a dictionary which contains the details of  every transit
source_labels={}
light_curve_details={}

keys=['planet','eb','simulation','candidate']
def barchart(dictionary,colour='b'):
    '''creates a barchart from a dictionary dictionary={label:frac}'''
    labels=[]
    fracs=[]
    indices=[]
    i=0
    width=0.35
    for key in keys:
        indices.append(i)
        labels.append(key)
        fracs.append(dictionary[key])
        i+=1
    indices=np.array(indices)
    p.bar(indices,fracs,width,color=colour)
    p.xticks(indices+width/2., labels )
    return True

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
    except:
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

        
class transit(box):
    def __init__(self,group):
        centre=sum([box.centre for box in group])/len(group)
        width=sum([box.x2-box.x1 for box in group])/len(group)
        self.x1=centre-0.5*width
        self.x2=centre+0.5*width
        self.centre=centre
        self.y1=0
        self.y2=2 #these are probably unessesery but include them so I can use all the box methods
        self.users=set([box.user for box in group])
        self.id='%.3f' %centre#need a unique id which can identify the transit so use x coordinate
        
    def contains(self,x):
        return self.x1<=x<=self.x2
    
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
        self.zero=self.get_zero()
        self.data=False
        self.scores=False
        self.transits=False
        self.kind=False
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
                out=out or tran.contains(t)
            out=str(int(out))
        elif self.kind=='simulation':
            out=False
            for centre in simulation_dict[self.id]:
                out=out or tran.contains(centre)
            out=str(int(out))
        else:
            print 'error finding label'
            out='0'
        return out
    
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
            label=self.get_label(tran)
            labels+=[label for i in range(len(users))]
        return (objects,people,scores,labels)
    
    def cleanup(self):
        out=False
        primaries=[]
        temp=[]
        for tran in self.boxes:
            depth=tran.get_depth(self.get_data())
            if depth<0 or depth>max_allowed_depth:#a binary transit
                primaries.append(tran.centre+self.get_zero())
            else:
                temp.append(tran)
                out=True
        self.boxes=temp#shouldn't remove objects for a list being iterated over
        self.data=False#this should free up some memory. Hopefully.
        return (out,primaries)
                
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
        self.label=label.strip('"')
        
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
        objects=[self.id+'_'+obj for obj in objects]
        return (objects,people,scores,labels)
    
    def get_centres(self):
        centres=[]
        lookup={}
        for lc in self.light_curves.values():
            for tran in lc.boxes:
                t=tran.centre+lc.get_zero()
                centres.append(t)
                lookup[t]=(lc,tran)
        centres=np.sort(centres)
        return (centres,lookup)
    
    def get_period(self,centres):
        if len(centres)<2:
            return False
        periods=[centres[i+1]-centres[i] for i in range(len(centres)-1)]
        scores={}
        for i in range(len(periods)):
            score=0
            for centre in centres:
                to_add=(abs(centres[i]-centre)%periods[i])/periods[i]
                if to_add>0.5: #takes account of measured transit being closer to the next actual transit
                    to_add=1.0-to_add
                score+=to_add
            scores[periods[i]]=score
        period=min(scores, key=scores.get)
        return period
            
    
    def remove_secondaries(self,primaries):
        centres,lookup=self.get_centres()
        if len(centres)<2:
            return True
        period=self.get_period(primaries)
        mx=-np.inf
        for start in centres:
            temp=[]
            score=0
            for end in centres:
                lc,tran=lookup[end]
                t=np.round((end-start)/period)*period+start
                if len(tran.contains(t-lc.get_zero(),1)):
                    score+=1
                    temp.append(end)
            if score>mx:
                mx=score
                to_remove=temp
        if abs(mx-len(primaries))<=1 or float(abs(mx-len(primaries)))/len(primaries)<0.1:
            if len(centres)==mx:#all marked transits are secondaries
                return False
            for centre in to_remove:
                lc,tran=lookup[centre]
                lc.boxes.remove(tran)
            return True
        return True 
            
    def cleanup(self):
        out=False
        primaries=[]
        for lc in self.light_curves.values():
            temp,prim=lc.cleanup()
            primaries+=prim
            out=out or temp#source is real candidate if ANY light curves pass the test
        if len(primaries):
            out=out and self.remove_secondaries(primaries)
        return out
    
    def look_for_repeated(self):
        allowance=0.2
        centres,lookup=self.get_centres()
        if len(centres)<3:
            return False
        period=self.get_period(centres)
        start_scores=[]
        for start in centres:
            out=0
            for i in centres:
                to_add=(abs(start-i)%period)/period
                if to_add>0.5: #takes account of measured transit being closer to the next actual transit
                    to_add=1.0-to_add
                out+=to_add
            start_scores.append(out)
        if min(start_scores)/(period*(len(centres)-1))>allowance:
            return False
        i=start_scores.index(min(start_scores))
        start=centres[i]
        for i in centres:
            temp=(abs(start-i)%period)/period
            if temp>0.5:
                temp=1.0-temp
            repetition_file.write('%s,%f\n' %(self.label,temp))
            if temp>allowance:
                lc,tran=lookup[i]
                lc.boxes.remove(tran)
        return True
            
    
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
        
    def get_depths(self):
        out=[]
        for lc in self.light_curves.values():
            for tran in lc.boxes:
                out.append(tran.get_depth(lc.get_data()))
        return float(sum(out))/len(out) #may want to change that to median
            
        
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

def make_list(fname,sources):
    out=defaultdict(int)
    f=open(fname,'w')
    for i in sources.keys():
        out[sources[i].label]+=1
        no_transits=sum([len(lc.boxes) for lc in sources[i].light_curves.values()])
        f.write('%s,%d,%s\n' %(i,no_transits,source_labels[i]))
    f.close()
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

try:
    del sources #free up some memory
except:
    pass

print '\n'
print 'total transits found: %s' %add_comma(len(out_objects))
f=open(dir_name+'/'+out,'w')
for i in range(len(out_objects)):
    f.write('%s,%s,%d,%s\n' %(out_objects[i],out_people[i],out_scores[i],out_labels[i]))
f.close()

fname='depth_list.csv'
f=open(fname,'r')
needed=[]
for line in f:
    sid=line.split(',')[0]
    sid=sid.strip().strip('"')
    needed.append(sid)
f.close()
needed=set(needed)



sources={}
for tran in out_objects:
    source_id,light_curve_id,tran_id=tran.split('_')
    if source_id in needed:
        x,width=all_transits[light_curve_id+'_'+tran_id]
        url,release_id=light_curve_details[light_curve_id]
        if source_id not in set(sources.keys()):
            sources[source_id]=source(source_id,source_labels[source_id])
        lc=sources[source_id].get_light_curve(light_curve_id,url,release_id)
        lc.add_box(x,0,width,2,tran_id)#again, using 0-2 for y axis to include everything. May want to make this more sophisticated at some point

fname='depth_list.csv'
f=open(fname,'r')
r=f.readlines()
f.close()
f=open(fname,'w')
f.write('id,kepler,transits,depth,mean,weighted,frequentist,ibcc,promising,notes\n')
for line in r[1:]:
    sid,kepler,transits,mean,weighted,frequentist,ibcc,promising,notes=line.split(',',8)
    depth=sources[sid.strip().strip('"')].get_depths()
    f.write('%s,%s,%s,%.3e,%s,%s,%s,%s,%s,%s'%(sid,kepler,transits,depth,mean,weighted,frequentist,ibcc,promising,notes))
f.close()

repetition_file.close()
print 'Total running time: %s' %hms(time.time()-beginning)
settings.close_all()