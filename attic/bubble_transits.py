'''
Created on 13 Feb 2013

@author: Kieran Finn
'''
import pylab as p
from collections import defaultdict
from glob import glob
import numpy as np
import pickle
from functions import *

folder='D:/Documents/Planet_hunters/results/01_Feb_2013_11_14/'
#folder='full_data_28_01_13'
fname='investigate_recent.csv'
out_name='bubble_consensus.dat'

colours={0:'b',1:'r',2:'g',3:'c',4:'m',5:'y',6:'k'}

class line():
    def __init__(self,end1,end2,coord,orientation):
        self.end1=end1
        self.end2=end2
        self.coord=coord
        self.orientation=orientation
        
    def above(self,x,y):
        if self.orientation=='vertical':
            c1=x
            c2=y
        else:
            c1=y
            c2=x
        if not (self.end1<=c2<=self.end2):
            return False
        else:
            return c1<=self.coord
            

class box():
    def __init__(self,x1,y1,width,height):
        self.x1=x1
        self.x2=x1+width
        self.y1=y1
        self.y2=y1+height
        self.centre=(x1+0.5*width,y1+0.5*height)
        
    def overlap(self,bx):
        x=self.x1<=bx.x1<=self.x2 or self.x1<=bx.x2<=self.x2
        y=self.y1<=bx.y1<=self.y2 or self.y1<=bx.y2<=self.y2
        return (x and y)
    
    def width(self):
        return self.x2-self.x1
    
    def split(self):
        x3,_=self.centre
        width=x3-self.x1
        height=self.y2-self.y1
        out=[box(self.x1,self.y1,width,height),
             box(x3,self.y1,width,height)]
        return out
        
    
    
class transit():
    def __init__(self,x1,x2,y1,y2,level):
        self.x1=x1
        self.x2=x2
        self.y1=y1
        self.y2=y2
        self.level=level
        
    def contains(self,x,y):
        x=self.x1<=x<=self.x2
        y=self.y1<=y<=self.y2
        return (x and y)
    
    def plot(self):
        colour=colours[(self.level-1)%7]
        x=[self.x1,self.x1,self.x2,self.x2,self.x1]
        y=[self.y1,self.y2,self.y2,self.y1,self.y1]
        p.fill(x,y,colour)
        
def get_scores(folder):
    out={}
    try:
        files=glob(folder+'*.dat')
    except:
        return out
    for fname in files:
        name=fname.split('\\')[-1].split('.')[0]
        f=open(fname,'r')
        out[name]=pickle.load(f)
        f.close()
    return out

def merge(boxes):
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for box in boxes:
        x1.append(box.x1)
        x2.append(box.x2)
        y1.append(box.y1)
        y2.append(box.y2)
    x1=max(x1)
    x2=min(x2)
    y1=max(y1)
    y2=min(y2)
    return box(x1,y1,x2-x1,y2-y1)
        
class light_curve():
    def __init__(self,light_curve_id,release_id):
        self.id=light_curve_id
        self.release_id=release_id
        self.boxes=[]
        self.lower_xs=[]
        self.lower_ys=[]
        self.x_lines={}
        self.y_lines={}
        self.ordered=False
        self.transits=False
        self.cleaned=False
        self.users=[]
        
    def add_box(self,x,y,width,height,user):
        self.ordered=False
        self.cleaned=False
        self.users.append(user)
        x=float(x)
        y=float(y)
        width=float(width)
        height=float(height)
        x2=x+width
        y2=y+height
        self.x_lines[x2]=line(y,y2,x2,'vertical')
        self.y_lines[y2]=line(x,x2,y2,'horizontal')
        self.lower_xs.append(x)
        self.lower_ys.append(y)
        self.boxes.append(box(x,y,width,height,user))
        '''
        x2=x+width
        y2=y+height
        self.lower_x.append(x)
        self.lower_y.append(y)
        self.upper_x.append(x2)
        self.upper_y.append(y2)
        self.y_ends[y]=x2
        self.x_ends[x]=y2
        self.x_starts[x]=y
        self.y_starts[y]=x
        temp_list=[temp]
        for box in self.transits:
            temp_list+=temp.merge(box)
        self.transits=temp_list'''
    
    def order(self):
        self.ordered=True
        self.xs=np.sort(self.x_lines.keys())
        self.ys=np.sort(self.y_lines.keys())
        
    def score(self,x,y):
        out=[]
        for box in self.boxes:
            out+=box.contains(x,y)
        return len(set(out))
    
    def get_transit(self,x,y):
        if not self.ordered:
            self.order()
        s=self.score(x,y)
        if s==0:
            return (False,False)
        else:
            i=0
            while not self.x_lines[self.xs[i]].above(x,y):
                i+=1
                if i>=len(self.xs):
                    return(False,False)
            x2=self.xs[i]
            i=0
            while not self.y_lines[self.ys[i]].above(x,y):
                i+=1
                if i>=len(self.ys):
                    return(False,False)
            y2=self.ys[i]
            return (s,transit(x,x2,y,y2,s))
    
    def find_transits(self):
        to_check=[box(5.0*i,0.95,5.0,0.1) for i in range(6)]
        for grid_box in to_check:
            no_boxes=0
            i=0
            while i<len(self.boxes) and no_boxes<5:
                if self.boxes[i].overlap(grid_box):
                    no_boxes+=1
                i+=1
            if no_boxes<5:
                continue
            current_boxes=[]
            for box in self.boxes:            
                if (0.5<=box.width()/grid_box.width()<=1):
                    current_boxes.append(box)
            if len(current_boxes)<5:
                to_check+=grid_box.split()
                continue
            el
                
                
                
        
        
    
    def should_use(self,box,to_check):
        for l in to_check:
            for tran in l:
                x=tran.x1
                y=tran.y1
                if box.contains(x,y) and not (x==box.x1 and y==box.y1):
                    return False
        return True
    
    def clean(self,transits):
        self.cleaned=True
        out={}
        scores=np.sort(transits.keys())
        for i in range(len(scores)):
            temp=[]
            for tran in transits[scores[i]]:
                if self.should_use(tran,[transits[scores[j]] for j in range(i,len(scores))]):
                    temp.append(tran)
            out[scores[i]]=temp
        return out
                
    
    def plot(self):
        if not self.transits:
            self.transits=self.find_transits()
        if not self.cleaned:
            self.transits=self.clean(self.transits)
        p.figure()
        for i in np.sort(self.transits.keys()):
            for tran in self.transits[i]:
                tran.plot()
        p.title('light curve id: %s\nrelease id: %s' %(self.id,self.release_id))
        
    def max_score(self):
        if not self.transits:
            self.transits=self.find_transits()
        try:
            return float(max(self.transits.keys()))/len(set(self.users))
        except ValueError:
            return 0
            
'''    def score(self,lower_x,lower_y):
        if not self.ordered:
            self.order()
        x=self.lower_x[lower_x]
        y=self.lower_y[lower_y]
        if not(self.y_starts[y]<=x<=self.y_ends[y] and self.x_starts[x]<=y<=self.x_ends[x]):
            return [0,0,0]
        upper_x=0
        while self.upper_x[upper_x]<x:
            upper_x+=1
        x=lower_x-upper_x+1
        upper_y=0
        while self.upper_y[upper_y]<y:
            upper_y+=1
        y=lower_y-upper_y+1
        return [min([y,x]),self.upper_x[upper_x],self.upper_y[upper_y]]
        
    def score(self,x,y):
        if not self.ordered:
            self.order()
        lower_x=0:
        while self.lower_x[lower_x]<x:
            lower_x+=1
        upper_x=0:
        while self.upper_x[upper_x]<x:
            upper_x+=1
        x=lower_x-upper_x
        lower_y=0:
        while self.lower_y[lower_y]<y:
            lower_y+=1
        upper_y=0:
        while self.upper_y[upper_y]<y:
            upper_y+=1
        y=lower_y-upper_y
        return min([y,x])
'''
            
class source():
    def __init__(self,source_id,label):
        self.id=source_id
        self.light_curves={}
        self.label=label
        
    def get_light_curve(self,light_curve_id,release_id):
        if light_curve_id not in self.light_curves.keys():
            self.light_curves[light_curve_id]=light_curve(light_curve_id,release_id)
        return self.light_curves[light_curve_id]
    
    def score(self):
        out=[lc.max_score() for lc in self.light_curves.values()]
        return max(out)
    
    def plot(self):
        print 'Details for source number %s' %self.id
        print 'Label: %s' %self.label
        for method in scores.keys():
            print '%s: %.4f' %(method, scores[method][self.id])
        for light_curve_id in self.light_curves.keys():
            self.light_curves[light_curve_id].plot()
            
        
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
        source_id,light_curve_id,user,label,_,release_id,x,y,width,height=line.strip().split(',')
        if source_id not in out.keys():
            out[source_id]=source(source_id,label)
        lc=out[source_id].get_light_curve(light_curve_id,release_id)
        lc.add_box(x,y,width,height,user)
    print '\n'
    return out

scores=get_scores(folder)
sources=read_data(fname)
results={}
source_ids=sources.keys()
for i in range(len(sources)):
    source_id=source_ids[i]
    print 'finding transits and calculating scores for source %s. (%s of %s)' %(source_id,add_comma(i),add_comma(len(sources)))
    results[source_id]=sources[source_id].score()
f=open(out_name,'w')
pickle.dump(results,f)
f.close()
print 'done'