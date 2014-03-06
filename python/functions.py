'''
Created on 14 Jan 2013

@author: Kieran Finn
'''
import sys
import time
from tempfile import mkdtemp
import os.path as path
import pickle
import shelve
import shutil
from copy import copy

MAX_LENGTH=2020000
class longlist(): #this class allow lists which are too big to be held in the memory
    def __init__(self,initial_value,max_length=MAX_LENGTH):
        import settings#this is done here to avoid unnecesary user input
        self.max_length=max_length
        settings.longlists.append(self)#this is a bit of a hack but it works
        self.dir=mkdtemp(dir="D:/temp_files/")
        self.length=0
        self.fnumber=-1#get_file will increase this to 0
        self.cnumber=0
        self.a=[]
        self.files=[]
        self.f=self.get_file()
        for i in initial_value:
            self.append(i)
        
    def get_file(self):
        fname=path.join(self.dir,str(self.fnumber))
        self.fnumber+=1
        f=open(fname,'w')
        self.files.append(fname)
        return f
    
    def switch(self,n):
        pickle.dump(self.a, self.f)
        self.f.close()
        self.f=open(self.files[n],'r+')
        self.a=pickle.load(self.f)
        self.f.seek(0)
        self.cnumber=n
    
    def append(self,item):
        self.length+=1
        if self.cnumber!=self.fnumber:
            self.switch(self.fnumber)
        if len(self.a)>=self.max_length:
            pickle.dump(self.a, self.f)
            self.f.close()
            self.f=self.get_file()
            self.cnumber=self.fnumber+1
            self.a=[item]
        else:
            self.a.append(item)
            
    def __getitem__(self,i):
        if isinstance(i,slice):
            out=longlist([])
            for j in xrange(*i.indices(self.length)):
                out.append(self.__getitem__(j))
            return out
        if i<0:
            i=self.length+i
        n=i/self.max_length
        i=i%self.max_length
        if n!=self.cnumber:
            self.switch(n)
        return self.a[i]
    
    def __len__(self):
        return self.length
    
    def add(self,l):#adds in place
        for item in l:
            self.append(item)
        return self
        
    
    def close(self):
        self.f.close()
        shutil.rmtree(self.dir,True)
 
def csvload(fname):
    out={}
    f=open(fname,'r')
    for line in f:
        key,value=line.split(',')
        out[key]=float(value)
    f.close()
    return out
        
class defaultshelf():
    def __init__(self,default):
        import settings
        settings.longlists.append(self)
        self.default=default
        self.dir=mkdtemp()
        fname=path.join(self.dir,'temp.dat')
        self.d=shelve.open(fname)
        
    def __getitem__(self,key):
        if not self.d.has_key(key):
            self.d[key]=self.default()
        return self.d[key]
    
    def __len__(self):
        return len(self.d)
    
    def __setitem__(self,i,j):
        self.d[i]=j
    
    def keys(self):
        return self.d.keys()
    
    def close(self):
        self.d.close()
        shutil.rmtree(self.dir,True)
        
class pickle_defaultdict(): #rewriting ofcollections.defaultdict in such a way that can be pickled
    def __init__(self,default):
        self.default=default
        self.d={}
        
    def __getitem__(self,key):
        if not self.d.has_key(key):
            self.d[key]=copy(self.default)
        return self.d[key]
    
    def __len__(self):
        return len(self.d)
    
    def __setitem__(self,i,j):
        self.d[i]=j
    
    def keys(self):
        return self.d.keys()
            
def set_item(d,i,j,item):
    temp=d[i]
    temp[j]=item
    d[i]=temp

def overprint(s):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(s)

def pload(fname):
    f=open(fname,'rb')
    try:
        out=pickle.load(f)
    except:
        f.close()
        f=open(fname,'r')
        out=pickle.load(f)
    f.close()
    return out

def pdump(obj,fname):
    f=open(fname,'wb')
    pickle.dump(obj,f)
    f.close()
    
def add_comma(number): #makes a large number more readable by adding a comma every three digits
    out=''
    i=1
    number=str(number)
    while i<=len(number):
        out=number[-i]+out
        if i%3==0 and i!=len(number):
            out=','+out
        i+=1
    return out

def date_string():
    s=time.ctime()
    wday,month,day,t,year=s.split()
    t=t.split(':')
    t=t[0]+'_'+t[1]
    return day+'_'+month+'_'+year+'_'+t

def hms(time): #converts a number in seconds into a string of the form HH:MM:SS
    hours=int(time/3600)
    time-=hours*3600
    minutes=int(time/60)
    seconds=int(time-minutes*60)
    out= '%d:%02d:%02d' %(hours,minutes,seconds)
    return out

def progress_bar(current,total):
    screen_length=80.0
    n=int(screen_length*float(current)/total)
    overprint('#'*n)