'''
Created on Dec 23, 2012

@author: kieranfinn
'''
from random import random
import settings

fname='test.dat'
objects=15000
people=25000
classifications=1000000

obj_id=0
person_id=0
labeldict={0:'candidate',2:'planet',1:'candidate'}

pepdict={i:[] for i in range(people)}

def confusion_matrix():
    return [0.0,1.0]
        
    

class sn():
    def __init__(self):
        global obj_id
        self.id=obj_id
        obj_id+=1
        if random()>0.8:
            if random()>settings.fraction:
                self.value=1
            else:
                self.value=2
        else:
            self.value=0
        
class person():
    def __init__(self):
        global person_id
        self.id=person_id
        person_id+=1
        
        self.cm=confusion_matrix()
        
    def classify(self,sun):
        a=int((sun.value+1)/2)
        if random()<self.cm[a]:
            return 0
        else:
            return 1
        
def classification(p,s):
    pid=p.id
    sid=s.id
    score=p.classify(s)
    label=s.value
    return '%d,%d,%d,%s' %(sid,pid,score,labeldict[label])

f=open(fname,'w')
objs=[sn() for i in range(objects)]
pers=[person() for i in range(people)]

for i in range(classifications):
    if i%1000==0:
        print 'classifying %d of %d' %(i+1,classifications)
    a=int(random()*len(pers))
    while True:
        b=int(random()*len(objs))
        if b not in pepdict[a]:
            pepdict[a].append(b)
            break
    a=pers[a]
    b=objs[b]
    f.write(classification(a,b))
    f.write('\n')

a=int(random()*len(objs))
b=int(random()*len(pers))
a=objs[a]
b=pers[b]
f.write(classification(b,a))  
f.close()

print 'done'