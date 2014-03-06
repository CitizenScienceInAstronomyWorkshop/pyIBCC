'''
Created on Dec 27, 2012

@author: kieranfinn
'''
import time
beginning=time.time()
import numpy as np
import settings
import os
from random import shuffle
import pickle
import mean
import ibcc
import frequentist
import pylab as p
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from functions import *

comb_methods={'mean':mean.mean,
              'weighted mean': mean.weighted_mean,
              'ibcc':ibcc.main,
              'frequentist': frequentist.main}

def read_data(fname):
    print 'reading data from %s' %fname
    f=open(fname,'r')
    objects=[]
    people=[]
    scores=[]
    labels=[]
    for line in f:
        entries=line.split(',')
        s=entries[settings.column_map[2]].strip() #scores and labels are binary so convert to 1,0 actual output scores stored in constants.py
        try:
            scores.append(settings.scoredict[s])
        except KeyError:
            print 'Error reading data file. %s is not a legal score.' %s
            continue
        if len(entries)<4:
            labels.append(settings.unsure_value)
        else:
            l=entries[settings.column_map[3]].strip().strip('"')
            labels.append(l)
        objects.append(entries[settings.column_map[0]].strip())
        people.append(entries[settings.column_map[1]].strip())
    f.close()
    
    #shuffle the entries so that the training data is a good proportion.
    #May be able to shuffle the order in which input file lines are read bu not sure how to do this without reading the whole file to the memory
    a=range(len(objects))
    shuffle(a)
    tobjects=[]
    tpeople=[]
    tscores=[]
    tlabels=[]
    for i in a:
        tobjects.append(objects[i])
        tpeople.append(people[i])
        tscores.append(scores[i])
        tlabels.append(labels[i])
    
    testing,training=split_data(tobjects,tpeople,tscores,tlabels,settings.split_frac)
    
    return (testing,training)

def split_data(objects,people,scores,labels,split_frac):
    '''method for splitting the data into two sets: training and testing data.'''
    dat=[objects,people,scores,labels]
    '''unsure=[[],[],[],[]]
    sure=[[],[],[],[]]
    for i in range(len(objects)):
        if labels[i]==c.unsure_value:
            for j in range(len(dat)):
                unsure[j].append(dat[j][i])
        else:
            for j in range(len(dat)):
                sure[j].append(dat[j][i])
    
    sure_split=int(round(split_frac*len(sure[0])))
    unsure_split=int(round(split_frac*len(unsure[0])))
    training=[sure[i][:sure_split]+unsure[i][:unsure_split] for i in range(len(dat))]
    testing=[sure[i][sure_split:]+unsure[i][unsure_split:] for i in range(len(dat))]'''
    split_frac=int(round(split_frac*len(objects)))
    training=[dat[i][:split_frac] for i in range(len(dat))]
    testing=[dat[i][split_frac:] for i in range(len(dat))]
    return (testing,training)

def get_plot(results,labels):
    ''' may be a quicker way. also may need to amend to calculate auc'''
    known_fraction=[]
    unknown_fraction=[]
    r1={}
    r2={}
    threashold=[]
    resolution=0.0001
    N_kf=0.0
    N_uf=0.0
    for o in labels.keys():
        if settings.labeldict[labels[o]]==1:
            N_kf+=1
        else:
            N_uf+=1
    if N_kf*N_uf==0:
        print 'not enough known labels to produce curve.'
        return [0,0,0]
    for i in np.arange(0,1.0+resolution,resolution):
        kf=0.0
        uf=0.0
        for o in results.keys():
            if results[o]>=i:
                if settings.labeldict[labels[o]]==1:
                    kf+=1
                    r1[o]=results[o]
                else:
                    uf+=1
                    r2[o]=results[o]
        known_fraction.append(kf/N_kf)
        unknown_fraction.append(uf)
        threashold.append(i)
    return (threashold,known_fraction,unknown_fraction,r1.values(),r2.values())

def get_plot(results,labels):
    ''' may be a quicker way. also may need to amend to calculate auc'''
    known_fraction=[]
    unknown_fraction=[]
    r1=[]
    r2=[]
    threashold=[]
    resolution=0.0001
    N_kf=0.0
    N_uf=0.0
    for o in labels.keys():
        if settings.labeldict[labels[o]]==1:
            N_kf+=1
        else:
            N_uf+=1
    if N_kf*N_uf==0:
        print 'not enough known labels to produce curve.'
        return [0,0,0]
    
    to_test=set(results.keys())
    tot_kf=0.0
    tot_uf=0.0
    for i in np.arange(1,0-resolution,0.0-resolution):
        kf=0.0
        uf=0.0
        temp=deepcopy(to_test)
        for o in temp:
            if results[o]>=i:
                to_test.remove(o)
                if settings.labeldict[labels[o]]==1:
                    kf+=1
                    r1.append(results[o])
                else:
                    uf+=1
                    r2.append(results[o])
        tot_kf+=kf
        tot_uf+=uf
        known_fraction.append(tot_kf/N_kf)
        unknown_fraction.append(tot_uf)
        threashold.append(i)
    return (threashold,known_fraction,unknown_fraction,r1,r2)

def multiplot(x,y1,y2):
    p.figure()
    host = host_subplot(111, axes_class=AA.Axes)
    par = host.twinx()
    host.set_xlim(0, 1)
    
    host.set_xlabel("Threashold")
    host.set_ylabel("Candidates")
    par.set_ylabel("Fraction")
    par.set_ylim(0,1)
    
    
    p1, = host.plot(x,y2, label="Number of new candidates")
    p2, = par.plot(x,y1, label="Fraction of known planets")
    
    host.legend()
    
    host.axis["left"].label.set_color(p1.get_color())
    par.axis["right"].label.set_color(p2.get_color())

'''main program'''

dir_name=date_string()#have this as a new folder to avoid overwriting old files
os.mkdir(dir_name)
results={}
testing,training=read_data(settings.data_file)
a=len(testing[0])
b=len(training[0])
print 'Total of %s entries. Using %s for training purposes and %s for testing.' %(add_comma(a+b),add_comma(b),add_comma(a))

for method in comb_methods.keys():
    print 'classifying objects using the %s method' %method
    results[method]=comb_methods[method](training,testing)
    fname=dir_name+'/'+method+'.dat'
    print 'writing data to file %s' %fname
    f=open(fname,'w')
    pickle.dump(results[method],f)
    f.close()
    
objects=testing[0]+training[0]
labels=testing[3]+training[3]
compare_labels={i:settings.unsure_value for i in set(objects)}
print 'Collecting known information'
for i in range(len(objects)):
    if i%10000==0:
        overprint('Processing line %s of %s' %(add_comma(i),add_comma(len(objects))))
    if compare_labels[objects[i]] != labels[i] and compare_labels[objects[i]]!=settings.unsure_value:
        print "provided labels don't agree"
        compare_labels[objects[i]]=settings.unsure_value
    else:
        compare_labels[objects[i]]=labels[i]
print '\n'
   
for method in results.keys():
    print 'plotting the %s method' %method
    x,y1,y2,r1,r2=get_plot(results[method],compare_labels)
    f,ax=p.subplots(2,sharex=True)
    p.title(method)
    ax[0].hist(r1,bins=100)
    ax[1].hist(r2,bins=100)    
    '''''p.plot(x,y1,label='known planets')
    p.plot(x,y2,label='candidate light curves')
    p.xlim(0,1)
    #p.ylim(0,1)
    p.legend()
    p.xlabel('Threashold')
    p.ylabel('Fraction classified as planets')'''
    multiplot(x,y1,y2)
    p.title(method)
print 'Total running time: %s' %hms(time.time()-beginning)
p.show()    

print 'done'