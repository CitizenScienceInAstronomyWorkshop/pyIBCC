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
import repeated_ibcc
import pylab as p
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from functions import *
from copy import deepcopy

def dict2str(alpha):
    return '%d,%d,%d,%d' %(alpha[(0,0)],alpha[(0,1)],alpha[(1,0)],alpha[(1,1)])

comb_methods={}
alphas=[{(0,0):1,(0,1):1,(1,0):1,(1,1):1},
        {(0,0):10,(0,1):9,(1,0):9,(1,1):10},
        {(0,0):300,(0,1):290,(1,0):290,(1,1):300},
        {(0,0):1000,(0,1):995,(1,0):995,(1,1):1000}]
alphas_2={}
for i in range(len(alphas)):
    method='ibcc_'+dict2str(alphas[i])
    comb_methods[method]=repeated_ibcc.main
    alphas_2[method]=alphas[i]

data_folder="D:/Documents/Planet_hunters/results/"

def read_data(fname):
    print 'reading data from %s' %fname
    f=open(fname,'r')
    r=f.readlines()
    f.close()
    a=range(len(r))#shuffle the data
    shuffle(a)
    objects=longlist([])
    people=longlist([])
    scores=longlist([])
    labels=longlist([])
    count=0
    for i in a:
        if count%10000==0:
            overprint('Processing line %s of %s' %(add_comma(count),add_comma(len(r))))
        count+=1
        line=r[i]
        entries=line.split(',')
        s=entries[settings.column_map[2]].strip() #scores and labels are binary so convert to 1,0 actual output scores stored in constants.py
        try:
            scores.append(settings.scoredict[s])
        except KeyError:
            print '\ nError reading data file. %s is not a legal score.' %s
            continue
        if len(entries)<4:
            labels.append(settings.unsure_value)
        else:
            l=entries[settings.column_map[3]].strip().strip('"')
            labels.append(settings.labeldict[l])
        objects.append(entries[settings.column_map[0]].strip())
        people.append(entries[settings.column_map[1]].strip())
    print '\n'
    del r
    return [objects,people,scores,labels]

def get_plot(results,labels):
    ''' may be a quicker way. also may need to amend to calculate auc'''
    r1=[]
    r2=[]
    resolution=0.0001
    threashold=np.arange(0,1.0+resolution,resolution)
    known_fraction=np.zeros_like(threashold)
    unknown_fraction=np.zeros_like(threashold)
    check_objects=set(results.keys())
    count=0
    for i in check_objects:
        if count%1000==0:
            overprint('Processing object %s of %s' %(add_comma(count),add_comma(len(check_objects))))
        count+=1
        index=int(results[i]/resolution)
        if labels[i]==1:
            known_fraction[index]+=1.0
            r1.append(results[i])
        else:
            unknown_fraction[index]+=1.0
            r2.append(results[i])
    print '\n'
    N_kf=sum(known_fraction)
    if N_kf==0:
        print 'not enough known labels to produce curve.'
        return [0,0,0]
    for i in range(len(known_fraction)):
        known_fraction[i]/=N_kf
    for i in range(len(known_fraction)-1)[::-1]:#this should cumulate the results withouut many sums
        known_fraction[i]+=known_fraction[i+1]
        unknown_fraction[i]+=unknown_fraction[i+1]
    print '\n'
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

dir_name=data_folder+date_string()#have this as a new folder to avoid overwriting old files
settings.dir_name=dir_name
os.mkdir(dir_name)
results={}
data=read_data(settings.data_file)
print 'Total of %s entries.' %(add_comma(len(data[0])))

for method in comb_methods.keys():
    print 'classifying objects using the %s method' %method
    repeated_ibcc.alphadict=alphas_2[method]
    results[method]=comb_methods[method](data)
    fname=dir_name+'/'+method+'.dat'
    print 'writing data to file %s' %fname
    f=open(fname,'w')
    pickle.dump(results[method],f)
    f.close()
    
objects=data[0]
labels=data[3]
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

compare={} 
for method in results.keys():
    print 'plotting the %s method' %method
    x,y1,y2,r1,r2=get_plot(results[method],compare_labels)
    f,ax=p.subplots(2,sharex=True)
    p.title(method)
    ax[0].hist(r1,bins=100)
    ax[1].hist(r2,bins=100)    
    p.savefig(dir_name+'/'+method+'_hist.png')
    '''''p.plot(x,y1,label='known planets')
    p.plot(x,y2,label='candidate light curves')
    p.xlim(0,1)
    #p.ylim(0,1)
    p.legend()
    p.xlabel('Threashold')
    p.ylabel('Fraction classified as planets')'''
    multiplot(x,y1,y2)
    p.title(method)
    p.savefig(dir_name+'/'+method+'_curve.png')
    compare[method]=[y2,y1]
p.figure()
for method in compare.keys():
    x,y=compare[method]
    p.plot(x,y,label=method)
p.title('Comparison')
p.ylabel('Fraction of known planets recovered')
p.xlabel('Number of new candidates')
p.ylim(0,1)
p.legend(loc='lower right')
p.savefig(dir_name+'/comparison.png')
print 'Total running time: %s' %hms(time.time()-beginning)
settings.close_all()
p.show()    

print 'done'