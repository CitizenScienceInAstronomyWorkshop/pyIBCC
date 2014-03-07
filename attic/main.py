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
from  collections import defaultdict

def from_file(folder,name):
    fname=folder+name+'.dat'
    out={name:lambda x:read_file(fname)}
    return out

def read_file(fname):
    f=open(fname,'r')
    out=pickle.load(f)
    f.close()
    return out

comb_methods={'mean':mean.mean,
              'weighted mean': mean.weighted_mean,
              'ibcc':ibcc.main,
              'frequentist': frequentist.main}
comb_methods={}
folder='D:/Documents/Planet_hunters/results/priors/'

folder='test/'
#comb_methods.update(from_file(folder,'ibcc'))
#comb_methods.update(from_file(folder,'mean'))
#comb_methods.update(from_file(folder,'weighted mean'))
comb_methods['Schwamb']=lambda x:read_file(folder+'frequentist.dat')
comb_methods['IBCC']=lambda x:read_file(folder+'ibcc.dat')
comb_methods['Mean']=lambda x:read_file(folder+'mean.dat')
comb_methods['Weighted Mean']=lambda x:read_file(folder+'weighted mean.dat')

'''
p.rcParams.update({'font.size': 18}) # makes the default text size larger
comb_methods.update(from_file(folder,'ibcc_1,1,1,1'))
comb_methods.update(from_file(folder,'ibcc_10,9,9,10'))
comb_methods.update(from_file(folder,'ibcc_1000,995,995,1000'))
comb_methods.update(from_file(folder,'ibcc_300,290,290,300'))
'''

'''folder='D:/Documents/Planet_hunters/results/27_Feb_2013_16_57/'
comb_methods={}
from glob import glob
for fname in glob(folder+'*.dat'):
    fname=fname.split('\\')[-1].split('.')[0]
    comb_methods.update(from_file(folder,fname))'''

data_folder="D:/Documents/Planet_hunters/results/"
data_folder='test_data/'
print 'Will be using the following methods:'
for method in comb_methods.keys():
    print method
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
    compare_labels=defaultdict(lambda:'unsure')
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
        o=entries[settings.column_map[0]].strip()
        if len(entries)<4:
            l='unsure'            
        else:
            l=entries[settings.column_map[3]].strip().strip('"')
        labels.append(settings.labeldict[l])
        if compare_labels[o]!=l and compare_labels[o]!='unsure':
            print "labels for source %s don't agree"
            compare_labels[o]='unsure'
        else:
            compare_labels[o]=l
        objects.append(o)
        people.append(entries[settings.column_map[1]].strip())
    print '\n'
    del r
    return [[objects,people,scores,labels],compare_labels]

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
        if settings.labeldict[labels[i]]==1:
            known_fraction[index]+=1.0
            r1.append(results[i])
        else:
            unknown_fraction[index]+=1.0
            r2.append(results[i])
    print '\n'
    N_kf=sum(known_fraction)
    if N_kf==0:
        print 'not enough known labels to produce curve.'
    else:
        for i in range(len(known_fraction)):
            known_fraction[i]/=N_kf
    for i in range(len(known_fraction)-1)[::-1]:#this should cumulate the results without many sums
        known_fraction[i]+=known_fraction[i+1]
        unknown_fraction[i]+=unknown_fraction[i+1]
    print '\n'
    return (threashold,known_fraction,unknown_fraction,r1,r2)

def calc_auc(x,y):
    out=0
    for i in range(len(x)-1):
        out+=0.5*(y[i]+y[i+1])*(x[i]-x[i+1])
    out=out/x[0]#normalises it
    return out

def multiplot(x,y1,y2):
    p.figure()
    host = host_subplot(111, axes_class=AA.Axes)
    par = host.twinx()
    host.set_xlim(0, 1)
    
    host.set_xlabel("Threshold")
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
results={}
data,compare_labels=read_data(settings.data_file)
print 'Total of %s entries.' %(add_comma(len(data[0])))

for method in comb_methods.keys():
    print 'classifying objects using the %s method' %method
    results[method]=comb_methods[method](data)
    fname=dir_name+'/'+method+'.dat'
    print 'writing data to file %s' %fname
    f=open(fname,'w')
    pickle.dump(results[method],f)
    f.close()

compare={}  
for method in results.keys():
    print 'plotting the %s method' %method
    x,y1,y2,r1,r2=get_plot(results[method],compare_labels)
    f,ax=p.subplots(2,sharex=True)
    p.title(method)
    ax[0].hist(r1,bins=100)
    ax[1].hist(r2,bins=100)    
    p.savefig(dir_name+'/'+method+'_hist.png')
    multiplot(x,y1,y2)
    p.title(method)
    p.savefig(dir_name+'/'+method+'_curve.png')
    compare[method]=[y2,y1]
p.figure()
for method in compare.keys():
    x,y=compare[method]
    auc=calc_auc(x,y)
    p.plot(x,y,label=method+': AUC=%.3f' %auc)
    #p.plot(x,y,label=method)
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
