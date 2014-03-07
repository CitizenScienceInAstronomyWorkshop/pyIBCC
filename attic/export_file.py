'''
Created on 29 Jan 2013

@author: Kieran Finn
'''
import pickle

folder='full_data_28_01_13/'
fname=folder+'ibcc.dat'
out='interesting_candidates.csv'
threashold=0.8


f=open(fname,'r')
results=pickle.load(f)
f.close()

f=open(out,'w')
for candidate in results.keys():
    if results[candidate]>threashold:
        f.write(candidate+'\n')
f.close()
print 'done'

import Tkinter as tk
import webbrowser
f=open('sources_id.csv','r')
r=f.readlines()
f.close()
zoo_dict={}
kepler_dict={}
label_dict={}
for line in r:
    source_id,zoo_id,kepler_id,label=line.split(',')
    kepler_id=kepler_id.strip().strip('"_s')
    zoo_id=zoo_id.strip().strip('"')
    label=label.strip().strip('"')
    zoo_dict[zoo_id]=source_id
    kepler_dict[kepler_id]=zoo_id
    label_dict[zoo_id]=label
    
def look(kepler_id=False,zoo_id=False):
    if not kepler_id and not zoo_id:
        x=tk.Tk()
        kepler_id=x.clipboard_get().strip()
    if not zoo_id:
        zoo_id=kepler_dict['kplr%09d' %int(kepler_id)]
    webbrowser.open("http://www.planethunters.org/sources/%s" %zoo_id)
    print 'source %s. labelled as %s' %(zoo_id,label_dict[zoo_id])
    print 'mean score: %.3f' %mean[zoo_dict[zoo_id]]
    print 'weighted mean score: %.3f' %weighted_mean[zoo_dict[zoo_id]]
    try:
        print 'ibcc score: %.3f' %ibcc[zoo_dict[zoo_id]]
    except KeyError:
        print 'No score for ibcc method'
    print 'frequentist score: %.3f' %frequentist[zoo_dict[zoo_id]]
    try:
        x.destroy()
    except:
        pass
    

f=open('ibcc_specific.csv','w')
count=0
to_check=[]
for i in ibcc.keys():
    a=ibcc[i]
    b=[0]
    try:
        b.append(mean[i])
    except KeyError:
        pass
    try:
        b.append(weighted_mean[i])
    except KeyError:
        pass
    try:
        b.append(frequentist[i])
    except KeyError:
        pass
    b=max(b)
    if a>0.8 and b<0.3:
        count+=1
        f.write(i+'\n')
        to_check.append(i)
f.close()
    
