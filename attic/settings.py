'''
Created on Dec 27, 2012

@author: kieranfinn
'''
import sys
try:
    fname=sys.argv[1]
except:
    fname='full_data.txt'
if fname=='--pylab':
    fname='full_data.txt'
     
unsure_value=-1
positive_responses=['y','Y','yes','Yes','1']
fraction=0.95#fraction of possible planets which are keplar planets
longlists=[]

def close_all():
    for ll in longlists:
        ll.close()
        

def extract_dict(s):
    s=s.strip('}{')
    out={}
    for item in s.split(','):
        key,value=item.split(':')
        key=key.strip()
        value=value.strip()
        if value=='?':
            value=unsure_value
        else:
            try:
                value=int(value)
            except ValueError:
                pass
        out[key]=value
    return out

while True:
    try:
        print 'reading settings from %s' %fname
        f=open(fname,'r')
        break
    except:
        fname=raw_input('Error reading file. Please enter the  name of the settings file: ')
dic={}
for line in f:
    words=line.split(';')
    dic[words[0].strip()]=words[1].strip()
f.close()
data_file=dic['file name']
cmap=dic['column map'].split()
column_map=[cmap.index('object'),cmap.index('person'),cmap.index('score'),cmap.index('label')]
scoredict=extract_dict(dic['scores'])
labeldict=extract_dict(dic['labels'])
labeldict.update({'unsure':unsure_value})
split_frac=float(dic['split'])
try:
    nlabels=int(dic['nlabels'])
except (ValueError, KeyError):
    nlabels=max(labeldict.values())+1#this may need to be changed
try:
    nscores=int(dic['nscores'])
except (ValueError, KeyError):
    nscores=max(scoredict.values())+1
            
unsure_value=-1
positive_responses=['y','Y','yes','Yes','1']
fraction=0.95#fraction of possible planets which are keplar planets