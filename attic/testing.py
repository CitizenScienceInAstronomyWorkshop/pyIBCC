import pickle
import pylab as p
import settings
from functions import *

f=open('16_Jan_2013_13_46/mean.dat','r')
mean=pickle.load(f)
f.close()

test_objects=[]
f=open('outputlist.txt','r')
for line in f:
    words=line.split()
    test_objects.append(words[0])
f.close()

fname='q1_data.csv'
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


results={i:[0.0,0.0] for i in set(objects)}
for i in range(len(objects)):
    if i%10000==0:
        overprint('Updating results for object %s of %s' %(add_comma(i),add_comma(len(objects))))
    o=objects[i]
    s=scores[i]#shorthand
    results[o][s]+=1
print '\n'

scores=[mean[i] for i in test_objects]
positives=[results[i][1] for i in test_objects]
negatives=[results[i][0] for i in test_objects]

p.hist(scores,bins=100)
p.title('mean')
p.figure()
p.title('positive')
p.hist(positives,range(101))
p.figure()
p.title('negative')
p.hist(negatives,range(101))
p.show()