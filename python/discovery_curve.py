import pylab as p
import numpy as np
from collections import defaultdict
p.rcParams.update({'font.size': 18}) # makes the dfault text size larger

fname='exoplanet.eu_catalog.csv'
f=open(fname,'r')
r=f.readlines()
f.close()
fields=r[0][1:].split(',')
for i in range(len(fields)):
    fields[i]=fields[i].strip()
i=fields.index('discovered')
data=defaultdict(lambda:0)
for line in r[1:]:
    try:
        date=int(line.split(',')[i].strip())
        data[date]+=1
    except:
        print line.split(',')[i].strip()

x=list(np.sort(data.keys()))
y=[data[i] for i in x]
for i in range(1,len(y)):
    y[i]+=y[i-1]

x=[x[0]-1]+x+[x[-1]]
y=[0]+y+[0]
p.fill(x,y)
p.xlim(min(x),max(x))
p.xlabel('Year')
p.ylabel('Exoplanets Discovered')
p.show()
