'''
Created on 5 May 2015

@author: edwin
'''
import logging
logging.basicConfig(level=logging.DEBUG)
import ibcc, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

zoodatafile = "./data/rescue_global_nepal_2015.csv"
# format of file: 
# user_id,user_ip,workflow_id,created_at,gold_standard,expert,metadata,annotations,subject_data
zoodata = pd.read_csv(zoodatafile, sep=',', parse_dates=False, index_col=False, usecols=[0,1,7,8], 
                      skipinitialspace=True, quotechar='"')
userid = zoodata['user_id']
userip = zoodata['user_ip']
subjectdata = zoodata['subject_data']
annotations = zoodata['annotations']

agentids = {}
subjectids = {}
reverse_subjectids = {}
Cagents = []
Cobjects = []
Cscores = []
for i, user in enumerate(userid):
    annotation = json.loads(annotations[i])
    score = annotation[0]["value"]
    if score==6:
        continue
    else:
        Cscores.append(score)
    if not user or np.isnan(user):
        user = userip[i]
    if not user in agentids:
        agentids[user] = len(agentids.keys()) 
    Cagents.append(agentids[user])
    subjectdict = json.loads(subjectdata[i])
    subject = int(subjectdict.keys()[0])
    if not subject in subjectids:
        subjectids[subject] = len(subjectids.keys())
        reverse_subjectids[subjectids[subject]] = subject
    Cobjects.append(subjectids[subject])
     
osmfile = "./data/OSM_labels.csv"
osmdata = pd.read_csv(osmfile, sep=',', parse_dates=False, index_col=False, skipinitialspace=True, quotechar='"', 
                    header=None, names=['subject_id','value'])
osm_subjects = osmdata["subject_id"]# alpha0 = np.tile(alpha0[:,:,np.newaxis], (1,1,len(agentids)))
osm_scores = osmdata["value"] - 1
agentids["OSMData"] = len(agentids.keys())
for i, subject in enumerate(osm_subjects):
    Cagents.append(agentids["OSMData"])
    if not subject in subjectids:
        subjectids[subject] = len(subjectids.keys())
        reverse_subjectids[subjectids[subject]] = subject
    Cobjects.append(subjectids[subject])
    score = osm_scores[i]
    Cscores.append(score)
    
Cagents = np.array(Cagents)[:,np.newaxis]
Cobjects = np.array(Cobjects)[:, np.newaxis]
Cscores = np.array(Cscores)[:, np.newaxis]    
C = np.concatenate((Cagents,Cobjects,Cscores), axis=1)
alpha0 = np.ones((6,6,len(agentids)))
alpha0[:, :, 5] = 2.0
alpha0[np.arange(6),np.arange(6),:] += 1.0
# alpha0[:,:,:] = np.array([[4.0, 2.0, 1.5, 1.0, 1.0, 2.0], [2.0, 4.0, 2.0, 1.5, 1.0, 2.5], [1.5, 2.0, 4.0, 2.0, 1.5, 2.5], 
#                         [1.0, 1.5, 2.0, 4.0, 2.0, 2.5], [1.0, 1.0, 1.5, 2.0, 4.0, 3.0], [1.0, 1.0, 1.0, 1.0, 1.0, 4.0]])[:,:,np.newaxis]
# alpha0 = np.tile(alpha0[:,:,np.newaxis], (1,1,len(agentids)))
alpha0[np.arange(6),np.arange(6),-1] += 20
# alpha0[:, 5, -1] += 50
nu0 = np.array([1,1,1,1,1,1], dtype=float)
combiner = ibcc.IBCC(nclasses=6, nscores=6, alpha0=alpha0, nu0=nu0)
preds = combiner.combine_classifications(C)

from scipy.stats import beta
plt.figure()
# for k in range(combiner.alpha.shape[2]):
k = 0 # worker ID to plot
alpha_k  = combiner.alpha[:, :, k]
pi_k = alpha_k / np.sum(alpha_k, axis=1)[:, np.newaxis]
print "Confusion matrix for worker %i" % k
print pi_k
    
for j in range(alpha_k.shape[0]):
    pdfj = beta.pdf(np.arange(20) / 20.0, alpha_k[j, j], np.sum(alpha_k[j, :]) - alpha_k[j,j] )
    plt.plot(pdfj, label='True class %i' % j)
plt.legend(location='best')


results_subjectids = []
for i in range(preds.shape[0]):
    results_subjectids.append(reverse_subjectids[i])
results_subjectids = np.array(results_subjectids)#                                                                     skipinitialspace=True, quotechar='"', header=None, names=['subject_id','x','y'] )

# get the coordinates for the subjects and save to another file
nsubjects = len(results_subjectids)
minxarr = np.zeros(nsubjects)
minyarr = np.zeros(nsubjects)
maxxarr = np.zeros(nsubjects)
maxyarr = np.zeros(nsubjects)
for i, subjectstr in enumerate(subjectdata):
    subject = json.loads(subjectstr)
    sidstr = subject.keys()[0]
    sid = int(subject.keys()[0])
    if not sid in subjectids:
        continue 
    sidx = subjectids[sid]
    minxarr[sidx] = subject[sidstr]["minx"]
    minyarr[sidx] = subject[sidstr]["miny"]
    maxxarr[sidx] = subject[sidstr]["maxx"]
    maxyarr[sidx] = subject[sidstr]["maxy"]
    
results = pd.DataFrame(data={'subject_id':results_subjectids, 'priority1': preds[:,0], 'priority2':preds[:,1],
                             'priority3':preds[:,2], 'priority4':preds[:,3], 'priority5':preds[:,4], 
                             'no_priority':preds[:,5], 'minx':minxarr, 'miny':minyarr, 'maxx':maxxarr, 'maxy':maxyarr}, 
                       index=None)
results.to_csv("./output/zooresults_osm.csv", sep=',', index=False, float_format='%1.4f', 
               cols=['subject_id','priority1','priority2','priority3','priority4','priority5','no_priority','minx','miny','maxx','maxy'])    

nepal_subjects = []
for subject in results_subjectids:
    if subject in np.array(osm_subjects):
        nepal_subjects.append(subjectids[subject])
preds_nepal = preds[nepal_subjects,:]

print np.around(combiner.alpha[:,:,56] - alpha0[:,:,-1], 3)
print np.around(np.sum(combiner.alpha[:,:,0:56], axis=2),3)

idxs = (Cagents==56)
objs = Cobjects[idxs]
scores = Cscores[idxs]
osm_top_objs = objs[scores<=2]
preds_osmtop = np.around(preds[osm_top_objs,:], 2)
local_conflict_ids = osm_top_objs[np.sum(preds_osmtop[:,0:3],axis=1)<0.5]
print np.around(preds[local_conflict_ids,:], 2)

osm_empty_objs = objs[scores>=4]
preds_osmempty = np.around(preds[osm_empty_objs,:], 2)
local_conflict_ids = osm_empty_objs[np.sum(preds_osmempty[:,2:],axis=1)<0.2]
zoo_conflict_ids = results_subjectids[local_conflict_ids]
print zoo_conflict_ids
print np.around(preds[local_conflict_ids,:], 2)

coordsfile = './data/transformed_subject_id_metadata_Kathmandu_ring_1.csv'
coordsdata = pd.read_csv(coordsfile, sep=',', parse_dates=False, index_col=False, usecols=[0,2,3], 
                    skipinitialspace=True, quotechar='"', header=None, names=['subject_id','x','y'] )

osmresults = np.zeros(len(osm_subjects))
crowdpreds = np.zeros((len(osm_subjects), 6))
xcoords = np.zeros(len(osm_subjects))
ycoords = np.zeros(len(osm_subjects))
for i, s in enumerate(osm_subjects):
    sidx = subjectids[s]
    crowdpreds[i] = preds[sidx, :]
    osmresults[i] = osm_scores[i]
    for j, s2 in enumerate(coordsdata['subject_id']):
        if s2==s:
            xcoords[i] = coordsdata['x'][j]
            ycoords[i] = coordsdata['y'][j]

# get the chosen category from the crowd
cs = np.cumsum(crowdpreds, axis=1)
c = 5
crowdresults = np.zeros(len(osm_subjects))
while c>=0:
    crowdresults[cs[:,c]>=0.9] = c
    c -= 1
    
# chose the minimum from the two sets
combinedresults = crowdresults#np.min([osmresults, crowdresults], axis=0)    

output = np.concatenate((osm_subjects[:, np.newaxis], combinedresults[:, np.newaxis]), axis=1)

np.savetxt("./output/combined_categories.csv", output, fmt="%i", delimiter=',')

combinedresults = combinedresults[3:]
xcoords = xcoords[3:]
ycoords = ycoords[3:]

nx = len(np.unique(xcoords))
ny = len(np.unique(ycoords))

grid = np.empty((nx+1, ny+1))
grid[:] = np.nan

xgrid = (xcoords-np.min(xcoords)) / float(np.max(xcoords)-np.min(xcoords)) * nx
ygrid = (ycoords-np.min(ycoords)) / float(np.max(ycoords)-np.min(ycoords)) * ny
xgrid = np.round(xgrid).astype(int)
ygrid = np.round(ygrid).astype(int)
grid[xgrid, ygrid] = combinedresults

dpi = 96.0
fig = plt.figure(frameon=False)#, figsize=(float(nx)/dpi,float(ny)/dpi))
plt.autoscale(tight=True)
#Can also try interpolation=nearest or none
ax = fig.add_subplot(111)
ax.set_axis_off()    
        
# bin the results so we get contours rather than blurred map
# grid = grid.T
contours = np.zeros((grid.shape[0], grid.shape[1], 4))#bcc_pred.copy()
contours[grid==4, :] = [0, 1, 1, 0.7]
contours[grid==3, :] = [0, 1, 0, 0.7]
contours[grid==2, :] = [1, 1, 0, 0.7]
contours[grid==1, :] = [1, 0.2, 0, 0.7]
contours[grid==0, :] = [1, 0, 0.5, 0.7]
 
plt.imshow(contours, aspect=None, origin='lower', interpolation='nearest')

fig.tight_layout(pad=0,w_pad=0,h_pad=0)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())

plt.savefig('./output/popdensity.png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=96)  

gridsize_lat = float(np.max(xcoords)-np.min(xcoords)) / float(nx)
gridsize_lon = float(np.max(ycoords)-np.min(ycoords)) / float(ny)

print np.min(xcoords)
print np.max(xcoords) + gridsize_lat
print np.min(ycoords)
print np.max(ycoords) + gridsize_lon