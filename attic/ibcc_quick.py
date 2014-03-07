'''
Created on Jan 3, 2013

@author: kieranfinn
'''
import numpy as np
#from scipy.special import digamma
from copy import deepcopy
import settings
import pylab as pl
import mean
import pickle
from functions import *
from collections import defaultdict

fname='alpha.pkl'

def refine_data(dat):
    objects,people,scores,labels=dat
    kindices=[]
    uindices=[]
    data=defaultdict(lambda: defaultdict(lambda:0))
    training_data=longlist([])
    for i in range(len(objects)):
        if i%10000==0:
            overprint('Processing line %s of %s' %(add_comma(i),add_comma(len(objects))))
        p=people[i]
        o=objects[i]
        s=scores[i]
        l=labels[i]
        set_item(data,o,p,max(s,data[o][p]))#consider a marked transit in ANY light curve to be marked as a planet
        if l==1:
            training_data.append(o)        
        
        if l==1 and o not in kindices and len(kindices)<5:
            kindices.append(o)
        if l!=1 and o not in uindices and len(uindices)<5:
            uindices.append(o)
        indices=kindices+uindices
    tobjects=set(objects)
    tpeople=set(people)
    split=int(round(settings.split_frac*len(training_data)))
    print '\n%s objects and %s users' %(add_comma(len(tobjects)),add_comma(len(tpeople)))
    return (tobjects,tpeople,data,training_data[:split],indices)

def init_hyper():#this is currently not set up for general labels but is for general scores
    print 'initialising hyperparameters'
    nu_dict={0:1,1:1}
    alpha_dict={(0,0):1,(0,1):1,(1,0):1,(1,1):1}
    
    nu_out=[nu_dict[j] for j in range(settings.nlabels)]
    '''for p in pers: #takes into account people that exist just in the training set. may want to select training set better in future so this doesn't happen
        if p not in people:
            people.append(p)'''
    alpha_out={k:[[alpha_dict[(j,l)] for l in range(settings.nscores)] for j in range(settings.nlabels)] for k in people}#this looks more confusing than it is
    
    '''total_planets=0.0
    total_objects=0.0
    for l in labels:
        if l=='planet':
            total_planets+=1
        elif l!='synthetic':
            total_objects+=1
    f=(total_objects-(total_planets/settings.fraction))/(total_objects-total_planets)
    for i in range(len(obs)):
        p=pers[i]
        s=scores[i]
        l=settings.labeldict[labels[i]]
        if l==settings.unsure_value:
            alpha_out[p][0][s]+=f
            alpha_out[p][1][s]+=(1-f)
        else:
            alpha_out[p][l][s]+=1
    nu_out[1]+=total_planets/settings.fraction
    nu_out[0]+=total_objects-(total_planets/settings.fraction)'''
    
    return (nu_out,alpha_out)
                
def plot_pis(indices,it):
    f,ax=pl.subplots(len(indices),sharex=True)
    for a in range(len(indices)):
        k=people[a]
        x=[]
        y=[]
        for j in range(settings.nlabels):
            for l in range(settings.nscores):
                x.append(l+j*settings.nlabels-0.4)
                y.append(pi(k,j,l))
        ax[a].bar(x,y)
        ax[a].get_yaxis().set_ticks([])
        ax[a].set_ylim(0,1)
    ax[0].set_xlim(-0.5,len(y)-0.5)
    pl.savefig('D:\Documents\images\pi_%03d.png' %it)
    
    f,ax=pl.subplots(len(indices),sharex=True)
    for a in range(len(indices)):
        i=indices[a]
        x=[]
        y=[]
        for j in range(settings.nlabels):
            x.append(j-0.4)
            y.append(results[i][j])
        ax[a].bar(x,y)
        ax[a].get_yaxis().set_ticks([])
        ax[a].set_ylim(0,1)
    ax[0].set_xlim(-0.5,len(y)-0.5)
    pl.savefig('D:\Documents\images\\result_%03d.png' %it)
    pl.close('all')
'''
this is what's given in the paper. Changed it so that we deal with the quantities themselves not the logs
def kappa(j):
    return digamma(nu[j])-digamma(sum(nu))

def pi(k,j,l):
    return digamma(alpha[k][j][l])-digamma(sum(alpha[k][j]))

def rho(i,j):
    out=kappa(j)
    for k in data[i].keys():
        out+=pi(k,j,data[i][k])
    return np.exp(out)
'''
    
def kappa(j):
    return float(nu[j])/float(sum(nu))

def pi(k,j,l):
    return float(alpha[k][j][l])/float(sum(alpha[k][j]))

def rho(i,j):
    out=kappa(j)
    for k in data[i].keys():
        out*=pi(k,j,data[i][k])
    return out

def Ns():
    out={k:[[0.0 for l in range(settings.nscores)] for j in range(settings.nlabels)] for k in people}
    count=0
    for i in objects:
        if count%1000==0:
            overprint('Counting scores for object %s of %s' %(add_comma(count),add_comma(len(objects))))
        count+=1
        for k in data[i].keys():
            for j in range(settings.nlabels):
                for l in range(settings.nscores):
                    out[k][j][l]+=int(l==data[i][k])*results[i][j]
    print '\n'
    return out
    

def check_convergence(presults,results):
    '''may need to change. currently set so convergence means the median difference is less than 10^-4'''
    to_check=[]
    for i in presults.keys():
        for j in range(len(results[i])):
            to_check.append(abs(presults[i][j]-results[i][j]))
    out=np.median(to_check)
    if out>1e-4:
        print 'median difference is %.3e. Continuing algorithm' %out
        return False
    else:
        return True


def main(dat):
    global alpha,nu,data,objects,people,results
    objects,people,data,training_data,indices=refine_data(dat)
    results=mean.mean(dat)
    
    for i in training_data:
        results[i]=1.0

    results={i:[1-results[i],results[i]] for i in results.keys()}
    nu0,alpha0=init_hyper()
    nu,alpha=nu0,alpha0
    
    
    '''main loop'''
    iteration=0
    while True:
        #plot_pis(indices,iteration)
        iteration+=1
        print '\nIteration %d' %iteration
        presults=deepcopy(results)
        
        '''update hyperparameters'''
        print 'updating hyperparameters'
        N_current=[sum([results[i][j] for i in results.keys()]) for j in range(settings.nlabels)]
        for i in range(len(nu)):
            nu[i]=nu0[i]+N_current[i]
        N=Ns()
        for k in people:            
            for j in range(settings.nlabels):
                for l in range(settings.nscores):
                    alpha[k][j][l]=alpha0[k][j][l]+N[k][j][l]
        print '\nUpdating results'
        count=0
        for i in objects:
            if count%1000==0:
                overprint('Updating results for object %s of %s' %(add_comma(count),add_comma(len(objects))))
            count+=1
                
            rhoi=[rho(i,j) for j in range(settings.nlabels)]
            if sum(rhoi)==0:
                results[i]=[0.0 for j in range(settings.nlabels)]
            else:
                for j in range(settings.nlabels):
                    results[i][j]=rhoi[j]/sum(rhoi)
        print '\n'
        if check_convergence(presults,results):            
            f=open(fname,'w')
            pickle.dump(alpha,f)
            f.close()
            return {i:results[i][1] for i in objects}
        
        
        
    