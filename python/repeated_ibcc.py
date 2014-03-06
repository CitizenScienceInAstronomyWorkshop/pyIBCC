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
from random import sample

fname='alpha.pkl'
alphadict={}

def get_training():#this may cause memory errors
    print 'getting training data'
    out=longlist([])
    for i in xrange(len(objects)):
        if i%1000==0:
            overprint('processing line %s of %s' %(add_comma(i),add_comma(len(objects))))
        if labels[i]==1:
            out.append(objects[i])
    out=set(out)
    split=int(round(settings.split_frac*len(out)))
    out=set(sample(out,split))
    print '\n'
    return out

def init_hyper(alpha_dict):#this is currently not set up for general labels but is for general scores
    print 'initialising hyperparameters'
    nu_dict={0:1,1:1}
    
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


def main(data):
    global alpha,nu,results,objects,people,scores,labels
    objects,people,scores,labels=data
    results=mean.mean(data)
    training_planets=get_training()
    
    for i in training_planets:
        results[i]=1.0

    results={i:[1-results[i],results[i]] for i in results.keys()}
    nu0,alpha0=init_hyper(alphadict)
    nu,alpha=nu0,alpha0
    
    
    '''main loop'''
    iteration=0
    while True:
        iteration+=1
        print '\nIteration %d' %iteration
        presults=deepcopy(results)
        
        '''update hyperparameters'''
        print 'updating hyperparameters'
        N_current=[sum([results[i][j] for i in results.keys()]) for j in range(settings.nlabels)]
        for i in range(len(nu)):
            nu[i]=nu0[i]+N_current[i]
        
        alpha=deepcopy(alpha0)
        for a in xrange(len(objects)):
            if a%1000==0:
                overprint('processing line %s of %s' %(add_comma(a),add_comma(len(objects))))
            i=objects[a]
            k=people[a]
            l=scores[a]
            for j in range(settings.nlabels):
                alpha[k][j][l]+=results[i][j]
        print '\nUpdating results'
        results={i:[kappa(j) for j in range(settings.nscores)] for i in results.keys()}#could change it to a defaultdict but this is probably clearer
        for a in xrange(len(objects)):
            if a%1000==0:
                overprint('processing line %s of %s' %(add_comma(a),add_comma(len(objects))))
            i=objects[a]
            k=people[a]
            l=scores[a]
            for j in range(settings.nlabels):
                results[i][j]*=pi(k,j,l)
        #normalise
        for i in results.keys():
            N=sum(results[i])
            if N!=0.0:
                for j in range(settings.nlabels):
                    results[i][j]/=N
        print '\n'     
          
        if check_convergence(presults,results):
            print 'algorithm converged'
            '''print 'calculating confusion matrices'
            final_pi={k:[[pi(k,j,l) for l in range(settings.nscores)] for j in range(settings.nlabels)] for k in people}
            f=open('pi_results.dat','w')
            pickle.dump(final_pi,f)
            f.close()'''
            
            f=open(settings.dir_name+fname,'w')
            pickle.dump(alpha,f)
            f.close()
            return {i:results[i][1] for i in set(results.keys())-training_planets}
        
        
        
    