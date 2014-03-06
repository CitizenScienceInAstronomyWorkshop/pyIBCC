'''
Created on 5 Jan 2013

@author: Kieran Finn
'''
import numpy as np
from random import random
import pylab as p
from mpl_toolkits.mplot3d import axes3d #used for 3d axese

def get3dax(): #returns a pylab 3d axis instance labelled with x, y and z
    f=p.figure()
    out=f.gca(projection='3d')
    return out

def d(pi1,pi2):
    return 1-sum([pi2[i]*pi1[i] for i in range(len(pi1))])

def initial_matrix(pi):
    index2id={}#may need to add reverse in here
    i=0
    for j in pi.keys():
        index2id[i]=j
        i+=1
    N=i
    out=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            m=index2id[i]
            n=index2id[j]
            out[i][j]=np.exp(0.0-sum([d(pi[m][k],pi[n][k]) for k in range(len(pi[m]))]))
    return (np.matrix(out),index2id)

def elem_operate(A,B,operation):
    A=np.asarray(A)#can I do this without converting to array?
    B=np.asarray(B)
    out=np.zeros_like(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if operation=='multiply':
                out[i][j]=A[i][j]*B[i][j]
            elif operation=='divide':
                try:
                    out[i][j]=A[i][j]/B[i][j]
                except:
                    print A[i][j],B[i][j]
            else:
                print 'operation not understood'
                return False
    return np.matrix(out)

def elem_multiply(A,B):
    return elem_operate(A,B,'multiply')

def elem_divide(A,B):
    return elem_operate(A,B,'divide')

def identity(dim):
    return np.matrix(np.identity(dim))

def B_mat(beta):
    k=len(beta)
    out=np.zeros((k,k))
    for i in range(k):
        out[i][i]=beta[i]
    return np.matrix(out)

def random_init(a,b):
    out=np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            out[i][j]=random()
    return np.matrix(out)

''' old method to check for convergence. using cost_function should be better
def check_convergence(preW,W):
    preW=np.asarray(preW)
    W=np.asarray(W)
    for i in range(len(W)):
        preN=sum([preW[i][l] for l in range(len(W[i]))])
        N=sum([W[i][l] for l in range(len(W[i]))])
        for j in range(len(W[0])):
            preW[i][j]/=preN
            W[i][j]/=N
            if W[i][j]>0.01 and W[i][j]<0.99 and abs(preW[i][j]-W[i][j])/W[i][j]>0.1:
                return False
    return True'''

def cost_function(V,H,W,beta,N,K,a,b):
    V_tild=W*H
    t1=0
    for i in range(N):
        for j in range(N):
            t1+=V[i,j]*np.log(V[i,j]/V_tild[i,j])+V_tild[i,j]
    t2=0
    for k in range(K):
        t2+=sum([beta[k]*W[i,k]*W[i,k] for i in range(N)])+sum([beta[k]*H[k,j]*H[k,j] for j in range(N)])-2*N*np.log(beta[k])
    t3=sum([beta[k]*b-(a-1)*np.log(beta[k]) for k in range(K)])
    return t1+0.5*t2+t3
    
    
def clean(mat):
    mat=np.asarray(mat)
    out=[]
    for i in range(len(mat)):
        if sum(mat[i])>0.01:#possibility to change here
            out.append(mat[i])
    return np.array(out)

def extract_communities(pi,results,index2id):
    out=[]
    for i in pi.keys():
        pi[i]=np.array(pi[i])
    for k in range(len(results)):
        print 'new catagory'
        temp=np.zeros_like(pi[index2id[0]])
        N=0
        for i in range(len(results[k])):
            print pi[index2id[i]],results[k][i],temp
            temp+=pi[index2id[i]]*results[k][i]
            N+=results[k][i]
        out.append(temp/N)
    return out

def plot_communities(communities):
    colours=['b','r','g','y']    
    for pi in communities:
        dx=[]
        dy=[]
        dz=[]
        x=[]
        y=[]
        z=[]
        c=[]
        for i in range(len(pi)):
            for j in range(len(pi[0])):
                dz.append(pi[i][j])
                dx.append(1)
                dy.append(1)
                x.append(i-0.5)
                y.append(j-0.5)
                z.append(0)
                c.append(colours[j])
        ax=get3dax()
        ax.set_xlabel('True Label')
        ax.set_ylabel('Score')
        ax.set_zlabel('Probability')
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.bar3d(x,y,z,dx,dy,dz,color=c)
        ax.set_zlim(0,1)
        
def plot_person(y,it):
    x=range(len(y))
    for i in range(len(x)):
        x[i]-=0.4
    p.clf()
    N=sum(y)
    for i in range(len(y)):
        y[i]/=N
    p.bar(x,y)
    p.savefig('%03d.png' %it)
    
def plot_people(y,it,normalise=True):
    K=len(y)
    x=range(len(y[0]))
    for i in range(len(x)):
        x[i]-=0.4
    f,ax=p.subplots(K,sharex=True)
    for i in range(K):
        if normalise:
            N=sum(y[i])
            for j in range(len(y[i])):
                y[i][j]/=N
        ax[i].bar(x,y[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].set_ylim(0,1)
    ax[0].set_xlim(-0.5,len(y[0])-0.5)
    p.savefig('%03d.png' %it)
    
def normalise(mat):
    mat=np.asarray(mat)
    for i in range(len(mat)):
        N=sum(mat[i])
        for j in range(len(mat[i])):
            mat[i][j]/=N
    return np.matrix(mat)
          

def main(pi):
    '''initialise'''
    N=len(pi)
    V,index2id=initial_matrix(pi)
    K=N #initial number of classes (most general set equal to number of people
    a,b=1.0,1.0 #hyperparameters, not sure what to set these to
    W=random_init(N,K)
    H=random_init(K,N)
    beta=np.zeros(K)
    I=identity(K)
    cf=[]
    max_cf=cost_function(V,H,W,beta,N,K,a,b)
    
    '''main loop'''
    converged=False
    iteration=1
    while iteration<24 and not converged:
        print 'Iteration %d' %iteration 
        #plot_people(np.asarray(W),iteration)
        iteration+=1
        B=B_mat(beta)
        
        #update H
        denom=W.T*I+B*H
        pre=elem_divide(H,denom)
        far_post=elem_divide(V,W*H)
        post=W.T*far_post
        H=elem_multiply(pre,post)
        
        #update W
        denom=I*H.T+W*B
        pre=elem_divide(W,denom)
        mid=elem_divide(V,W*H)
        post=mid*H.T
        W=elem_multiply(pre,post)
        
        #update beta
        for k in range(K):
            denom=sum([W[i,k]**2 for i in range(N)])+sum([H[k,j]**2 for j in range(N)])
            denom=0.5*denom +b
            num=N+a-1
            beta[k]=num/denom
           
        cf0=cost_function(V,H,W,beta,N,K,a,b)
        cf.append(cf0)
        converged=(cf0<0.01*max_cf)#rather ad-hoc, may want to tighten this up
    
    #W and H^T should be the same at this point so reduce to one but they're not. Not sure what's going on
    W=normalise(W)
    results=clean(W.T)
    communities=extract_communities(pi,results,index2id)
    plot_communities(communities)
    p.figure()
    p.plot(cf)
    p.show()
    return (communities,results,W,H,index2id)
    
    
    