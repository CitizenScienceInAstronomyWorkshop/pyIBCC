'''
@author: Edwin Simpson
'''
import sys
import numpy as np
from copy import deepcopy
from scipy.special import psi, gammaln
import ibcc

def state_to_alpha(logodds, var):
    alpha1 = 1/var * (1+np.exp(logodds))
    alpha2 = alpha1 * (1+np.exp(-logodds))
    return alpha1.reshape(logodds.shape), alpha2.reshape(logodds.shape)

def alpha_to_state(alpha, l):
    Wmean = np.log(alpha[:,l])-np.log(np.sum(alpha,axis=1)-alpha[:,l])
    P = np.diagflat( (1/alpha[:,l]) + \
        (1/(np.sum(alpha,axis=1)-alpha[:,l])) )
    return Wmean, P

class Dynibcc(ibcc.Ibcc):
        
    Tau = 1 # the total number of responses from all crowd members

    def postAlpha(self):#Posterior Hyperparams        
        if self.nClasses>2:
            for l in range(1,self.nScores):
                self.postAlpha_binary(l)
        else:
            self.postAlpha_binary(1)
    
    def postAlpha_binary(self, l=1):
        #l is the index into alpha we are dealing with
        #FILTERING -- UPDATES GIVEN PREVIOUS TIMESTEPS
        #p(\pi_t | data up to and including t)
        Wmean_po = np.zeros((self.nClasses, self.Tau))
        P_po = np.zeros((self.nClasses,self.nClasses,self.Tau))
        q = np.zeros(self.K)
        tau_prev = np.zeros(self.K) -1
        I = np.eye(self.nClasses)
        
        eta_pr = np.zeros(self.Tau)
        r_pr = np.zeros(self.Tau)
        
        Kalman = np.zeros((self.nClasses, self.Tau))
        
        for tau in range(self.Tau): 
            k = self.crowdlabels[tau,0]
            i = self.crowdlabels[tau,1]
            c = int(self.crowdlabels[tau,2]==l)
            tau_pr = tau_prev[k]
            tau_prev[k] = k
            h = self.ET[i,:]
            
            if tau_pr == -1:
                Wmean_pr, P_pr = alpha_to_state(self.alpha0[:,:,tau], l)
            else:
                Wmean_pr = Wmean_po[:,tau_pr]
                P_pr = P_po[:,:,tau_pr] + q[k]*I
            
                pi_var = np.diag(P_pr)
            
                alpha0_l, alpha0sum = state_to_alpha(Wmean_pr,pi_var)
                self.alpha0[:,l,tau] = alpha0_l.reshape((self.nClasses,1))
                if self.nClasses==2:
                    self.alpha0[:,1-l,tau] = alpha0sum.reshape((self.nClasses,1))-self.alpha0[:,l,tau]
            
            eta_pr[tau] = h.T.dot(Wmean_pr)
            r_pr[tau] = h.T.dot(P_pr).dot(h)
            
            alpha_tilde_pr,alpha_tilde_pr_sum = state_to_alpha(eta_pr[tau], r_pr[tau])

            #update to get posterior given current time-step
            alpha_tilde_po = alpha_tilde_pr + c
            alpha_tilde_po_sum = alpha_tilde_pr_sum + 1
            
            #check r_po
            eta_po = np.log(alpha_tilde_po/alpha_tilde_po_sum)
            r_po = (1/alpha_tilde_po) + (1/alpha_tilde_po_sum)
            
            z = eta_po - eta_pr[tau]
            
            pi_tilde_pr = alpha_tilde_pr/alpha_tilde_pr_sum
            u_pr = pi_tilde_pr * (1-pi_tilde_pr)
            pi_tilde_po = alpha_tilde_po/alpha_tilde_po_sum
            u_po = pi_tilde_po * (1-pi_tilde_po)
            q[k] = (u_po>u_pr) * (u_po - u_pr)
            
            Kalman[:,tau] = P_pr.T.dot(h) / r_pr[tau]
            R = 1 - (r_po/r_pr[tau])
            
            Wmean_po[:,tau] = Wmean_pr.reshape(self.nClasses) + Kalman[:,tau]*z
            P_po[:,:,tau] = P_pr - (Kalman[:,tau].dot(P_pr)*R)
            
        #SMOOTHING -- UPDATES GIVEN ALL TIMESTEPS
        #pi(\pi_t | all data up to time self.Tau)
        lambda_mean = np.zeros((self.nClasses, self.K))
        Lambda_cov = np.zeros((self.nClasses, self.nClasses, self.K))
        
        for tau in list(reversed(range(self.Tau))):
            k = self.crowdlabels[tau,0]
            i = self.crowdlabels[tau,1]
            h = self.ET[i,:]
                        
            delta_Wmean = P_po[:,:,tau].dot(lambda_mean[:,k])
            delta_P = P_po[:,:,tau].dot(Lambda_cov[:,:,k]).dot(P_po[:,:,tau].T)
            
            Wmean_po[:,tau] = Wmean_po[:,tau] - delta_Wmean
            P_po[:,:,tau] = P_po[:,:,tau]-delta_P
            
            eta_po = h.T.dot(Wmean_po[:,tau])
            r_po = h.T.dot(P_po[:,:,tau]).dot(h)
            
            z = eta_po - eta_pr[tau]
            R = 1 - (r_po/r_pr[tau])
            B = I - Kalman[:,tau].dot(h.T)
            
            lambda_mean[:,k] = B.T.dot(lambda_mean[:,k]) - h*z/r_pr[tau]
            Lambda_cov[:,:,k] = B.T.dot(Lambda_cov[:,:,k]).dot(B) + (h*R/r_pr[tau]).dot(h.T)
            
            pi_var = np.diag(P_po[:,:,tau])
            
            alpha_l, alphasum = state_to_alpha(Wmean_po[:,tau],pi_var)
            self.alpha[:,l,tau] = alpha_l.reshape((self.nClasses,1))
            alphasum = alphasum.reshape((self.nClasses,1))
            if self.nClasses==2:
                self.alpha[:,1-l,tau] = alphasum - self.alpha[:,l,tau]
                
    def expecLnPi(self):
        self.postAlpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for s in range(self.nScores):        
            self.lnPi[:,s,:] = psi(self.alpha[:,s,:]) - psiSumAlpha
    
    def lowerBound(self, lnjoint):
                        
        #probability of these targets is 1 as they are training labels
        #lnjoint[self.trainT!=-1,:] -= np.reshape(self.lnKappa, (1,self.nClasses))
        lnpCT = self.postLnJoint(lnjoint)                    
        
        # !!! Need to replace? Record the priors at each step. Save alpha0 as alpha0[:,:,0].
        #alpha0 then refers to the prior at each step.
        # !!! Does it need reshaping?
        #alpha0 = np.reshape(self.alpha0, (self.nClasses, self.nScores, 1))
        lnpPi = gammaln(np.sum(self.alpha0, 1))-np.sum(gammaln(self.alpha0),1) \
                    + np.sum((self.alpha0-1)*self.lnPi, 1)
        lnpPi = np.sum(np.sum(lnpPi))
            
        lnpKappa = self.postLnKappa()
            
        EEnergy = lnpCT + lnpPi + lnpKappa
        
        ET = self.ET[self.ET!=0]
        lnqT = np.sum( ET*np.log(ET) )

        lnqPi = gammaln(np.sum(self.alpha, 1))-np.sum(gammaln(self.alpha),1) + \
                    np.sum( (self.alpha-1)*self.lnPi, 1)
        lnqPi = np.sum(np.sum(lnqPi))        
            
        lnqKappa = self.qLnKappa()
            
        H = - lnqT - lnqPi - lnqKappa
        L = EEnergy + H
        
        #print 'EEnergy ' + str(EEnergy) + ', H ' + str(H)
        return L
         
    def preprocessCrowdLabels(self, crowdLabels):
        super(Dynibcc,self).preprocessCrowdLabels(crowdLabels)
        
        if self.crowdTable != None:
            self.Tau = self.crowdTable.shape[0] * self.crowdTable.shape[1]
            print "implementation for table format not complete -- no way of knowing which order data points occurred in"
        else:
            self.Tau = self.crowdlabels.shape[0]
            
        #need to re-init pi now we know how many time steps there are 
        self.initLnPi()
         
    def initLnPi(self):       
        if self.alpha!=[] and self.alpha.shape[2]==self.Tau:
            return
        
        if self.alpha!=[]:
            oldTau = self.alpha0.shape[2]
            
            ending = self.alpha0[:,:,oldTau-1]
            ending = np.repeat(ending, self.Tau-oldTau, axis=2)
            self.alpha0 = np.concatenate(self.alpha0, ending, axis=2)

            ending = self.alpha[:,:,oldTau-1]
            ending = np.repeat(ending, self.Tau-oldTau, axis=2)
            self.alpha = np.concatenate(self.alpha, ending, axis=2)
        else:    
            self.alpha0 = np.float64(self.alpha0[:,:,np.newaxis])
            self.alpha0 = np.repeat(self.alpha0, self.Tau, axis=2)
            self.alpha = deepcopy(self.alpha0)        
        
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        self.lnPi = np.zeros((self.nClasses,self.nScores,self.Tau))
        self.lnPi = psi(self.alpha) - psiSumAlpha 
       

    def __init__(self, nClasses=2, nScores=2, alpha0=None, nu0=None, K=1, tableFormat=False, dh=None):
        super(Dynibcc,self).__init__(nClasses, nScores, alpha0, nu0, K, tableFormat, dh)
#        
#     def __init__(self, nClasses, nScores, alpha0, nu0, K, tableFormat=False):
#         self.nClasses = nClasses
#         self.nScores = nScores
#         self.alpha0 = alpha0
#         self.nu0 = nu0
#         self.K = K
#         self.initParams()
#         if tableFormat:
#             self.crowdTable = True
#         else:
#             self.crowdTable = None             

#TODO convert these to static class methods so we can inherit from Ibcc
def initFromConfig(configFile, K, tableFormat=False):
    nClasses = 2
    scores = np.array([3, 4])
    nu0 = np.array([50.0, 50.0])
    alpha0 = np.array([[2, 1], [1, 2]])     
        
    #read configuration
    with open(configFile, 'r') as conf:
        configuration = conf.read()
        exec(configuration)        
        
    nScores = len(scores)
    #initialise combiner
    combiner = Dynibcc(nClasses, nScores, alpha0, nu0, K, tableFormat)
    
    return combiner

def save_pi(alpha, nClasses, nScores, K, confMatFile):
    #TODO update this to dynamic matrices?
    #write confusion matrices to file if required
    if not confMatFile is None:
        print 'writing confusion matrices to file'
        pi = alpha
        for l in range(nScores):
            pi[:,l,:] = np.divide(alpha[:,l,:], np.sum(alpha,1) )
        
        flatPi = pi.reshape(1, nClasses*nScores, K)
        flatPi = np.swapaxes(flatPi, 0, 2)
        np.savetxt(confMatFile, flatPi.reshape(K, nClasses*nScores), fmt='%1.3f')    
    
    
if __name__ == '__main__':
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    ibcc.runIbcc(configFile, Dynibcc)
    