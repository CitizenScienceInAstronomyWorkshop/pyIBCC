'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
import ibcc
from scipy.linalg import block_diag
from scipy.special import gammaln

def state_to_alpha(logodds, var):
    alpha1 = 1/var * (1+np.exp(logodds))
    alpha2 = alpha1 * (1+np.exp(-logodds))
    return alpha1.reshape(logodds.shape), alpha2.reshape(logodds.shape)

def alpha_to_state(alpha, l):
    Wmean = np.log(alpha[:, l]) - np.log(np.sum(alpha, axis=1) - alpha[:, l])
    Wmean = Wmean.flatten('F')    
    diag_vals = (1 / alpha[:, l]) + (1 / (np.sum(alpha, axis=1) - alpha[:, l]))
    P = np.diagflat(diag_vals)
    return Wmean, P

class DynIBCC(ibcc.IBCC):
    Tau = 1 # the total number of responses from all crowd members
    tauidxs_test = [] # the time-step indexes of the test data points
    alpha0_tau = [] # the hyper-parameters copied out for each time-step. alpha0 has only one matrix for each agent. 
# Initialisation ---------------------------------------------------------------------------------------------------
    def __init__(self, nclasses=2, nscores=2, alpha0=None, nu0=None, K=1, dh=None):
        super(DynIBCC, self).__init__(nclasses, nscores, alpha0, nu0, K, dh)
        
    def init_lnPi(self):
        '''
        Always creates new self.alpha and self.lnPi objects and calculates self.alpha and self.lnPi values according to 
        either the prior, or where available, values of self.E_t from previous runs.
        '''
        self.alpha0 = self.alpha0.astype(float)
        # if we specify different alpha0 for some agents, we need to do so for all K agents. The last agent passed in 
        # will be duplicated for any missing agents.
        if len(self.alpha0.shape)==2:
            self.alpha0  = self.alpha0[:,:,np.newaxis]
        if self.alpha0.shape[2] < self.K:
            # We have a new dataset with more agents than before -- create more priors.
            nnew = self.K - self.alpha0.shape[2]
            alpha0new = self.alpha0[:, :, 0]
            alpha0new = alpha0new[:, :, np.newaxis]
            alpha0new = np.repeat(alpha0new, nnew, axis=2)
            self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
        # Make sure self.alpha is the right size as well. Values of self.alpha not important as we recalculate below
        self.alpha = np.zeros((self.nclasses, self.nscores, self.Tau), dtype=np.float)
        self.lnPi = np.zeros((self.nclasses, self.nscores, self.Tau))        
        self.expec_lnPi()        
        
    def init_params(self, force_reset=False):
        '''
        Checks that parameters are intialized, but doesn't overwrite them if already set up.
        '''
        if self.verbose:
            logging.debug('Initialising parameters...Alpha0: ' + str(self.alpha0))
        if self.table_format_flag:
            # select the right function for the data
            self.post_Alpha_binary = self.post_Alpha_binary_table
        else:
            self.post_Alpha_binary = self.post_Alpha_binary_sparselist
        # get a table to indicate where responses are missing
        self.Cknown = self.C[0].copy()
        for l in range(1,self.nscores):
            self.Cknown += self.C[l]
        #if alpha is already initialised, and no new agents, skip this
        if self.alpha == [] or self.alpha.shape[2] != self.Tau or force_reset:
            self.init_lnPi()
        if self.verbose:
            logging.debug('Nu0: ' + str(self.nu0))
        if self.nu ==[] or force_reset:
            self.init_lnkappa()

    def preprocess_crowdlabels(self, crowdlabels):
        # Initialise all objects relating to the crowd labels.
        self.C = {}
        self.Ctest = {}
        crowdlabels[np.isnan(crowdlabels)] = -1
        if self.discretedecisions:
            crowdlabels = np.round(crowdlabels).astype(int)
        if self.table_format_flag:# crowd labels as a full KxN table? If false, use a sparse 3-column list, where 1st
            # column=classifier ID, 2nd column = obj ID, 3rd column = score.
            self.K = crowdlabels.shape[1]
            for l in range(self.nscores):
                Cl = np.zeros((self.N, self.K))
                #crowd labels may not be supplied for all N data points in the gold labels, so use argwhere
                lidxs = np.argwhere(crowdlabels==l)
                Cl[lidxs[:,0], lidxs[:,1]] = 1
                if not self.discretedecisions:
                    if l + 1 < self.nscores:
                        partly_l_idxs = np.bitwise_and(crowdlabels > l, crowdlabels < (l+1))  # partly above l
                        Cl[partly_l_idxs] = (l + 1) - crowdlabels[partly_l_idxs]
                    if l > 0:
                        partly_l_idxs = np.bitwise_and(crowdlabels < l, crowdlabels > (l-1))  # partly below l
                        Cl[partly_l_idxs] = crowdlabels[partly_l_idxs] - l + 1
                self.C[l] = Cl
            # pre-compute the indices into the pi arrays
            # repeat for test labels only                
            for l in range(self.nscores):
                self.Ctest[l] = self.C[l][self.testidxs, :] 
                
            # Set the number of time-steps
            self.Tau = self.N * self.K
            # indexes for calculating the joint likelihood efficiently
            self.tauidxs_test = np.arange(self.Tau, dtype=int).reshape((self.N, self.K))[self.testidxs, :]
        else:
            self.K = np.nanmax(crowdlabels[:,0])+1 # add one because indexes start from 0
            self.Cagents = crowdlabels[:,0].astype(int)
            self.Cobjects = crowdlabels[:,1].astype(int)
            testidxs = [i for i in range(len(self.Cobjects)) if self.testidxs[self.Cobjects[i]]]
            self.Cagents_test = self.Cagents[testidxs]
            self.Cobjects_test = self.Cobjects[testidxs]
            for l in range(self.nscores):
                Cl = crowdlabels[:, 2] == l
                if not self.discretedecisions:
                    partly_l_idxs = np.bitwise_and(crowdlabels[:, 2] > l, crowdlabels[:, 2] < l + 1)  # partly above l
                    Cl[partly_l_idxs] = (l + 1) - crowdlabels[partly_l_idxs, 2]
                    
                    partly_l_idxs = np.bitwise_and(crowdlabels[:, 2] < l, crowdlabels[:, 2] > l - 1)  # partly below l
                    Cl[partly_l_idxs] = crowdlabels[partly_l_idxs, 2] - l + 1
                self.C[l] = Cl
                self.Ctest[l] = Cl[testidxs]
            # Set the number of time-steps
            self.Tau = len(self.C[0])
            self.tauidxs_test = testidxs
        # Set and reset object properties for the new dataset
        self.lnpCT = np.zeros((self.N, self.nclasses))
        self.conf_mat_ind = []
        # Reset the pre-calculated data for the training set in case goldlabels has changed
        self.alpha_tr = []

# Posterior Updates to Hyper-parameters -----------------------------------------------------------------------------
    def post_Alpha(self):#Posterior update to hyper-parameters 
        if self.nclasses>2:
            for l in range(self.nscores):
                self.post_Alpha_binary(l)
        else:
            self.post_Alpha_binary(1)

    def h_cov(self, n):
        h = self.E_t[n, :].reshape((1,1,self.nclasses))
            # the expanded vector to use with block-diagonal covariance
        return block_diag(*np.tile(h, (self.K, 1, 1))).T  # nclasses*K x K):
               
    def post_Alpha_binary_sparselist(self, l=1):
        #l is the index into alpha we are dealing with
        #FILTERING -- UPDATES GIVEN PREVIOUS TIMESTEPS
        #p(\pi_t | data up to and including t)
        # Variables used in smoothing step but calculated during filtering
        Wmean_po = np.zeros((self.nclasses, self.Tau))
        P_po = np.zeros((self.nclasses,self.nclasses,self.Tau))
        Kalman = np.zeros((self.nclasses, self.Tau))
        # filtering variables
        q = np.zeros(self.K)
        tau_prev = np.zeros(self.K) -1
        I = np.eye(self.nclasses)
        eta_pr = np.zeros(self.Tau)
        r_pr = np.zeros(self.Tau)
        # loop through all timesteps individually by iterating through each row of the sparselist of crowd labels
        tau = 0
        while tau < self.Tau:
            k = self.Cagents[tau]
            i = self.Cobjects[tau]
            c = self.C[l][tau]
            tau_pr = tau_prev[k]
            tau_prev[k] = tau
            h = self.E_t[i, :].reshape((self.nclasses, 1))  # column vector
            if tau_pr == -1:
                Wmean_pr, P_pr = alpha_to_state(self.alpha0[:,:,k], l)  # vector of size nclasses, matrix nclasses^2
            else:
                Wmean_pr = Wmean_po[:,tau_pr]
                P_pr = P_po[:,:,tau_pr] + q[k]*I
            eta_pr[tau] = h.T.dot(Wmean_pr)
            r_pr[tau] = h.T.dot(P_pr).dot(h)
            alpha_tilde_pr,alpha_tilde_pr_sum = state_to_alpha(eta_pr[tau], r_pr[tau])
            #update to get posterior given current time-step
            alpha_tilde_po = alpha_tilde_pr + c
            alpha_tilde_po_sum = alpha_tilde_pr_sum + 1
            #check r_po
            eta_po = np.log(alpha_tilde_po/alpha_tilde_po_sum)
            r_po = (1/alpha_tilde_po) + (1/alpha_tilde_po_sum)
            # calculate update vector
            z = eta_po - eta_pr[tau]
            # estimated prior mean and uncertainty
            pi_tilde_pr = alpha_tilde_pr/alpha_tilde_pr_sum
            u_pr = pi_tilde_pr * (1-pi_tilde_pr)
            # estimated means given data up to time tau
            pi_tilde_po = alpha_tilde_po/alpha_tilde_po_sum
            u_po = pi_tilde_po * (1-pi_tilde_po)
            q[k] = (u_po>u_pr) * (u_po - u_pr)
            
            Kalman_tau = P_pr.T.dot(h) / r_pr[tau]
            Kalman[:, tau] = Kalman_tau.reshape(-1)
            R = 1 - (r_po/r_pr[tau])
            
            Wmean_po[:, tau] = Wmean_pr + Kalman[:, tau] * z
            P_po[:, :, tau] = P_pr - (Kalman_tau.dot(h.T).dot(P_pr) * R)
            
            tau += 1

        #SMOOTHING -- UPDATES GIVEN ALL TIMESTEPS
        #pi(\pi_t | all data up to time self.Tau)
        lambda_mean = np.zeros((self.nclasses, self.K))
        Lambda_cov = np.zeros((self.nclasses, self.nclasses, self.K))
        
        while tau > 0:
            tau-=1
            k = self.Cagents[tau]
            i = self.Cobjects[tau]
            h = self.E_t[i, :].reshape((self.nclasses, 1))
                        
            delta_Wmean = P_po[:, :, tau].dot(lambda_mean[:, k])  # P.T?
            delta_P = P_po[:, :, tau].dot(Lambda_cov[:, :, k]).dot(P_po[:, :, tau].T)  # both are P.T?
            
            Wmean_po[:,tau] = Wmean_po[:,tau] - delta_Wmean
            P_po[:,:,tau] = P_po[:,:,tau]-delta_P
            
            eta_po = h.T.dot(Wmean_po[:,tau])
            r_po = h.T.dot(P_po[:,:,tau]).dot(h)
            
            z = eta_po - eta_pr[tau]
            R = 1 - (r_po/r_pr[tau])
            B = I - Kalman[:,tau][:,np.newaxis].dot(h.T)
            
            lambda_mean[:,k] = B.T.dot(lambda_mean[:,k]) - h.reshape(-1)*z/r_pr[tau]
            Lambda_cov[:,:,k] = B.T.dot(Lambda_cov[:,:,k]).dot(B) + (h*R/r_pr[tau]).dot(h.T)
            
            pi_var = np.diag(P_po[:,:,tau])
            
            alpha_l, alphasum = state_to_alpha(Wmean_po[:,tau],pi_var)
            self.alpha[:, l, tau] = alpha_l
            if self.nclasses==2:
                self.alpha[:,1-l,tau] = alphasum - self.alpha[:,l,tau]
                
    def post_Alpha_binary_table(self, l=1):
        # l is the index into alpha we are dealing with
        # FILTERING -- UPDATES GIVEN PREVIOUS TIMESTEPS
        # p(\pi_t | data up to and including t)
        # Variables used in smoothing step but calculated during filtering
        Wmean_po = np.zeros((self.nclasses * self.K, self.Tau))
        P_po = {}  # np.zeros((self.nclasses * self.K, self.nclasses * self.K, self.N))
        Kalman = {}  # np.zeros((self.nclasses * self.K, self.K, self.N))
        # indicates which blocks of the kalman matrix are used
        # filtering variables
        q = np.zeros((1, self.K))
        I = np.tile(np.eye(self.nclasses)[np.newaxis, :, :], (self.K, 1, 1))
        I = block_diag(*I)
        eta_pr = np.zeros((self.Tau, self.K))
        r_pr = np.zeros((self.Tau, self.K))
        # loop through each row of the table in turn (each row is a timestep). Tau also corresponds to the object ID
        n = 0 # the data point index
        while n < self.N:
            logging.debug("Alpha update filter: %i / %i" % (n,self.N) )
            h_cov = self.h_cov(n)
            # initialise the state priors
            if n==0:
                Wmean_pr, P_pr = alpha_to_state(self.alpha0, l)
            else:
                Wmean_pr = Wmean_po[:, n-1]
                P_pr = P_po[n-1] + np.tile(q, (self.nclasses, 1)).flatten('F') * I
            # priors
            eta_pr[n, :] = h_cov.T.dot(Wmean_pr).reshape(-1)
            r_pr[n, :] = np.diag(h_cov.T.dot(P_pr).dot(h_cov)).reshape(-1) # possibly doing too much computation here?
            alpha_tilde_pr, alpha_tilde_pr_sum = state_to_alpha(eta_pr[n, :], r_pr[n, :])
            # replace any missing c values
            c = self.C[l][n,:]
            total_n = self.Cknown[n,:]
            # update to get posterior given current time-step
            alpha_tilde_po = alpha_tilde_pr + c
            alpha_tilde_po_sum = alpha_tilde_pr_sum + total_n
#             logging.debug("Alpha_tilde_po " + str(alpha_tilde_po) + ", tau=" + str(tau))
            # check r_po
            eta_po = np.log(alpha_tilde_po / alpha_tilde_po_sum)
            r_po = (1.0 / alpha_tilde_po) + (1.0 / alpha_tilde_po_sum)
            # update from this observation at tau
            z = eta_po - eta_pr[n, :]
            pi_tilde_pr = alpha_tilde_pr / alpha_tilde_pr_sum
            u_pr = pi_tilde_pr * (1 - pi_tilde_pr)
            pi_tilde_po = alpha_tilde_po / alpha_tilde_po_sum
            u_po = pi_tilde_po * (1 - pi_tilde_po)
            q[:] = (u_po > u_pr) * (u_po - u_pr)
            Kalman[n] = P_pr.T.dot(h_cov) / r_pr[n, :]  # JK x K # / np.tile(r_pr[tau, :], (self.nclasses, 1)).reshape((self.nclasses * self.K, 1), order='F')
            R = 1 - (r_po / r_pr[n, :])
            R = np.tile(R, (self.nclasses, 1)).flatten('F')
            Wmean_po[:, n] = Wmean_pr + np.sum(Kalman[n] * z, axis=1)  #* np.tile(z, (self.nclasses, 1)).reshape((self.nclasses * self.K, 1), order='F')).reshape(-1)  # kalman has too many columns !!!
            P_po[n] = P_pr - (Kalman[n].dot(h_cov.T).dot(P_pr) * R)
            # increment the timestep counter
            n += 1
        logging.debug("Completed " + str(n) + " filter steps.")
        # SMOOTHING -- UPDATES GIVEN ALL TIMESTEPS
        # pi(\pi_t | all data up to time self.N)
        lambda_mean = np.zeros((self.nclasses * self.K))
        Lambda_cov = np.zeros((self.nclasses * self.K, self.nclasses * self.K))
        while n > 0:
            n -= 1
            logging.debug("Alpha update smoother: " + str(n) + "/" + str(self.N))
            h_cov = self.h_cov(n)
            delta_Wmean = P_po[n].dot(lambda_mean)
            delta_P = P_po[n].dot(Lambda_cov).dot(P_po[n].T)
            Wmean_po[:, n] = Wmean_po[:, n] - delta_Wmean.reshape(-1)
            P_po[n] = P_po[n] - delta_P
            eta_po = h_cov.T.dot(Wmean_po[:, n])
            r_po = np.diag(h_cov.T.dot(P_po[n]).dot(h_cov))
            z = eta_po - eta_pr[n, :]
            z_rpr = z / r_pr[n, :]  # np.tile(z / r_pr[tau, :], (self.nclasses, 1)).reshape((self.nclasses * self.K, 1), order='F')
            R = 1 - (r_po / r_pr[n, :])
            B = I - Kalman[n].dot(h_cov.T)
            lambda_mean = B.T.dot(lambda_mean) - np.sum(h_cov * z_rpr, axis=1)
            Lambda_cov = B.T.dot(Lambda_cov).dot(B) + (h_cov * R / r_pr[n, :]).dot(h_cov.T)
            pi_var = np.diag(P_po[n]).reshape((self.nclasses, self.K), order='F')
            alpha_l, alphasum = state_to_alpha(Wmean_po[:, n].reshape((self.nclasses, self.K), order='F'), pi_var)
            if self.verbose:
                logging.debug("Alpha: " + str(alpha_l))
            agentIdx = np.arange(self.K)
            timeIdx = np.tile([n], (self.K))
            tauIdx = np.ravel_multi_index((timeIdx, agentIdx), (self.N, self.K))

            self.alpha[:, l, tauIdx] = alpha_l
            if self.nclasses == 2:
                self.alpha[:, 1 - l, tauIdx] = alphasum - self.alpha[:, l, tauIdx]
                
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def lnjoint(self, alldata=False):
        '''
        For use with crowdsourced data in table format (should be converted on input)
        '''
        if self.uselowerbound or alldata:
            for j in range(self.nclasses):
                data = np.zeros((self.N, self.K), dtype=float)
                for l in range(self.nscores):
                    if self.table_format_flag:
                        data += self.lnPi[j, l, :].reshape((self.N,self.K)) * self.C[l]
                    else:
                        data[self.Cobjects, self.Cagents] += self.lnPi[j, l, :] * self.C[l]
                self.lnpCT[:, j] = np.sum(data, 1) + self.lnkappa[j]
        else:  # no need to calculate in full - only do for test data points
            for j in range(self.nclasses):
                if self.table_format_flag:
                    data = np.zeros((self.Ntest, self.K), dtype=float)
                else:
                    data = np.zeros((self.N, self.K), dtype=float)
                for l in range(self.nscores):
                    if self.table_format_flag:
                        data += self.lnPi[j, l, self.tauidxs_test] * self.Ctest[l]
                    else:
                        data[self.Cobjects_test, self.Cagents_test] += self.lnPi[j, l, self.tauidxs_test] * self.Ctest[l]
                if not self.table_format_flag:
                    data = data[self.testidxs,:]
                self.lnpCT[self.testidxs, j] = np.sum(data, 1) + self.lnkappa[j]


    def post_lnpi(self):
        if self.alpha0_tau==[]:
            if self.table_format_flag:
                self.alpha0_tau = np.tile(self.alpha0, (1,1,self.N))
                self.piprior_const = np.sum(gammaln(np.sum(self.alpha0,1)) - np.sum(gammaln(self.alpha0),1)) * self.N               
            else:
                self.alpha0_tau = self.alpha0[:,:,self.Cagents]
                self.piprior_const = np.sum(gammaln(np.sum(self.alpha0_tau,1)) - np.sum(gammaln(self.alpha0_tau),1))
        return np.sum(np.sum((self.alpha0_tau-1)*self.lnPi,1)) + self.piprior_const

# Loader and Runner helper functions -------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    ibcc.load_and_run_ibcc(configFile, DynIBCC)
    
