'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.special import psi, gammaln
from scipy.sparse import coo_matrix
import ibcc
from scipy.linalg import block_diag

def state_to_alpha(logodds, var):
    alpha1 = 1/var * (1+np.exp(logodds))
    alpha2 = alpha1 * (1+np.exp(-logodds))
    return alpha1.reshape(logodds.shape), alpha2.reshape(logodds.shape)

def alpha_to_state(alpha, l):
    Wmean = np.log(alpha[:, l]) - np.log(np.sum(alpha, axis=1) - alpha[:, l])
    P = np.diagflat((1 / alpha[:, l]) + (1 / (np.sum(alpha, axis=1) - alpha[:, l])))
    return Wmean, P

def alpha_to_state3d(alpha, l):
    Wmean = np.log(alpha[:, l, :]) - np.log(np.sum(alpha, axis=1) - alpha[:, l, :])
    Wmean = Wmean.flatten('F')
    P = np.diagflat((1.0 / alpha[:, l, :]) + (1.0 / (np.sum(alpha, axis=1) - alpha[:, l, :])))
    return Wmean, P

class DynIBCC(ibcc.IBCC):
        
    samples_per_timestep = 1  # the number of data points that belong to a single timestep. Increase this if a regular
    # number of data points occur at the same time -- it assumes they have the same confusion matrix.
    sample_bins = []  # records the number of data points in each timestep in case some responses are missing
    Tau = 1 # the total number of responses from all crowd members
# Initialisation ---------------------------------------------------------------------------------------------------
    def __init__(self, nclasses=2, nscores=2, alpha0=None, nu0=None, K=1, table_format=False, dh=None, samples_per_timestep=1, sample_bins=None):
        super(DynIBCC, self).__init__(nclasses, nscores, alpha0, nu0, K, table_format, dh)
        self.samples_per_timestep = float(samples_per_timestep)  # don't need to set this if sample_bins is predefined
        self.sample_bins = sample_bins

    def init_lnPi(self):
        if self.alpha != []:
            return  # the basic initialisation was done when comine_classifications was called on a previous dataset
        self.alpha0 = self.alpha0.astype(float)
        if hasattr(self.alpha0, 'ndim') and self.alpha0.ndim == 3:
            # we've already done the basic initialisation
            return
        # usually we input a 2-D matrix that we'll copy for all timesteps
        self.alpha0 = np.array(self.alpha0)
        if len(self.alpha0.shape) < 3:
            self.alpha0 = self.alpha0[:, :, np.newaxis]
        sumAlpha = np.sum(self.alpha0, 1)
        psiSumAlpha = psi(sumAlpha)
        self.lnPi = psi(self.alpha0) - psiSumAlpha
# Data preprocessing and helper functions --------------------------------------------------------------------------
    def desparsify_crowdlabels(self, crowdlabels):
        if crowdlabels.shape[1] != 3:
            self.table_format_flag = True
            # First, record which objects were actually observed.
            self.observed_idxs = np.argwhere(np.nansum(crowdlabels, axis=1) >= 0).reshape(-1)
            self.full_N = crowdlabels.shape[0]
        else:
            self.observed_idxs, mappedidxs = np.unique(crowdlabels[:, 1], return_inverse=True)
            self.full_N = np.max(crowdlabels[:, 1])
        windows = np.arange(np.ceil(self.full_N / self.samples_per_timestep), dtype=np.int)
        windows = np.tile(windows, (self.samples_per_timestep, 1)).flatten('F')
        windows = windows[self.observed_idxs]
        self.windows = windows
        bincounts = np.bincount(windows)
        self.sample_bins = bincounts[windows]
        if self.full_N > len(self.observed_idxs):
            self.sparse = True
            # cut out the unobserved data points. We'll put them back in at the end of the classification procedure.
            if self.table_format_flag:
                crowdlabels = crowdlabels[self.observed_idxs, :]
            else:
                # map the IDs so we skip unobserved data points. We'll map back at the end of the classification procedure.
                crowdlabels[:, 1] = mappedidxs
        return crowdlabels

    def preprocess_data(self, crowdlabels, train_t=None, testidxs=None):
        '''
        When crowd labels are supplied in table format, we assume that the responses from members of the crowd were all
        submitted concurrently in the same order as the objects are listed in the table.
        '''
        super(DynIBCC, self).preprocess_data(crowdlabels, train_t, testidxs)
        # Sort out the confusion matrices. In the dynamic case, there is one for every crowdsourced label!
#         if self.table_format_flag and np.any(np.bitwise_or(np.isnan(self.crowdlabels), self.crowdlabels < 0)):  # best to work with a sparse list in the dynamic case
#             self.crowdlabels = self.crowdtable_to_sparselist(self.crowdlabels)
#             self.table_format_flag = False
        if self.table_format_flag:
            self.Tau = self.crowdlabels.shape[0] * self.crowdlabels.shape[1]
        else:
            self.Tau = self.crowdlabels.shape[0]
        # indexes for calculating the joint likelihood efficiently
        timeidxs = np.tile(np.arange(self.N).reshape((self.N, 1)), (1, self.K))
        timeidxs = timeidxs[self.test_crowdlabel_idxs].astype(int)
        self.tauidxs = np.ravel_multi_index((self.test_agent_idxs, timeidxs), (self.K, self.N))
        # Once we have the dataset to classify, we can determine the number of timesteps self.Tau, which we need to
        # expand lnPi to the correct number of matrices.
        if self.alpha != [] and self.alpha.shape[2] < self.Tau:
            # Case where we have already run this combiner with fewer timesteps
            oldTau = self.alpha0.shape[2]
            # Expand the prior
            ending = self.alpha0[:, :, oldTau - 1][:, :, np.newaxis]
            ending = np.repeat(ending, self.Tau - oldTau, axis=2)
            self.alpha0 = np.concatenate((self.alpha0, ending), axis=2)
            # Do the same for the posterior
            ending = self.alpha[:, :, oldTau - 1][:, :, np.newaxis]
            ending = np.repeat(ending, self.Tau - oldTau, axis=2)
            self.alpha = np.concatenate((self.alpha, ending), axis=2)
            # And the same for lnpi
            ending = self.lnPi[:, :, oldTau - 1][:, :, np.newaxis]
            ending = np.repeat(ending, self.Tau - oldTau, axis=2)
            self.lnPi = np.concatenate((self.lnPi, ending), axis=2)
        elif self.alpha != [] and self.alpha.shape[2] > self.Tau:
            # The new dataset has fewer items than before
            self.alpha0[:, :, 0:self.Tau]
            self.alpha[:, :, 0:self.Tau]
            self.lnPi[:, :, 0:self.Tau]
        elif self.alpha0.shape[2] > 1:
            # Haven't run the combiner yet, so need to expand from the first matrix
            # Find the number of new alpha0 needed. We can pass in multiple at the start, but if not enough, we'll
            # duplicate the first alpha0.
            nduplicates = self.Tau - self.alpha[:, :, 0]
            input_alpha0 = self.alpha0
            input_lnPi = self.lnPi
            self.alpha0 = np.repeat(self.alpha0[:, :, 0], nduplicates, axis=2)
            self.lnPi = np.repeat(self.lnPi[:, :, 0], nduplicates, axis=2)
            self.alpha0 = np.concatenate((input_alpha0, self.alpha0), axis=2)
            self.lnPi = np.concatenate((input_lnPi, self.lnPi), axis=2)
            # Initialise alpha to the same as alpha0
            self.alpha = deepcopy(self.alpha0)
        else:
            self.alpha0 = np.repeat(self.alpha0[:, :, 0:1], self.Tau, axis=2)
            self.lnPi = np.repeat(self.lnPi[:, :, 0:1], self.Tau, axis=2)
            # Initialise alpha to the same as alpha0
            self.alpha = deepcopy(self.alpha0)
# Posterior Updates to Hyperparameters -----------------------------------------------------------------------------
    def post_Alpha(self):#Posterior Hyperparams        
        if self.nclasses>2:
            for l in range(1,self.nscores):
                self.post_Alpha_binary(l)
        elif self.table_format_flag:
            self.post_Alpha_binary_3d(1)
        else:
            self.post_Alpha_binary(1)
    
    def post_Alpha_binary(self, l=1):
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
            if len(self.sample_bins) > 1:
                samples_per_timestep = self.sample_bins[tau]
            else:
                samples_per_timestep = self.samples_per_timestep
            k = self.crowdlabels[tau,0]
            i = self.crowdlabels[tau,1]
            c = np.sum(self.crowdlabels[tau:tau + samples_per_timestep, 2] == l, axis=0).astype(float)
            tau_pr = tau_prev[k]
            tau_prev[k] = tau
            h = self.E_t[i, :].reshape((self.nclasses, 1))  # column vector
            
            if tau_pr == -1:
                Wmean_pr, P_pr = alpha_to_state(self.alpha0[:, :, tau], l)  # vector of size nclasses, matrix nclasses^2
            else:
                Wmean_pr = Wmean_po[:,tau_pr]
                P_pr = P_po[:,:,tau_pr] + q[k]*I
            eta_pr[tau] = h.T.dot(Wmean_pr)
            r_pr[tau] = h.T.dot(P_pr).dot(h)
            alpha_tilde_pr,alpha_tilde_pr_sum = state_to_alpha(eta_pr[tau], r_pr[tau])
            #update to get posterior given current time-step
            alpha_tilde_po = alpha_tilde_pr + c
            alpha_tilde_po_sum = alpha_tilde_pr_sum + samples_per_timestep
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
            
            tau += samples_per_timestep

        #SMOOTHING -- UPDATES GIVEN ALL TIMESTEPS
        #pi(\pi_t | all data up to time self.Tau)
        lambda_mean = np.zeros((self.nclasses, self.K))
        Lambda_cov = np.zeros((self.nclasses, self.nclasses, self.K))
        
        while tau >= 0:
            if len(self.sample_bins) > 1:
                samples_per_timestep = self.sample_bins[tau]
            else:
                samples_per_timestep = self.samples_per_timestep
            k = self.crowdlabels[tau,0]
            i = self.crowdlabels[tau,1]
            h = self.E_t[i, :].reshape((self.nclasses, 1))
                        
            delta_Wmean = P_po[:, :, tau].dot(lambda_mean[:, k])  # P.T?
            delta_P = P_po[:, :, tau].dot(Lambda_cov[:, :, k]).dot(P_po[:, :, tau].T)  # both are P.T?
            
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
            self.alpha[:, l, tau] = alpha_l
            if self.nclasses==2:
                self.alpha[:,1-l,tau] = alphasum - self.alpha[:,l,tau]
            tau -= samples_per_timestep

    def h_cov(self, tau):
        h = self.E_t[tau, :].reshape((self.nclasses, 1))
            # the expanded vector to use with block-diagonal covariance
        return block_diag(*np.tile(h[np.newaxis, :, :], (self.K, 1, 1)))  # nclasses*K x K):
                
    def post_Alpha_binary_3d(self, l=1):
        # l is the index into alpha we are dealing with
        # FILTERING -- UPDATES GIVEN PREVIOUS TIMESTEPS
        # p(\pi_t | data up to and including t)
        # Variables used in smoothing step but calculated during filtering
        Wmean_po = np.zeros((self.nclasses * self.K, self.N))
        P_po = {}  # np.zeros((self.nclasses * self.K, self.nclasses * self.K, self.N))
        Kalman = {}  # np.zeros((self.nclasses * self.K, self.K, self.N))
        # indicates which blocks of the kalman matrix are used
        # filtering variables
        q = np.zeros((1, self.K))
        I = np.tile(np.eye(self.nclasses)[np.newaxis, :, :], (self.K, 1, 1))
        I = block_diag(*I)
        eta_pr = np.zeros((self.N, self.K))
        r_pr = np.zeros((self.N, self.K))
        # loop through each row of the table in turn (each row is a timestep). Tau also corresponds to the object ID
        tau = 0
        samples_per_timestep = 1
        niter = 0
        while tau < self.N:
            logging.debug("Alpha update filter: " + str(tau) + ", " + str(self.windows[tau]) + "/" + str(np.max(self.windows)))
            tau_pr = tau - samples_per_timestep
            if len(self.sample_bins) > 1:
                samples_per_timestep = self.sample_bins[tau]
            else:
                samples_per_timestep = self.samples_per_timestep
            h_cov = self.h_cov(tau)
            # initialise the state priors
            if tau_pr == -1:
                agentIdx = np.arange(self.K)
                timeIdx = np.tile([tau], (self.K))
                tauIdx = np.ravel_multi_index((agentIdx, timeIdx), (self.K, self.N))
                Wmean_pr, P_pr = alpha_to_state3d(self.alpha0[:, :, tauIdx], l)
            else:
                Wmean_pr = Wmean_po[:, tau_pr]
                P_pr = P_po[tau_pr] + np.tile(q, (self.nclasses, 1)).flatten('F') * I
            # priors
            eta_pr[tau, :] = h_cov.T.dot(Wmean_pr)
            r_pr[tau, :] = np.diag(h_cov.T.dot(P_pr).dot(h_cov))  # possibly doing too much computation here?
            alpha_tilde_pr, alpha_tilde_pr_sum = state_to_alpha(eta_pr[tau, :], r_pr[tau, :])
            # replace any missing c values
            missingidxs = np.bitwise_or(np.isnan(self.crowdlabels[tau, :]), self.crowdlabels[tau, :] < 0)
            c = np.sum(self.crowdlabels[tau:tau + samples_per_timestep, :] == l, axis=0).astype(float)
            c[missingidxs] = alpha_tilde_pr[missingidxs] / (alpha_tilde_pr_sum[missingidxs]).astype(float)
            # update to get posterior given current time-step
            alpha_tilde_po = alpha_tilde_pr + c
            alpha_tilde_po_sum = alpha_tilde_pr_sum + samples_per_timestep
#             logging.debug("Alpha_tilde_po " + str(alpha_tilde_po) + ", tau=" + str(tau))
            # check r_po
            eta_po = np.log(alpha_tilde_po / alpha_tilde_po_sum)
            r_po = (1.0 / alpha_tilde_po) + (1.0 / alpha_tilde_po_sum)
            # update from this observation at tau
            z = eta_po - eta_pr[tau, :]
            pi_tilde_pr = alpha_tilde_pr / alpha_tilde_pr_sum
            u_pr = pi_tilde_pr * (1 - pi_tilde_pr)
            pi_tilde_po = alpha_tilde_po / alpha_tilde_po_sum
            u_po = pi_tilde_po * (1 - pi_tilde_po)
            q[:] = (u_po > u_pr) * (u_po - u_pr)
            Kalman[tau] = P_pr.T.dot(h_cov) / r_pr[tau, :]  # JK x K # / np.tile(r_pr[tau, :], (self.nclasses, 1)).reshape((self.nclasses * self.K, 1), order='F')
            R = 1 - (r_po / r_pr[tau, :])
            R = np.tile(R, (self.nclasses, 1)).flatten('F')
            Wmean_po[:, tau] = Wmean_pr + np.sum(Kalman[tau] * z, axis=1)  #* np.tile(z, (self.nclasses, 1)).reshape((self.nclasses * self.K, 1), order='F')).reshape(-1)  # kalman has too many columns !!!
            P_po[tau] = P_pr - (Kalman[tau].dot(h_cov.T).dot(P_pr) * R)
            # increment the timestep counter
            tau += samples_per_timestep
            niter += 1
        logging.debug("Completed " + str(niter) + " filter steps.")
        # SMOOTHING -- UPDATES GIVEN ALL TIMESTEPS
        # pi(\pi_t | all data up to time self.N)
        lambda_mean = np.zeros((self.nclasses * self.K))
        Lambda_cov = np.zeros((self.nclasses * self.K, self.nclasses * self.K))
        while tau > 0:
            tau -= samples_per_timestep
            if len(self.sample_bins) > 1:
                samples_per_timestep = self.sample_bins[tau - 1]  # need to get the gap from the previous tau
            else:
                samples_per_timestep = self.samples_per_timestep
            logging.debug("Alpha update smoother: " + str(tau) + ", " + str(self.windows[tau]) + "/" + str(np.max(self.windows)))
            h_cov = self.h_cov(tau)
            delta_Wmean = P_po[tau].dot(lambda_mean)
            delta_P = P_po[tau].dot(Lambda_cov).dot(P_po[tau].T)
            Wmean_po[:, tau] = Wmean_po[:, tau] - delta_Wmean.reshape(-1)
            P_po[tau] = P_po[tau] - delta_P
            eta_po = h_cov.T.dot(Wmean_po[:, tau])
            r_po = np.diag(h_cov.T.dot(P_po[tau]).dot(h_cov))
            z = eta_po - eta_pr[tau, :]
            z_rpr = z / r_pr[tau, :]  # np.tile(z / r_pr[tau, :], (self.nclasses, 1)).reshape((self.nclasses * self.K, 1), order='F')
            R = 1 - (r_po / r_pr[tau, :])
            B = I - Kalman[tau].dot(h_cov.T)
            lambda_mean = B.T.dot(lambda_mean) - np.sum(h_cov * z_rpr, axis=1)
            Lambda_cov = B.T.dot(Lambda_cov).dot(B) + (h_cov * R / r_pr[tau, :]).dot(h_cov.T)
            pi_var = np.diag(P_po[tau]).reshape((self.nclasses, self.K), order='F')
            alpha_l, alphasum = state_to_alpha(Wmean_po[:, tau].reshape((self.nclasses, self.K), order='F'), pi_var)
#             logging.debug("Alpha: " + str(alpha_l))
            agentIdx = np.arange(self.K)
            timeIdx = np.tile([tau], (self.K))
            tauIdx = np.ravel_multi_index((agentIdx, timeIdx), (self.K, self.N))

            self.alpha[:, l, tauIdx] = alpha_l
            if self.nclasses == 2:
                self.alpha[:, 1 - l, tauIdx] = alphasum - self.alpha[:, l, tauIdx]
# Expectations: methods for calculating expectations with respect to parameters for the VB algorithm ---------------
    def expec_lnPi(self):
        self.post_Alpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for s in range(self.nscores):        
            self.lnPi[:,s,:] = psi(self.alpha[:,s,:]) - psiSumAlpha
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def lnjoint_table(self, alldata=False):
        if self.uselowerbound or alldata:
            idxs = np.ones(self.N, dtype=np.bool)
        else:  # no need to calculate in full
            idxs = self.testidxs    
        for j in range(self.nclasses):
            data = self.lnPi[j, self.test_crowd_labels, self.tauidxs]
            self.lnPi_table[self.test_crowdlabel_idxs] = data
            self.lnpCT[idxs, j] = np.sum(self.lnPi_table, 1) + self.lnkappa[j]            

    def lnjoint_sparselist(self):
        if self.conf_mat_ind == []:
            crowdlabels = self.crowdlabels.astype(int)
            logging.info("Using only discrete labels at the moment.")
            self.conf_mat_ind = np.ravel_multi_index((crowdlabels[:, 2], np.arange(self.Tau)), dims=(self.nscores, self.Tau))
        for j in range(self.nclasses):
            weights = self.lnPi[j, :].ravel()[self.conf_mat_ind]
            self.lnpCT[:, j] = np.bincount(self.crowdlabels[:, 1], weights=weights, minlength=self.N) + self.lnkappa[j]

# Loader and Runner helper functions -------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'
    ibcc.load_and_run_ibcc(configFile, DynIBCC)
    
