"""Compute feature vectors of multivariate timeseries.

INPUT: .mat file containing the trials, expected to be in the folder
/data/.

OUTPUT: three .pickle files with the r2 score, the mne score and the
granger index, saved in the folder /results/.
"""

import numpy as np
import pickle
from scipy.io import loadmat
from scipy.misc import comb
from sklearn.linear_model import LinearRegression, BayesianRidge, LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sys import stdout
from joblib import Parallel, delayed

from create_level2_dataset import regression_scores, granger_scores, configuration_to_class
   
causality_structures = [((0,),0), ((0,),1), ((0,),2),
                        ((1,),0), ((1,),1), ((1,),2),
                        ((2,),0), ((2,),1), ((2,),2),
                        ((0,1),0), ((0,1),1), ((0,1),2),
                        ((0,2),0), ((0,2),1), ((0,2),2),
                        ((1,2),0), ((1,2),1), ((1,2),2),
                        ((0,1,2),0), ((0,1,2),1), ((0,1,2),2)]

def compute_lev2_regression_general_case(filedata, time_window_size=10, time_lag=None, N=None, reg=None, cv=5, scoring='r2', n_jobs=-1):
    """Compute the regression scores for each trial in filedata.
    """
    print "Computing regression-based features."    
    data = loadmat(filedata, squeeze_me=True)
    #first 100 time points are discarded since not stable
    data_timeseries = data['data'][:,100:,:] #data dimensions are expected to be in this order: [trials, time, channel]   
    [nTrial, nTime, nCh] = data_timeseries.shape
    
    if not(N is None):
        data_timeseries = data_timeseries[:,:N,:]
        nTime = N

    ch = np.arange(nCh)
    y_level2_conf = data['conf'] 
    gamma_test_level2 = data['synpaticEfficacies']#None in case we don't know
    
    print "Data set shape:", data_timeseries.shape
    n_comb = comb(nCh,3,exact=1)
    X_lev2_regression = np.zeros((nTrial, n_comb, len(causality_structures), 2), dtype=np.float64) #added 2 dimensions to compute r2 and mse
    y_level2 = np.zeros((nTrial, n_comb), dtype=np.int32)
    order_combinations = np.zeros((n_comb, 3), dtype=np.int32)
    i_comb = 0
    for i in range(0,nCh):
        for j in range(i+1,nCh):
            for z in range(j+1,nCh):
                print "combination", i_comb
                conditionCh = np.delete(ch,[i,j,z])
                #uncomment the following line and update timeseriesZ when passed to regression_score function, if nCh>3 the problem is faced by considering all possible combinations of 3 time series 
                #timeseriesZ=data_timeseries[trial_i,:,conditionCh].T
                result = Parallel(n_jobs=n_jobs)(delayed(regression_scores)(data_timeseries[trial_i,:,[i,j,z]].T, time_window_size=time_window_size, time_lag=time_lag, reg=reg, cv=n_folds, scoring=scoring, timeseriesZ=None) for trial_i in range(nTrial))
                X_lev2_tmp = zip(*result) # See http://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python
                X_lev2_regression[:,i_comb,:,0] = np.squeeze(np.array(X_lev2_tmp)[:,:,0]).T #r2score
                X_lev2_regression[:,i_comb,:,1] = np.squeeze(np.array(X_lev2_tmp)[:,:,1]).T #mse               
                order_combinations[i_comb] = np.array([i,j,z], dtype=int)
                #conversion of the class from binary matrix to int in [0,63]
                tmp = []
                tmp += [(y_level2_conf[i_trial][order_combinations[i_comb][:,None], order_combinations[i_comb]]) for i_trial in range(nTrial)]                 
                tmp = np.array(tmp)
                tmp_res = []
                tmp_res += [(configuration_to_class(tmp[i_trial], verbose=False)) for i_trial in range(nTrial)]
                y_level2[:,i_comb] = np.array(tmp_res) 
                i_comb += 1
    
    y_level2_conf = np.array(y_level2_conf, dtype=np.int32)
    gamma_test_level2 = np.array(gamma_test_level2, dtype=np.float32)
    return X_lev2_regression, y_level2_conf, y_level2, gamma_test_level2, order_combinations
    

def compute_lev2_granger_general_case(filedata, order=10, N=None, n_jobs=-1):
    """Compute the granger causality coefficients for each triad in the entire set
    """
    print "Computing Granger causality coefficients"
    data = loadmat(filedata, squeeze_me=True)
    data_timeseries = data['data'][:,100:,:]
    [nTrial, nTime, nCh] = data_timeseries.shape

    if not(N is None):
        data_timeseries = data_timeseries[:,:N,:]
        nTime = N
    
    y_level2_conf = data['conf'] 
    gamma_test_level2 = data['synpaticEfficacies']#None in case we don't know
    
    print "Data set shape:", data_timeseries.shape
    n_comb = comb(nCh,3,exact=1)
    X_lev2_granger = np.zeros((nTrial, n_comb, 6))
    order_combinations = np.zeros((n_comb, 3), dtype=int)
    i_comb = 0
    for i in range(0,nCh):
        for j in range(i+1,nCh):
            for z in range(j+1,nCh):
                print "combination", i_comb
                result = Parallel(n_jobs=n_jobs)(delayed(granger_scores)(data_timeseries[trial_i,:,[i,j,z]], order) for trial_i in range(nTrial))
                X_lev2_tmp = zip(*result) # See http://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python
                X_lev2_granger[:,i_comb,:] = np.vstack(X_lev2_tmp).T
                order_combinations[i_comb] = np.array([i,j,z], dtype=int)
                i_comb += 1
    
    return X_lev2_granger, y_level2_conf, gamma_test_level2, order_combinations


if __name__ == '__main__':

    time_window_size_array = [3]
    time_lag = 4  # time_lag is where the lower bound of time_window_size is placed
    # if t* is the current time point, t*-time_lag <= t <= t*-time_lag+time_window_size 
    N = None #n. of evaluated time points, if None the entire time series

    reg = LinearRegression(fit_intercept=True, normalize=True)

    n_folds = 5 # cross-validation folds
    scoring = 'r2' # 'mean_squared_error': computed togheter with r2, 'residual_tests': tests on the whiteness of the residuals
    n_jobs = -1 # '-1' = use all available CPU cores
    
    import os
    pwd = '%s/results/' % os.getcwd()
    for time_window_size in time_window_size_array:
        print time_window_size
        filename_level2_r2 = '%sdataset_level2_tws%d_lag%d_cv%d_r2_shift_window_25000trial_perTimeNorm_25conf.pickle' % (pwd, time_window_size, time_lag, n_folds)       
        filename_level2_mse = '%sdataset_level2_tws%d_lag%d_cv%d_mse_shift_window_25000trial_perTimeNorm_25conf.pickle' % (pwd, time_window_size, time_lag, n_folds)     
     
        filedata_list = ['netsIndependentConnection1000_perTimeNorm_1class.mat',
                         'netsUnivariateConnection6000-RNDMax0.18SynEff_perTimeNorm_6classes.mat',
                         'netsDoubleUnivConnection3000-RNDMax0.18SynEff_perTimeNorm_3classes.mat',
                         'netsChainConnection6000-RNDMax0.18SynEff_perTimeNorm_6classes.mat',
                         'netsBivariateConnection3000-RNDMax0.18SynEff_perTimeNorm_3classes.mat',
                         'netsTrivariateConnection6000-RNDMax0.18SynEff_perTimeNorm_6classes.mat']
        
        #Test r2 and mse
        filename_level2 = filename_level2_r2
        #filename_level2 = filename_level2_mse
        try:
            print "Loading", filename_level2
            level2 = pickle.load(open(filename_level2))
            X_test_level2 = level2['X_level2']
            y_test_level2 = level2['y_level2']
            gamma_test_level2 = level2['gamma_level2']
            order_combinations = level2['order_combinations']
            
        except IOError:
            print "Not found!"
            print

            pwd_source = '%s/data/' % os.getcwd()
            X_test_level2 = []
            y_test_level2 = []
            y_level2 = []
            gamma_test_level2 = []
            
            n_file = np.shape(filedata_list)[0]
            for i_file in range(n_file):
            
                print "n. file:", i_file
                filedata = '%s%s' % (pwd_source,filedata_list[i_file])
                tmp_X_test_level2, tmp_y_test_level2, tmp_y_level2, tmp_gamma_test_level2, order_combinations = compute_lev2_regression_general_case(filedata, time_window_size, time_lag, N, reg, cv=n_folds, scoring=scoring, n_jobs=n_jobs)          
                X_test_level2.append(tmp_X_test_level2) 
                y_test_level2.append(tmp_y_test_level2)
                y_level2.append(tmp_y_level2)
                gamma_test_level2.append(tmp_gamma_test_level2)
            
            X_test_level2 = np.vstack(X_test_level2)
            y_test_level2 = np.vstack(y_test_level2)
            y_level2 = np.vstack(y_level2)
            print
            print "Saving level2 dataset in", filename_level2
            pickle.dump({'time_window_size': time_window_size,
                         'reg': reg,
                         'cv': n_folds,
                         'X_level2': np.squeeze(X_test_level2[:,:,:,0]),
                         'y_level2_conf': y_test_level2,
                         'y_level2': y_level2,
                         'gamma_level2': gamma_test_level2,
                         'order_combinations': order_combinations,
                         },
                        open(filename_level2, 'w'),
                        protocol = pickle.HIGHEST_PROTOCOL)
            
            filename_level2 = filename_level2_mse
            print
            print "Saving level2 dataset in", filename_level2
            pickle.dump({'time_window_size': time_window_size,
                         'reg': reg,
                         'cv': n_folds,
                         'X_level2': np.squeeze(X_test_level2[:,:,:,1]),
                         'y_level2_conf': y_test_level2,
                         'y_level2': y_level2,
                         'gamma_level2': gamma_test_level2,
                         'order_combinations': order_combinations,
                         },
                        open(filename_level2, 'w'),
                        protocol = pickle.HIGHEST_PROTOCOL)    
        
        #Test granger   
        order_granger = 5 
        filename_level2_granger = '%sdataset_level2_tws%d_cv%d_granger_25000trial_perTimeNorm_25conf.pickle' % (pwd, order_granger, n_folds)
        
        try:
            print "Loading", filename_level2_granger
            level2 = pickle.load(open(filename_level2_granger))
            X_test_level2_granger = level2['X_level2']
            
        except IOError:
            print "Not found!"
            print
            pwd_source = '../data/'
            X_test_level2_granger = []
            y_test_level2 = []
            gamma_test_level2 = []
            
            n_file = np.shape(filedata_list)[0]
            for i_file in range(n_file):
            
                print "n. file:", i_file        
                filedata = '%s%s' %(pwd_source,filedata_list[i_file])
                tmp_X_test_level2_granger, tmp_y_test_level2, tmp_gamma_test_level2, order_combinations = compute_lev2_granger_general_case(filedata, order_granger, N, n_jobs)
                X_test_level2_granger.append(tmp_X_test_level2_granger) 
                y_test_level2.append(tmp_y_test_level2)
                gamma_test_level2.append(tmp_gamma_test_level2) 
                
            X_test_level2_granger = np.vstack(X_test_level2_granger)
            y_test_level2 = np.vstack(y_test_level2)
            #gamma_test_level2 = np.hstack(gamma_test_level2)    
            print
            print "Saving level2 dataset in", filename_level2_granger
            pickle.dump({'time_window_size': time_window_size,
                         #'reg': reg,
                         #'cv': n_folds,
                         #'X_level2': X_test_level2_r2,
                         'order_granger': order_granger,
                         'X_level2': np.squeeze(X_test_level2_granger),
                         'y_level2_conf': y_test_level2,
                         'gamma_level2': gamma_test_level2,
                         'order_combinations': order_combinations,
                         },
                        open(filename_level2_granger, 'w'),
                        protocol = pickle.HIGHEST_PROTOCOL)
        
