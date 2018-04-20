"""Run the experiment called Supervised[MAR] in the paper.

INPUT: .pickle files with the r2 score, mse score and granger index of
both the MAR and NN datasets.

OUTPUT: .mat file with the prediction results, saved in the folder
/results/
"""

import os
import numpy as np
import pickle
from scipy.io import loadmat
from scipy.io import savemat
from scipy.misc import comb
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sys import stdout
from joblib import Parallel, delayed
import pdb

from score_function import compute_score_matrix, best_decision
from create_level2_dataset import feature_engineering, feature_normalisation

      
if __name__=='__main__':
    
    # parameters used to identify which files need to be loaded
    time_window_size_train = 10
    order_granger_train = 10    
    n_folds = 5 
    synEff = 18 
    
    n_jobs = -1 # '-1' = use all available CPU cores    
    ###############################################
    #Training data: MAR dataset
    pwd_source_train = '%s/results/' % os.getcwd()
    
    #granger index
    filename_level2_granger = '%sdataset_level2_tws%d_cv%d_granger_norm.pickle' % (pwd_source_train, time_window_size_train, n_folds)
    print "Loading", filename_level2_granger
    level2 = pickle.load(open(filename_level2_granger))
    X_train_level2_granger = level2['X_level2']
    
    #r2 
    filename_level2 = '%sdataset_level2_tws%d_cv%d_r2_shift_window_norm.pickle' % (pwd_source_train, time_window_size_train, n_folds)        
    print "Loading", filename_level2
    level2 = pickle.load(open(filename_level2))
    X_train_level2 = level2['X_level2']
    y_train_level2 = np.squeeze(level2['y_level2'])
    gamma_train_level2 = level2['gamma_level2']

    #mse
    filename_level2_mse = '%sdataset_level2_tws%d_cv%d_mse_shift_window_norm.pickle' % (pwd_source_train, time_window_size_train, n_folds)
    print "Loading", filename_level2_mse
    level2_mse = pickle.load(open(filename_level2_mse))
    X_train_level2_mse = level2_mse['X_level2'] * (-1)
    y_train_level2_mse = np.squeeze(level2_mse['y_level2'])
    gamma_train_level2_mse = level2_mse['gamma_level2']
    assert((y_train_level2_mse == y_train_level2).all())#sanity check
    
    #Test data: NN dataset
    #parameters used to identify files
    time_window_size_test=3
    order_granger_test=5
    pwd_source = '%s/results/' % os.getcwd()
    
    filename_level2_r2 = '%sdataset_level2_tws%d_lag4_cv%d_r2_shift_window_25000trial-RNDMax0.%dSynEff_perTimeNorm_25conf.pickle' % (pwd_source, time_window_size_test, n_folds, synEff)     
    filename_level2 = filename_level2_r2
    print "Loading", filename_level2
    level2 = pickle.load(open(filename_level2))
    X_level2 = level2['X_level2']
    y_level2_conf = level2['y_level2_conf'] #the entire configuration matrix nCh x nCh
    y_level2 = np.squeeze(level2['y_level2'])
    gamma_level2 = level2['gamma_level2']
    order_combinations = level2['order_combinations']
        
    filename_level2_mse = '%sdataset_level2_tws%d_lag4_cv%d_mse_shift_window_25000trial-RNDMax0.%dSynEff_perTimeNorm_25conf.pickle' % (pwd_source, time_window_size_test, n_folds, synEff)     
    filename_level2 = filename_level2_mse
    print "Loading", filename_level2
    level2 = pickle.load(open(filename_level2))
    X_level2_mse = level2['X_level2'] * (-1)
    y_level2_conf_mse = level2['y_level2_conf']
    y_level2_mse = np.squeeze(level2['y_level2'])
    gamma_level2_mse = level2['gamma_level2']
    order_combinations = level2['order_combinations']
    
    #sanity checks
    assert((y_level2_conf == y_level2_conf_mse).all())    
    assert((y_level2 == y_level2_mse).all())    
    
    filename_level2_granger = '%sdataset_level2_tws%d_cv%d_granger_25000trial-RNDMax0.%dSynEff_perTimeNorm_25conf.pickle' % (pwd_source, order_granger_test, n_folds, synEff)           
    print "Loading", filename_level2_granger
    level2 = pickle.load(open(filename_level2_granger))
    X_level2_granger = level2['X_level2']

    print
    print "Doing classification."    
    from create_trainset import class_to_configuration
    from sklearn.linear_model import LogisticRegression
    from itertools import izip
    from sklearn.lda import LDA

    clf = LogisticRegression(C=1.0, penalty='l2', random_state=0)    
    print clf
    
    gamma_threshold = 1.0
    gamma_threshold_flag = False
    print "Keeping all training examples with gamma < %s" % gamma_threshold
    nTrial_train = y_train_level2.shape[0]    
    if not gamma_threshold_flag:
        idx = np.ones(nTrial_train,dtype=np.bool)
    else:
        idx = gamma_train_level2 <= gamma_threshold
    
    #Class filtering
    reduced = True
    if reduced:
        dag_conf = np.unique(y_level2) #select the same classes of the test set
        y_train_reduced = np.zeros(y_train_level2.shape[0],dtype=bool)
        for i_conf, conf_i in enumerate(dag_conf):
            y_train_reduced = np.logical_or(y_train_reduced,y_train_level2==conf_i)
        idx = np.logical_and(idx,y_train_reduced)
    
    #Feature filtering
    reduce_feat = False
    if reduce_feat:
        feat_sel = np.array([1,2,3,5,6,7,11,13,15])
        X_train_level2 = X_train_level2[:,feat_sel]
        X_train_level2_mse = X_train_level2_mse[:,feat_sel]
        
    X_train = X_train_level2[idx]
    y_train = y_train_level2[idx]
    X_train_granger = X_train_level2_granger[idx] 
    X_train_mse = X_train_level2_mse[idx]
    del X_train_level2, y_train_level2, X_train_level2_granger, X_train_level2_mse

    print "Preprocessing Train set"
    block_normalisation = False
    feature_space_train = feature_engineering([X_train, X_train_granger, X_train_mse], block_normalisation=block_normalisation) 
    del X_train, X_train_granger, X_train_mse

    block_normalisation = False
    X_train = feature_normalisation(feature_space_train, block_normalisation=block_normalisation)
    
    #Test set
    gamma_threshold = -1
    gamma_threshold_flag = False
    print "Keeping all testing examples with gamma >= %s" % gamma_threshold
    if not gamma_threshold_flag:
        idx = range(X_level2.shape[0])
    else:
        idx = gamma_level2 >= gamma_threshold
    
    #Feature filtering
    if reduce_feat:
        feat_sel = np.array([1,2,3,5,6,7,11,13,15])
        X_level2 = X_level2[:,feat_sel]
        X_level2_mse = X_level2_mse[:,feat_sel]
    
    X = X_level2[idx]
    y = y_level2[idx]
    y_level2_conf = y_level2_conf[idx]
    X_granger = X_level2_granger[idx] 
    X_mse = X_level2_mse[idx]
    
    del X_level2, X_level2_mse, X_level2_granger, gamma_level2, gamma_level2_mse, y_level2, y_level2_mse, y_level2_conf_mse
    
    if len(X.shape==3):
        print "Reshape to have an array of triads"
        [nTrial, nComb, nFeat] = X.shape
        X = np.reshape(X, [nTrial*nComb, nFeat])
        X_mse = np.reshape(X_mse, [nTrial*nComb, nFeat])
        [nTrial, nComb, nFeat] = X_granger.shape
        X_granger = np.reshape(X_granger, [nTrial*nComb, nFeat])
        y = np.reshape(y, [nTrial*nComb])
        y = np.array(y, dtype=int)  
    
    print "Preprocessing Test set"
    block_normalisation = False
    feature_space = feature_engineering([X, X_granger, X_mse], block_normalisation=block_normalisation)
    del X, X_granger, X_mse    
    
    block_normalisation = False
    X = feature_normalisation(feature_space, block_normalisation=block_normalisation)
    ###############
    
    print "X:", X.shape
    print "Computing score."
    nComb = order_combinations.shape[0]
    cv_train = StratifiedKFold(y_train, n_folds=n_folds)
    cv_test = StratifiedKFold(y, n_folds=n_folds)
    cv = izip(cv_train, cv_test)

    optimal_decision = True
    binary_class = True

    if binary_class:
            
        score_matrix = np.array([[1,0],[0,1]], dtype=int)#np.array([[0,-3],[0,1]], dtype=int) #score assigned to false/true positive and false/true negative
        y_new = [] #conversion from int representation of the class to binary matrix representation
        y_new += [(class_to_configuration(y_i, verbose=False)) for y_i in np.squeeze(y_train)] 
        y_new = np.array(y_new, dtype=int)
        y_pred = np.zeros(y_new.shape, dtype=int)
        predicted_probability = np.zeros((X.shape[0], 6, 2))
        
        for i, (train, test) in enumerate(cv):
            print "Fold %d" % i
            train = train[0] #train part of the train set 
            test = test[1] #test part of the test set
            
            index_x = np.append(np.triu_indices(3,1)[0], np.tril_indices(3,-1)[0])
            index_y = np.append(np.triu_indices(3,1)[1], np.tril_indices(3,-1)[1])
                        
            for j in range(len(index_x)):
                print "Train."   
                clf.fit(X_train[train], y_new[train,index_x[j],index_y[j]])
                y_pred_proba = clf.predict_proba(X[test])
                predicted_probability[test,j] = y_pred_proba.copy()
                if optimal_decision:
                    print "Predict."
                    y_pred[test,index_x[j],index_y[j]] = np.array([best_decision(prob_configuration, score_matrix=score_matrix)[0] for prob_configuration in y_pred_proba])
                else:
                    print "Predict."
                    y_pred[test,index_x[j],index_y[j]] = clf.predict(X[test])
        
        #combining predictions of each subset of 3 time series if nCh>3
        #since each cell appears in different subsets, y_pred_conf counts how many time a connection is predicted for each cell  
        nTrial, nCh = y_level2_conf.shape[:2]
        y_pred_conf = np.zeros((nTrial, nCh, nCh))
        for i_trial in range(nTrial):
            for i_comb in range(order_combinations.shape[0]):
                y_pred_conf[i_trial][order_combinations[i_comb][:,None], order_combinations[i_comb]] += y_pred[i_trial*nComb+i_comb]

    else: 
        
        classes = np.unique(y)
        y_pred = np.zeros(y.shape, dtype=int)
        score_matrix = compute_score_matrix()
        predicted_probability = np.zeros((X.shape[0], len(classes)))
        
        for i, (train, test) in enumerate(cv):
            print "Fold %d" % i            
            train = train[0] #train part of the train set 800 trials per class
            test = test[1] #test part of the test set, 200 trials per class
            print "Train."        
            clf.fit(X_train[train], y_train[train])
            y_pred_proba = clf.predict_proba(X[test])
            predicted_probability[test] = y_pred_proba.copy()
            if optimal_decision:
                print "Predict."
                y_pred[test] = np.array([best_decision(prob_configuration, score_matrix=score_matrix)[0] for prob_configuration in y_pred_proba])
            else:
                print "Predict."
                y_pred[test] = clf.predict(X[test])
        
        nTrial, nCh = y_level2_conf.shape[:2]
        y_pred_conf = np.zeros((nTrial, nCh, nCh))
        for i_trial in range(nTrial):
            for i_comb in range(order_combinations.shape[0]):
                y_pred_conf[i_trial][order_combinations[i_comb][:,None], order_combinations[i_comb]] += class_to_configuration(y_pred[i_trial*nComb+i_comb])
    
    
    print "Set zero the diagonal"
    for i_trial in range(nTrial):
        y_pred_conf[i_trial][np.diag_indices(nCh)] = 0
        y_level2_conf[i_trial][np.diag_indices(nCh)] = 0
    
    pwd_dest = '%s/results/' % os.getcwd()
    filename_save = '%ssimulatedMEG_25conf-RNDMax0.%dSynEff_tws%d_lag4_conf_r2_granger_mse_perTimeNorm_optimal_decision_binary_reduced_Ldataset_fEngNormRow_cv_testFeb.mat' % (pwd_dest, synEff, time_window_size_test)   
    print "Saving %s" % filename_save
    if optimal_decision:
        savemat(filename_save, {'y_test_pred': y_pred_conf,
                                'y_test_true': y_level2_conf,
                                'predicted_probability': predicted_probability,
                                'score_matrix': score_matrix
                            })  
    else:
        savemat(filename_save, {'y_test_pred': y_pred_conf,
                                'y_test_true': y_level2_conf,
                                'predicted_probability': predicted_probability
                            })   
