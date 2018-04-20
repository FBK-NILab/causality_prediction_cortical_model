"""Run the experiment called Supervised[NN] in the paper

INPUT: .pickle files with the r2 score, mse score and granger index.

OUTPUT: .mat file with the prediction results, saved in the folder
/results/
"""

import numpy as np
import pickle
import os
from scipy.io import savemat
from scipy.misc import comb
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sys import stdout
from joblib import Parallel, delayed

from score_function import compute_score_matrix, best_decision
from create_level2_dataset import feature_engineering, feature_normalisation

        
if __name__=='__main__':

    time_window_size = 3
    synEff = 18
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 17, 20, 24, 28, 32, 33, 34, 35, 40, 48, 49, 56];

    n_folds = 5
    order_granger = 5
    n_jobs = -1 # '-1' = use all available CPU cores
      
    ###############################################
    # Set r2 and mse
    pwd_source = '%s/results/' % os.getcwd()  
    
    filename_level2_r2 = '%sdataset_level2_tws%d_lag4_cv%d_r2_shift_window_25000trial-RNDMax0.%dSynEff_perTimeNorm_25conf.pickle' % (pwd_source, time_window_size, n_folds, synEff)     
    filename_level2 = filename_level2_r2
    print "Loading", filename_level2
    level2 = pickle.load(open(filename_level2))
    X_level2 = level2['X_level2']
    y_level2_conf = level2['y_level2_conf'] #the entire configuration matrix nCh x nCh
    y_level2 = np.squeeze(level2['y_level2'])
    gamma_level2 = np.squeeze(np.hstack(level2['gamma_level2']))
    order_combinations = level2['order_combinations']
        
    filename_level2_mse = '%sdataset_level2_tws%d_lag4_cv%d_mse_shift_window_25000trial-RNDMax0.%dSynEff_perTimeNorm_25conf.pickle' % (pwd_source, time_window_size, n_folds, synEff)     
    filename_level2 = filename_level2_mse
    print "Loading", filename_level2
    level2 = pickle.load(open(filename_level2))
    X_level2_mse = level2['X_level2'] * (-1)
    y_level2_conf_mse = level2['y_level2_conf']
    y_level2_mse = np.squeeze(level2['y_level2'])
    gamma_level2_mse = np.squeeze(np.hstack(level2['gamma_level2']))
    order_combinations = level2['order_combinations']
    
    assert((y_level2_conf == y_level2_conf_mse).all())    
    assert((y_level2 == y_level2_mse).all())    
    assert((gamma_level2_mse == gamma_level2).all())

    filename_level2_granger = '%sdataset_level2_tws%d_cv%d_granger_25000trial-RNDMax0.%dSynEff_perTimeNorm_25conf.pickle' % (pwd_source, order_granger, n_folds, synEff)           
    print "Loading", filename_level2_granger
    level2 = pickle.load(open(filename_level2_granger))
    X_level2_granger = level2['X_level2']
    
    print
    print "Doing classification."
    from create_trainset import class_to_configuration
    from sklearn.linear_model import LogisticRegression
    from sklearn.lda import LDA
    clf = LogisticRegression(C=1.0, penalty='l2', random_state=0)
    #clf = LDA()
    print clf
    
    gamma_threshold_flag = False
    if gamma_threshold_flag:
        gamma_threshold = 1.0
        print "Keeping all training examples with gamma <= %s" % gamma_threshold
        #idx = np.ones(X_level2.shape[0],dtype=np.bool)
        idx = gamma_level2 <= gamma_threshold, gamma_level2
        X_level2 = X_level2[idx]
        y_level2 = y_level2[idx]
        y_level2_conf = y_level2_conf[idx]
        X_level2_granger = X_level2_granger[idx]
        X_level2_mse = X_level2_mse[idx]
        
    print "Reduction of the labels"   
    reduce_class = False #filtering on the selected classes
    reduce_feat = False #filtering on the features
    y_train_reduced = np.zeros(y_level2.shape[0],dtype=bool)
    sel_conf = classes #selected configurations = classes (no class reduction)
    if reduce_class:
        for i_conf, conf_i in enumerate(sel_conf):
            y_train_reduced = np.logical_or(y_train_reduced,y_level2==conf_i)
    else:        
        y_train_reduced  = np.ones(y_level2.shape[0],dtype=bool)
    
    if reduce_feat:
        feat_sel=np.array([9,10,12,14,16,17,18,19,20])#pairWise conditional 
                #np.array([0,4,8,9,10,12,14,16,17])#pairWise features 
                #np.arange(9)#univariate pairs 
                #np.array([1,2,3,5,6,7,11,13,15])#remove autoregressive features
        X_level2 = X_level2[:,feat_sel]
        X_level2_mse = X_level2_mse[:,feat_sel]
    
    X = X_level2[y_train_reduced]
    y = y_level2[y_train_reduced]
    y_level2_conf = y_level2_conf[y_train_reduced]
    X_granger = X_level2_granger[y_train_reduced] 
    X_mse = X_level2_mse[y_train_reduced]
    
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
    
    print "Preprocessing Train set"
    block_normalisation = False
    feature_space = feature_engineering([X, X_granger, X_mse], block_normalisation=block_normalisation)
    del X, X_granger, X_mse
    
    block_normalisation = False
    X = feature_normalisation(feature_space, block_normalisation=block_normalisation)  
        
    print "X:", X.shape
    print "Computing score."
    nComb = order_combinations.shape[0]
    cv = StratifiedKFold(y, n_folds=n_folds)
    optimal_decision = True #decidion based on a predefined score_matrix
    binary_class = True

    if binary_class:
            
        score_matrix = np.array([[1,0],[0,1]], dtype=int)#np.array([[0,-3],[0,1]], dtype=int)#score assigned to false/true positive and false/true negative
        y_new = [] #conversion from int representation of the class to binary matrix representation 
        y_new += [(class_to_configuration(y_i, verbose=False)) for y_i in np.squeeze(y)] 
        y_new = np.array(y_new, dtype=int)
        y_pred = np.zeros(y_new.shape, dtype=int)
        predicted_probability = np.zeros((X.shape[0], 6, 2))
        
        for i, (train, test) in enumerate(cv):
            print "Fold %d" % i
            index_x = np.append(np.triu_indices(3,1)[0], np.tril_indices(3,-1)[0])
            index_y = np.append(np.triu_indices(3,1)[1], np.tril_indices(3,-1)[1])
            
            for j in range(len(index_x)):
                print "Train."   
                clf.fit(X[train], y_new[train,index_x[j],index_y[j]])
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
        
        y_pred = np.zeros(y.shape, dtype=int) #directly the int representation of the trial is predicted
        score_matrix = compute_score_matrix(classes=classes, binary_score=[1,0,0,1])
        predicted_probability = np.zeros((X.shape[0], len(classes)))
        
        for i, (train, test) in enumerate(cv):
            print "Fold %d" % i            
            print "Train."        
            clf.fit(X[train], y[train]) #one classifier for each fold, but in this case it is a multiple classes classifier
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
    
    pwd_dest =  '%/results/' % os.getcwd()
    filename_save = '%ssimulatedMEG_25conf-RNDMax0.%dSynEff_tws%d_lag4_conf_r2_granger_mse_perTimeNorm_optimal_decision_binary_cv.mat' % (pwd_dest, synEff, time_window_size)         
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
