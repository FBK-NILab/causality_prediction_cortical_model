"""Collection of functions used by the classification scripts.
"""

import numpy as np
from create_level2_dataset import class_to_configuration


def score(configuration_true, configuration_pred, binary_score):
    """Score a single prediction of the confuguration matrix.
    """
    configuration_true = configuration_true.astype(np.bool)
    configuration_pred = configuration_pred.astype(np.bool)
    np.fill_diagonal(configuration_true, False)
    np.fill_diagonal(configuration_pred, False)
    score = ( binary_score[0] * (configuration_true[configuration_pred]).astype(np.int)).sum() + \
            ( binary_score[1] * (1.0 - configuration_pred[configuration_true]).astype(np.int)).sum() + \
            ( binary_score[2] * (1.0 - configuration_true[configuration_pred]).astype(np.int)).sum() + \
            ( binary_score[3] * (1.0 - configuration_true[np.logical_not(configuration_pred)]).astype(np.int)).sum() - (binary_score[3]*3) #remove the diagonal
    return float(score)


def compute_score_matrix(classes=range(64), binary_score=[1,0,-3,0]): #binary_score=[true_pos, false_neg, false_pos, true_neg]
    n = len(classes)    
    score_matrix = np.zeros((n, n))
    for i_c_true, c_true in enumerate(classes):
        for i_c_pred, c_pred in enumerate(classes):
            score_matrix[i_c_true, i_c_pred] = score(class_to_configuration(c_true, verbose=False), class_to_configuration(c_pred, verbose=False), binary_score)

    return score_matrix


def best_decision(prob_configuration, score_matrix=None):
    """Given the probability of each configuration, compute the
    expected scores and the best decision.
    """
    if score_matrix is None:
        score_matrix = compute_score_matrix()
    
    # Sanity checks:
    assert((prob_configuration >=0).all())
    prob_configuration = prob_configuration / prob_configuration.sum()

    scores = (score_matrix * prob_configuration[:,None]).sum(0)

    best = scores.argmax()
    return best, scores
    

if __name__ == '__main__':

    np.random.seed(0)

    c_true = 1
    c_pred = 2
    configuration_true = class_to_configuration(c_true)
    configuration_pred = class_to_configuration(c_pred)
    binary_score = np.ones(4)

    print "Score:", score(configuration_true, configuration_pred, binary_score)

    score_matrix = compute_score_matrix()

    prob_configuration = np.random.dirichlet(alpha=np.arange(64)**2)

    print "Given", prob_configuration
    best, scores = best_decision(prob_configuration, score_matrix=score_matrix)
    print "The score of each decision is:", scores
    print "The best decision is:", best
    print "With score:", scores[best]
    print "And p(c|X):", prob_configuration[best]
