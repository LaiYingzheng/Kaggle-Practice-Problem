import numpy as np
import matplotlib.pyplot as plt
import sklearn
from Feature_eng import *
from Modelling import *
from cross_vali import *

sklearn.model_selection.learning_curve

def plot_learning_curve(estimator, title, X, Y, ylim = None, cv = None, n_jobs=1, train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
    '''''
    
    draw the learning curve by model
    
    :param
    -----------
    estimator: classifier 
    title: title
    X:feature, numpy type
    Y: target vector
    ylim: tuple formate(ymin/ymax)
    cv: cross validation, devide data into n sets, one of them is considered as cv, the rest(n-1) as training
    n_jobs: 1    
    
    '''''

    train_sizes, train_scores, test_scores = sklearn.model_selection.learning_curve(
        estimator, X, y, cv = cv, n_jobs= n_jobs, train_sizes = train_sizes, verbose = verbose)

    train_scores_mean = np.mean (train_scores, axis = 1)
    train_scores_std= np.std (train_scores, axis = 1)
    test_scores_mean = np.mean (test_scores, axis = 1)
    test_scores_std= np.std (test_scores, axis = 1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u'Training sample amount')
        plt.ylabel(u'score')

        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'b' )
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = 'r' )
        plt.plot(train_sizes, train_scores_mean, 'o-', color = 'b', label = u'training scores')
        plt.plot(train_sizes, test_scores_mean, 'o-', color = 'r', label = u'cross validation scores')

        plt.legend(loc = "best")

        plt.draw()
        plt.show()


    midpoint = ((train_scores_mean[-1] + train_scores_std[-1])+ (test_scores_mean[-1] - test_scores_std[-1]))/2
    diff = ((train_scores_mean[-1] + train_scores_std[-1]) -  (test_scores_mean[-1] - test_scores_std[-1]))
    return midpoint, diff

plot_learning_curve(clf, u"study curve", X, y)