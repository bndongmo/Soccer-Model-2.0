# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:01:28 2023

@author: Bryant Ndongmo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaincc, factorial
from scipy.optimize import brentq, minimize


def weighted_average(arr, decay_factor):
    # Get the length of the array
    length = len(arr)
    
    # Compute the weights based on the decay factor
    weights = np.power(decay_factor, np.arange(length)[::-1])
    
    # Normalize the weights
    weights /= np.sum(weights)
    
    # Compute the weighted average
    weighted_avg = np.dot(arr, weights)
    return weighted_avg

def metrics(team, goals_for, goals_ag, xgoals_for, xgoals_ag, lines_for, lines_ag, weighted=False, decay_factor=.5):
    if not weighted:
        xgf = np.mean(xgoals_for[team])
        lf = np.mean(lines_for[team])
        gf = np.mean(goals_for[team])
        xga = np.mean(xgoals_ag[team])
        la = np.mean(lines_ag[team])
        ga = np.mean(goals_ag[team])
    else:
        xgf = weighted_average(xgoals_for[team], decay_factor)
        lf = weighted_average(lines_for[team], decay_factor)
        gf = weighted_average(goals_for[team], decay_factor)
        xga = weighted_average(xgoals_ag[team], decay_factor)
        la = weighted_average(lines_ag[team], decay_factor)
        ga = weighted_average(goals_ag[team], decay_factor)
    return gammaincc(xgf, lf), gammaincc(gf, xgf), gammaincc(gf, lf), gammaincc(xga, la), gammaincc(ga, xga), gammaincc(ga, la)
    

def get_a(x, y):
    def f(a):
        # Define the inner function f(a)
        # It takes a single argument 'a'

        # Compute the difference between gammaincc(a, x) and y
        result = gammaincc(a, x) - y

        # Return the result
        return result

    # Return the inner function f(a)
    return f



def convert_metric(line, metric):
    # Convert a given line using a metric

    # Use brentq method to find the value of 'a' that satisfies the equation get_a(line, metric)(a) = 0,
    # within the interval [0, 10]
    adjusted_line = brentq(get_a(line, metric), 0, 10)
    
    # Compute the natural logarithm of the ratio between the adjusted_line and the original line
    # to determine parameter for model
    result = np.log(max(1e-8,adjusted_line) / line)
    
    # Return the result
    return result

def gpmf(x, alpha, lamb):
    lamb *= (1 - alpha)
    p = (lamb * np.power(lamb + alpha * x, x - 1) * np.exp(-lamb - alpha * x)) / factorial(x)
    p = np.where(p <= 0, 1e-8, p)
    p = np.where(p > 1, 1, p)
    return p

def objective_function(params, X, y, method='ll', l1_wt=0, l2_wt=0, plot=False):
    beta = params
    alpha = -0.024
    y = y.values.flatten()


    mu_errors = np.log(gpmf(y, alpha, np.exp(X @ beta)))
    if method == 'll':
        
        return -np.mean(mu_errors)
    if method == 'll_regularized':
        adj1 = abs(beta[1]-1)-abs(beta[1])
        adj2 = (beta[1]-1)**2-beta[1]**2
        loss = -np.mean(mu_errors)+l1_wt*(np.linalg.norm(beta, ord=1)+adj1)+l2_wt*(np.linalg.norm(beta, ord=2)**2+adj2) 
        return loss
        
   
def fit(X_train, y_train):
    null = [0,1]+[0]*(X_train.shape[1]-2)
    return minimize(objective_function, null, args=(X_train, y_train), method='bfgs', options={'disp':False})
    
def fit_regularized(X_train, y_train,l1_wt=0,l2_wt=0):
    null = [0,1]+[0]*(X_train.shape[1]-2)
    return minimize(objective_function, null, args=(X_train, y_train,'ll_regularized', l1_wt, l2_wt), method='bfgs', options={'disp':False})