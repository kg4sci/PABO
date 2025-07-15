
from scipy.stats import norm
import math
import numpy as np

############################acquisation functions#############################
def UCB(x,beta,GP):
    mean,std=GP.getPrediction(x)
    return mean + beta*std
def LCB(x,beta,GP):
    mean,std=GP.getPrediction(x)
    return mean - beta*std
def TS(x,beta,GP):
    mean=GP.model.sample_y(x.reshape(1,-1))
    mean=mean[0]
    return mean
def ei(x,beta,GP):
    mean,std=GP.getPrediction(x)
    y_best=min(GP.yValues)
    xi=1e-3    
    z = (y_best-xi-mean)/std
    return -1*(std*(z*norm.cdf(z) + norm.pdf(z)))
def pi(x,beta,GP):
    mean,std=GP.getPrediction(x)
    y_best=max(GP.yValues)
    xi=1e-3    
    z = (y_best-xi-mean)/std
    return -1*norm.cdf(z)
def compute_beta(iter_num,total_epochs,beta_start=1.0,beta_end=0.01):
   
    decay_rate = -np.log(beta_end / beta_start) / total_epochs  
    return beta_start * np.exp(-decay_rate * iter_num)
    
