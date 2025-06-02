import numpy as np 


def compactness(lambda0):
    return 3.531e−1 − 2.889e−2*np.log(lambda0)− 1.056e−3*(np.log(lambda0))**2 + 1.295e−4*(np.log(lambda0))**3

def m(lambda0,b0,b1,b2,b3):
    logl = np.log(lambda0)
    return b0 + b1* logl + b2*logl**2 + b3*logl**3


def contact_freq(m1,m2,lambda1,lambda2):
    return 0