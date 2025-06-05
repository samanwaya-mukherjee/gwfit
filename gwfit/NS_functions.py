import numpy as np 
import useful_relations as urel

## arXiv:2112.10824
def compactness(lambda0):
    return 3.531e-1 - 2.889e-2*np.log(lambda0)- 1.056e-3*(np.log(lambda0))**2 + 1.295e-4*(np.log(lambda0))**3

def m_of_lambda(lambda0,b0,b1,b2,b3):
    logl = np.log(lambda0)
    return b0 + b1* logl + b2*logl**2 + b3*logl**3

## arXiv:1804.02235
def contact_freq(m1,m2,lambda1,lambda2):
    C1 = compactness(lambda1)
    C2 = compactness(lambda2)
    k21 = (3/2)*lambda1*C1**5
    k22 = (3/2)*lambda2*C2**5
    X1 = m1/(m1+m2)
    X2 = m2/(m1+m2)

    k2T = 2*((X2/X1)*(X1/C1)**5*k21 + (X1/X2)*(X2/C2)**5*k22) # Eq. (10) 

    n1=3.354e-2
    n2=4.315e-5
    d1=7.542e-2
    d2=2.236e-4
    omega0=0.3586

    omega = omega0*(np.sqrt(X2/X1)*(1+n1*k2T+n2*k2T**2)/(1+d1*k2T+d2*k2T**2)) # Eq. (11) 
    
    M = (m1+m2)*urel.MTsun
    f = omega/(2*np.pi*M)

    return f

