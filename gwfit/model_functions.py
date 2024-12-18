import numpy as np
import useful_relations as urel 
import inspiral_model
from useful_relations import MTsun, time_fac, strain_fac
from pycbc import waveform
from scipy.optimize import curve_fit


def theta_pm(f,f0):
    theta = np.zeros_like(f)
    for i in range(len(f)):
        if f[i]<f0: theta[i] = -1
        else: theta[i] = 1
    
    thetap, thetam = (1+theta)/2., (1-theta)/2.
    
    return thetap, thetam

def Lambda(eta,chipn,*l_args):
    
    z = l_args[0] + l_args[1]*eta + (chipn-1)*(l_args[2] + l_args[3]*eta + l_args[4]*eta**2)\
                                  + (chipn-1)**2*(l_args[5] + l_args[6]*eta + l_args[7]*eta**2)\
                                  + (chipn-1)**3*(l_args[8] + l_args[9]*eta + l_args[10]*eta**2)
    return z

def Lambda_ns(eta,*l_args):
    
    z = l_args[0] + l_args[1]*eta + l_args[2]*eta**2 + l_args[3]*eta**3
    
    return z

# amplitude and phase ansatz ---------------------

# Here m1, m2, DL are not taken as inputs. previously defined values of these parameters will be used.

def aug_amp(f,m1,m2,DL,rho1,rho2,rho3):
    
    eta = (m1*m2)/(m1+m2)**2
    M = (m1+m2)*MTsun
    DLt = DL * time_fac/strain_fac
    Mf = M*f
#     x = (M**2*(np.sqrt(2.*eta/3.))*((Mf)**(-7./6.)) / (DLt*np.pi**(1./6.))) * (rho1*(Mf)**(7./3.) + rho2*(Mf)**(8./3.) + rho3*(Mf)**(9./3.))
#     x = (M**2*(np.sqrt(2.*eta/3.))*((Mf)**(-7./6.)) / (DLt*np.pi**(1./6.))) * (rho1*(Mf) + rho2*(Mf)**2 + rho3*(Mf)**3)
    x = (M**2*(np.sqrt(5.*eta/24.))*((Mf)**(-7./6.)) / (DLt*np.pi**(2./3.))) * (rho1*(Mf)**(7./3.) + rho2*(Mf)**(8./3.) + rho3*(Mf)**(9./3.))
#     x = (M**2*(np.sqrt(5.*eta/24.))*((Mf)**(-7./6.)) / (DLt*np.pi**(2./3.))) * (rho1*(Mf)**(8./3.) + rho2*(Mf)**(9./3.) + rho3*(Mf)**(10./3.))
    
    return x

def aug_phase(f,m1,m2,sigma1,sigma2,sigma3,sigma4):
    
    eta = (m1*m2)/(m1+m2)**2
    M = (m1+m2)*MTsun
    Mf = M*f
    x = (1/eta)*(sigma1*(Mf) + (3./4.)*sigma2*(Mf)**(4./3.) + (3./5.)*sigma3*(Mf)**(5./3.) + (1./2.)*sigma4*(Mf)**2)
    
    return x

def aug_phase_prime(f,m1,m2,sigma1,sigma2,sigma3,sigma4):
    
    eta = (m1*m2)/(m1+m2)**2
    M = (m1+m2)*MTsun
    Mf = M*f
    x = (1/eta)*(sigma1 + sigma2*(Mf)**(1./3.) + sigma3*(Mf)**(2./3.) + sigma4*(Mf))
    
    return x
  

# evaluating sigma and rho ----------------------

def sigma(hptilde,m1,m2,chi1,chi2,DL=200,tc=0,phic=0,iota=0,Mf_ini=0.0035,Mf_fin=0.019,PN45=True,show_plot=False,TH_phase_corr=True, TH_amp_corr=True,same_as_PhenomD=False):
    
    # The waveform hptilde should be a pycbc frequencyseries. 
    # Default values of initial and final values of Mf are 0.0035 and 0.019. Change as you will. 
    # If show_plot ==True, the function returns [best fit params array, frequency array of fit, phase derivative data, 
    # phase derivative fit]
    
    Ms = m1+m2
    M = Ms*MTsun
        
    phaseTHD = -np.unwrap(np.angle(hptilde))
    phiprime = urel.dydx(M*hptilde.sample_frequencies,phaseTHD)
    
    index1 = np.where(hptilde.sample_frequencies*M>=Mf_ini)[0][0]
    index2 = np.where(hptilde.sample_frequencies*M>=Mf_fin)[0][0]
    
    freq_inspiral = hptilde.sample_frequencies[index1:index2]
    phiprime = phiprime[index1:index2]
    
    def aug_phase_prime_fit(Mf,sigma1,sigma2,sigma3,sigma4):
    
        eta = (m1*m2)/(m1+m2)**2
        M = (m1+m2)*MTsun
        x = (1/eta)*(sigma1 + sigma2*(Mf)**(1./3.) + sigma3*(Mf)**(2./3.) + sigma4*(Mf))

        return x

    if PN45==False: _,htf2phase = inspiral_model.HTF2(freq_inspiral,m1,m2,chi1,chi2,DL,tc,phic,iota,amp_phase_only=True,TH_phase_corr=TH_phase_corr,TH_amp_corr=TH_amp_corr,same_as_PhenomD=same_as_PhenomD)
    else: _,htf2phase = inspiral_model.HTF2_45(f_ini=freq_inspiral[0],delta_f=freq_inspiral[1]-freq_inspiral[0],m1=m1,m2=m2,chi1=chi1,chi2=chi2,DL=DL,amp_phase_only=True)
    htf2prime = urel.dydx(M*freq_inspiral,htf2phase)
    difference = (phiprime - htf2prime)
    sigma,_ = curve_fit(aug_phase_prime_fit, M*freq_inspiral, difference)
    
    phiprimefit = htf2prime + aug_phase_prime(freq_inspiral,m1,m2,sigma[0],sigma[1],sigma[2],sigma[3])

    if show_plot == True: return sigma, freq_inspiral, phiprime, phiprimefit
    return sigma

def rho(hptilde,m1,m2,chi1,chi2,DL=200,tc=0,phic=0,iota=0,Mf_ini=0.0035,Mf_fin=0.019,show_plot=False):
    
    #The waveform hptilde should be a pycbc frequencyseries. 
    #Default values of initial and final values of Mf are 0.0035 and 0.019. Change as you will.
    # If show_plot ==True, the function returns [best fit params array, frequency array of fit, phase derivative data, 
    # phase derivative fit]
    
    Ms = m1+m2
    M = Ms*MTsun
    eta = (m1*m2)/(m1+m2)**2
    
    DLt = DL * time_fac/strain_fac
    
    
    ampTHD = waveform.utils.amplitude_from_frequencyseries(hptilde)
    
    index1 = np.where(hptilde.sample_frequencies*M>=Mf_ini)[0][0]
    index2 = np.where(hptilde.sample_frequencies*M>=Mf_fin)[0][0]
    
    freq_inspiral = hptilde.sample_frequencies[index1:index2]
    amp_inspiral = ampTHD[index1:index2]
    
    def aug_amp_fit(Mf,rho1,rho2,rho3):
    
        eta = (m1*m2)/(m1+m2)**2
        M = (m1+m2)*MTsun
        x = (M**2*(np.sqrt(2.*eta/3.))*((Mf)**(-7./6.)) / (DLt*np.pi**(1./6.))) * (rho1*(Mf)**(7./3.) + rho2*(Mf)**(8./3.) + rho3*(Mf)**(9./3.))
#         x =  (rho1*(Mf)**(7./3.) + rho2*(Mf)**(8./3.) + rho3*(Mf)**(9./3.))
#         x =  (rho1*(Mf) + rho2*(Mf)**2+ rho3*(Mf)**3)

        return x

    htf2amp,_ = inspiral_model.HTF2D(freq_inspiral,m1,m2,chi1,chi2,DL,tc,phic,iota,amp_phase_only=True)
    
    A = (M**2*(np.sqrt(2.*eta/3.))*((M*freq_inspiral)**(-7./6.)) / (DLt*np.pi**(1./6.)))
    
    difference = (amp_inspiral - htf2amp)
    rho,_ = curve_fit(aug_amp_fit, M*freq_inspiral, difference)

    ampfit = htf2amp + aug_amp(freq_inspiral,m1,m2,DL,rho[0],rho[1],rho[2])

    if show_plot == True: return rho, freq_inspiral, amp_inspiral, ampfit, difference
    return rho
