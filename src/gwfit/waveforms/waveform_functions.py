import numpy as np
from ..physics import useful_relations as urel
from ..physics.constants import MTsun, time_fac, strain_fac


lambdagamma1 = [0.0069274,0.0302047,0.00630802,-0.120741,0.262716,0.00341518,-0.107793,0.27099,0.000737419,-0.0274962,0.0733151]
lambdagamma2 = [1.01034,0.000899312,0.283949,-4.04975,13.2078,0.103963,-7.02506,24.7849,0.030932,-2.6924,9.60937]
lambdagamma3 = [1.30816,-0.00553773,-0.0678292,-0.668983,3.40315,-0.0529658,-0.992379,4.82068,-0.00613414,-0.384293,1.75618]

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


#------Heff defined as functions-----------(added by Samanwaya Mukherjee)
def H_eff5(h1, h2, m1, m2, chi1, chi2):
    m = m1 + m2
    return h1 * (m1/m)**3 * (3 * chi1**2 + 1) * chi1 + h2 * (m2/m)**3 * (3 * chi2**2 + 1) * chi2 
def H_eff8(h1, h2, m1, m2, chi1, chi2):
    m = m1 + m2
    return 4 * np.pi * H_eff5(h1, h2, m1, m2, chi1, chi2) + h1 * (m1/m)**4 * (1 + np.sqrt(1 - chi1**2)) * (1 + 3 * chi1**2) + h2 * (m2/m)**4 * (1 + np.sqrt(1 - chi2**2)) * (1 + 3 * chi2**2)

# Leading order amplitude ------------------   

def A0(m1,m2,DL):
    DLt = DL * time_fac/strain_fac
    eta = m1*m2/(m1+m2)**2
    M = (m1+m2)*MTsun
    A = ((5.*eta/24.)**0.5/np.pi**(2./3.))*(M**(5./6.)/DLt)

    return A

#-----f_isco-----

def f_isco(M):
    '''
    M ... in sec
    '''
    return 1./6.**(3./2.)/np.pi/M

def f_isco_Msolar(M):
    '''
    M ... in solar mass
    '''
    # convert to sec
    return f_isco(M * MTsun)


#-------------- ISCO for KBH -------------- (arXiv: 2108.05861)

def r_hat_isco(chi):
    z1 = 1 + (1 - chi**2)**(1./3.)*((1 + chi)**(1./3.) + (1 - chi)**(1./3.))
    z2 = np.sqrt(3*chi**2 + z1**2)
    if chi == 0: return 3 + z2
    else: return 3 + z2 - (chi/np.abs(chi))*np.sqrt((3 - z1)*(3 + z1 + 2*z2))
def E_hat_isco(chi):
    return np.sqrt(1 - (2.)/(3*r_hat_isco(chi)))
def L_hat_isco(chi):
    return (2./(3*np.sqrt(3.)))*(1 + 2*np.sqrt(3*r_hat_isco(chi) - 2))
def f_isco_Msolar_KBH(m1,m2,chi1z,chi2z):
    k01 = -1.2019
    k02 = -1.20764
    k10 = 3.79245
    k11 = 1.18385
    k12 = 4.90494
    zeta = 0.41616
    k00 = -3.821158961
    m1 = m1*MTsun
    m2 = m2*MTsun
    M = m1 + m2 # initial total mass
    eta = (m1*m2)/(m1 + m2)**2
    S_hat = (chi1z*m1**2 + chi2z*m2**2)/(M**2*(1 - 2*eta))
    Erad_by_M = (0.0559745*eta + 0.580951*eta**2 - 0.960673*eta**3 + 3.35241*eta**4)\
             *((1 + S_hat*(-0.00303023 - 2.00661*eta + 7.70506*eta**2))/(1 + S_hat*(-0.067144 - 1.47569*eta + 7.30468*eta**2)))
    M_f = M*(1 - Erad_by_M)  # mass of the final black hole
    a_tot = (chi1z*m1**2 + chi2z*m2**2)/(m1 + m2)**2
    a_eff = a_tot + zeta*eta*(chi1z + chi2z)
    chi_f = a_tot + eta*(L_hat_isco(a_eff) - 2*a_tot*(E_hat_isco(a_eff) - 1))\
          + (k00 + k01*a_eff + k02*a_eff**2)*eta**2 + (k10 + k11*a_eff + k12*a_eff**2)*eta**3  # final spin of the KBH
    
    omega_hat_isco = 1./((r_hat_isco(chi_f))**(3./2.) + chi_f)
    
    return omega_hat_isco/(np.pi*M_f)

def Mf_chif(m1,m2,chi1z,chi2z):
    
    k01 = -1.2019
    k02 = -1.20764
    k10 = 3.79245
    k11 = 1.18385
    k12 = 4.90494
    zeta = 0.41616
    k00 = -3.821158961
    m1 = m1*MTsun
    m2 = m2*MTsun
    M = m1 + m2 # initial total mass
    eta = (m1*m2)/(m1 + m2)**2
    S_hat = (chi1z*m1**2 + chi2z*m2**2)/(M**2*(1 - 2*eta))
    Erad_by_M = (0.0559745*eta + 0.580951*eta**2 - 0.960673*eta**3 + 3.35241*eta**4)\
             *((1 + S_hat*(-0.00303023 - 2.00661*eta + 7.70506*eta**2))/(1 + S_hat*(-0.067144 - 1.47569*eta + 7.30468*eta**2)))
    M_f = M*(1 - Erad_by_M)  # mass of the final black hole
    a_tot = (chi1z*m1**2 + chi2z*m2**2)/(m1 + m2)**2
    a_eff = a_tot + zeta*eta*(chi1z + chi2z)
    chi_f = a_tot + eta*(L_hat_isco(a_eff) - 2*a_tot*(E_hat_isco(a_eff) - 1))\
          + (k00 + k01*a_eff + k02*a_eff**2)*eta**2 + (k10 + k11*a_eff + k12*a_eff**2)*eta**3  # final spin of the KBH
    
    M_f = M_f/MTsun #converting back to solar masses
    
    return M_f, chi_f

def f_RD_f_damp(m1,m2,chi1,chi2):
    
    M = m1 + m2 # initial total mass
    eta = (m1*m2)/(m1 + m2)**2
    
    f1 = 1.5251
    f2 = -1.1568
    f3 = 0.1292
    q1 = 0.7
    q2 = 1.4187
    q3 = -0.499
    
    Mf, chif = Mf_chif(m1,m2,chi1,chi2)
    
    Mf = Mf*MTsun
        
    f_RD = (1./(2.*np.pi*Mf)) * (f1 + f2*(1.-chif)**f3)
    
    f_damp = f_RD/(2.*(q1 + q2*(1.-chif)**q3))
    
    return f_RD, f_damp

def f_peak(m1,m2,chi1,chi2):
    
    q = m1/m2
    eta = urel.eta_of_q(q)
    M = (m1+m2)*MTsun
    chiPN = urel.chi_PN(m1,m2,chi1,chi2)
    
    gamma1 = Lambda(eta,chiPN,*lambdagamma1)
    gamma2 = Lambda(eta,chiPN,*lambdagamma2)
    gamma3 = Lambda(eta,chiPN,*lambdagamma3)
    
    f_RD, f_damp = f_RD_f_damp(m1,m2,chi1,chi2)
    
    f_peak = f_RD + f_damp*gamma3*(np.sqrt(1-gamma2**2+0j)-1)/gamma2
    
    f_peak = np.abs(f_peak)
    
    return f_peak
    
def f_MECO(m1,m2,chi1,chi2):

    eta = urel.eta_of_q(m1/m2)
    S = urel.chi_hat(m1,m2,chi1,chi2)

    fmecog = (0.018744340279608845 + 0.0077903147004616865 *eta + 0.003940354686136861 *eta**2 - 0.00006693930988501673*eta**3)/(1 - 0.10423384680638834 *eta) + (S**3 *(-0.0028303320109541955 - 0.01319368225506923 *eta - 
       0.014331096768277578 *eta**2 + 
       0.00015144033978081733 *eta**3 - 
       0.865598311695623 *eta**4) + 
    S* (0.0101926920659295 + 0.009158503243006886 *eta + 
       0.013708626030328923 *eta**2 - 
       0.020320785712379934 *eta**3 - 
       0.025528737356553345 *eta**4) + 
    S**2* (-0.0009694741018756255 - 
       0.0003933817108395515 *eta - 
       0.00037579451118948445 *eta**2 + 
       0.0011504883794752892 *eta**3 + 
       0.8394906079170564 *eta**4))/(1 + 
    S**2* (-0.22006798741315858 - 1.0787955021207019 *eta - 
       0.7365550694256703 *eta**2 - 
       0.07337456927445352 *eta**3 - 
       95.6841985329734 *eta**4) + 
    S**3* (0.13153712995215916 + 0.5316213737412727 *eta + 
       0.47333562697783144 *eta**2 - 
       0.2879144067301339 *eta**3 + 
       37.50017276813759 *eta**4) + 
    S* (-0.5471477439918454 - 0.4661075156477155 *eta + 
       0.94000545001549 *eta**2 + 
       0.19300625361069154 *eta**3 + 
       55.72974629895666 *eta**4)) - 0.028476219509487793 * (1 - 4 *eta)**0.5 *(1 - 
    2.8320701810535875 *eta) *eta**2 *(chi1 - chi2) - 0.0011713735642446144 *S* (1 - 4 *eta)**0.5 *eta* (1 + 
    4.905810747693807 *eta) *(chi1 - chi2) 
    + 0.00034375176678815234* (1 + 
    0.047545158629336504 *eta) *eta**2 *(chi1 - chi2)**2

    fmeco = fmecog/((m1+m2)*urel.MTsun)
    
    return fmeco

def Mf_peak_td_fit(m1,m2):
    eta = m1*m2/(m1+m2)**2
    W0 = 0.3578
    W2 = -0.1211
    W4 = 0.0432
    return (1./(2*np.pi))*(W0 + W2*(1-4*eta) + W4*(1-4*eta)**2)

def f_phi_end(m1,m2,chi1,chi2):


    f_isco = f_isco_Msolar_KBH(m1,m2,chi1,chi2)
    f_meco = f_MECO(m1,m2,chi1,chi2)
    f_rd,_ = f_RD_f_damp(m1,m2,chi1,chi2)
    
    f_phi_T = 0.6*(0.5*f_rd + f_isco)
    delta_R = 0.03*(f_phi_T - f_meco)

    phiend = f_meco - delta_R

    return phiend

def f_amp_end(m1,m2,chi1,chi2):
    
    f_isco = f_isco_Msolar_KBH(m1,m2,chi1,chi2)
    f_meco = f_MECO(m1,m2,chi1,chi2)

    ampend = f_meco + (1./4.)*(f_isco - f_meco)

    return ampend

def ampint_v2(m1,m2,chi1,chi2):
    
    eta = urel.eta_of_q(m1/m2)
    S = urel.chi_hat(m1,m2,chi1,chi2)
    
    v2 = (2.2436523786378983 +2162.4749081764216 * eta+24460.158604784723* eta**2-12112.140570900956*eta**3)/(1+120.78623282522702* eta+416.4179522274108* eta**2)+(S**2 *(-1.6646004496154818+ 39.690635399351976* eta-72.25797935657836*eta**2)+S *(2.0740930179437536 -57.61505238282958*eta+127.67945689822587*eta**2))/(1-1.2256937428769066*S+0.3083001769574955*S**2)-59.52510939953099*(1-4* eta)**0.5* (1-0.22052532962015753* eta) *eta**2 *(chi1 - chi2)

    return v2

def ampint_v3(m1,m2,chi1,chi2):
    eta = urel.eta_of_q(m1/m2)
    S = urel.chi_hat(m1,m2,chi1,chi2)

    v3 = (1.195392410912163 +1677.2558976605421 *eta+24838.37133975971 *eta**2-17277.938868280915 *eta**3)/(1+144.78606839716073 *eta+428.8155916011666 *eta**2)+(S**3* (-0.8003280997506247+ 7.795342953446871 *eta-13.184393749315833 *eta**2)+S**2 *(-0.386783853162149- 0.27672690933853966 *eta+23.606779769106137 *eta**2)+S *(1.4482201622320323 -31.52459226147479 *eta+57.36570415789376 *eta**2))/(1-0.6762974720861794 *S)-333.7662575986524 *(1-4 *eta)**0.5 *(1-1.5946715606839124 *eta) *eta**3 *(chi1-chi2)

    return v3
