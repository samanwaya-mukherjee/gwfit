import numpy as np
import scipy
scipy_vers = scipy.__version__
v = scipy_vers.split('.')
if int(v[0])<2 and int(v[1])>= 6: 
    from scipy.integrate import simpson,trapezoid
    scipy_version = 1
if int(v[0])<2 and int(v[1])< 6: 
    from scipy.integrate import simps,trapezoid
    scipy_version = 0
from scipy.interpolate import interp1d


#-----constants in SI units-----
GNewton = 6.6743e-11
cLight = 2.99792458e8
Msun = 1.9884099021470415e+30
Mpc = 3.085677581491367e+22
REarth = 6378136.6
AU = 1.4959787066e11
year = 3.1536e7
hPlanck = 6.62607015e-34

#-----convert mass in solar masses to seconds-----
MTsun = Msun * GNewton/cLight**3.
time_fac = MTsun
#-----convert mass/distance in solar mass/Mpc to dimensionless-----
strain_fac = GNewton/cLight**2.*Msun/Mpc

#-----mass ratio functions-----
def eta_of_q(q):
    return q/np.power(1+q,2)

def delta_of_q(q):
    return np.sqrt(1-4*eta_of_q(q))

def delta_of_eta(eta):
    return np.sqrt(1-4*eta)

def q_of_eta(eta,q_gt_1=1):
    if q_gt_1:
        return (1+delta_of_eta(eta))/(1-delta_of_eta(eta))
    else:
        return (1-delta_of_eta(eta))/(1+delta_of_eta(eta))

def M_of_Mc_eta(Mc,eta):
    return Mc/np.power(eta,3./5)

def Mc_of_M_eta(M,eta):
    return M*np.power(eta,3./5)

def m1_m2_of_M_eta(M,eta):
    delta = delta_of_eta(eta)
    return 0.5*M*(1+delta), 0.5*M*(1-delta)

def m1_m2_of_Mc_eta(Mc,eta):
    return m1_m2_of_M_eta(M_of_Mc_eta(Mc,eta),eta)

def M_eta_of_m1_m2(m1,m2):
    return m1+m2, eta_of_q(m1/m2)

def Mc_eta_of_m1_m2(m1,m2):
    eta = eta_of_q(m1/m2)
    return Mc_of_M_eta(m1+m2,eta), eta

#-----spin ratio functions-----
def chi_s(chi1,chi2):
    return 0.5*(chi1+chi2)

def chi_a(chi1,chi2):
    return 0.5*(chi1-chi2)

def chi_eff(m1,m2,chi1,chi2):
    return (m1 * chi1 + m2 * chi2) / (m1+m2)

# spin parametrization functions--
def chi_PN(m1,m2,chi1,chi2):
    eta = eta_of_q(m1/m2)
    chieff = chi_eff(m1,m2,chi1,chi2)
    chipn = chieff - (38.*eta/113.)*(chi1+chi2)
    return chipn

def chi_hat(m1,m2,chi1,chi2):
    eta = eta_of_q(m1/m2)
    chipn = chi_PN(m1,m2,chi1,chi2)
    return chipn/(1-(76.*eta/113.))
#---------------------------------

#-----derivatives of spin and mass functions-----
def del_Mc_M_of_eta(eta):
    return np.power(eta,-3./5)

def del_eta_M_of_Mc_eta(Mc,eta):
    return -3./5. * Mc * np.power(eta,-8./5)

def del_Mc_m1_of_Mc_eta(Mc,eta):
    delta = delta_of_eta(eta)
    return 1./2 * del_Mc_M_of_eta(eta) * (1 + delta)

def del_eta_m1_of_Mc_eta(Mc,eta):
    M = M_of_Mc_eta(Mc,eta)
    delta = delta_of_eta(eta)
    return 1./2 * del_eta_M_of_Mc_eta(Mc,eta) * (1 + delta) - M/delta

def del_Mc_m2_of_Mc_eta(Mc,eta):
    delta = delta_of_eta(eta)
    return 1./2 * del_Mc_M_of_eta(eta) * (1 - delta)

def del_eta_m2_of_Mc_eta(Mc,eta):
    M = M_of_Mc_eta(Mc,eta)
    delta = delta_of_eta(eta)
    return 1./2 * del_eta_M_of_Mc_eta(Mc,eta) * (1 - delta) + M/delta

def del_Mc_chi_eff(Mc,eta,chi1,chi2):
    M = M_of_Mc_eta(Mc,eta)
    m1, m2 = m1_m2_of_M_eta(M,eta)
    return -1./M * del_Mc_M_of_eta(eta) * chi_eff(m1,m2,chi1,chi2) + 1./M * (del_Mc_m1_of_Mc_eta(Mc,eta) * chi1 + del_Mc_m2_of_Mc_eta(Mc,eta) * chi2)

def del_eta_chi_eff(Mc,eta,chi1,chi2):
    M = M_of_Mc_eta(Mc,eta)
    m1, m2 = m1_m2_of_M_eta(M,eta)
    return -1./M * del_eta_M_of_Mc_eta(Mc,eta) * chi_eff(m1,m2,chi1,chi2) + 1./M * (del_eta_m1_of_Mc_eta(Mc,eta) * chi1 + del_eta_m2_of_Mc_eta(Mc,eta) * chi2)

def del_chi1_chi_eff(Mc,eta,chi1,chi2):
    M = M_of_Mc_eta(Mc,eta)
    m1, m2 = m1_m2_of_M_eta(M,eta)
    return 1./M * (m1 + m2 * chi2)

def del_chi2_chi_eff(Mc,eta,chi1,chi2):
    M = M_of_Mc_eta(Mc,eta)
    m1, m2 = m1_m2_of_M_eta(M,eta)
    return 1./M * (m2 + m1 * chi1)

#------Heff defined as functions-----------(added by Samanwaya Mukherjee)
def Heff5(h1, h2, m1, m2, chi1, chi2):
    m = m1 + m2
    return h1 * (m1/m)**3 * (3 * chi1**2 + 1) * chi1 + h2 * (m2/m)**3 * (3 * chi2**2 + 1) * chi2 
def Heff8(h1, h2, m1, m2, chi1, chi2):
    m = m1 + m2
    return 4 * np.pi * Heff5(h1, h2, m1, m2, chi1, chi2) + h1 * (m1/m)**4 * (1 + np.sqrt(1 - chi1**2)) * (1 + 3 * chi1**2) + h2 * (m2/m)**4 * (1 + np.sqrt(1 - chi2**2)) * (1 + 3 * chi2**2)
    

# tidal parameters

def lam_ts_of_lam_12_eta(lam1,lam2,eta): # from arXiv:1402.5156
#    q = q_of_eta(eta)
#    lam_t = 16./13. * ( (12 + q) * q**4 * lam1 + (12*q + 1) * lam2) / (1 + q)**5
    delta = delta_of_eta(eta)
    lam_t = 8./13. * ( (1. + 7. * eta - 31. * eta**2) * (lam1 + lam2) +
                       delta * (1. + 9. * eta - 11. * eta**2) * (lam1 - lam2) )
    delta_lam_t = 0.5 * ( delta * (1319. - 13272. * eta + 8944. * eta**2) / 1319. * (lam1 + lam2)+
                          (1319. - 15910. * eta + 32850. * eta**2 + 3380. * eta**3) / 1319. * (lam1 - lam2) )
    return lam_t, delta_lam_t

def lam_12_of_lam_ts_eta(lam_t,delta_lam_t,eta):
    delta = delta_of_eta(eta)
    lam1 = ((-(-6.76923076923077*delta_lam_t*delta*(-0.09090909090909091 - 0.8181818181818182*eta + 1.*eta**2) +
            19.076923076923077*delta_lam_t*(-0.03225806451612903 - 0.22580645161290322*eta + 1.*eta**2) +
            3.3904473085670963*delta*(0.1474731663685152 - 1.4838998211091234*eta + 1.*eta**2)*lam_t -
            1.281273692191054*(0.39023668639053255 - 4.707100591715976*eta + 9.718934911242604*eta**2 + 1.*eta**3)*lam_t))/
            (8.881784197001252e-16*eta - 1.4210854715202004e-14*eta**2 + 2.842170943040401e-14*eta**3 + 4.500379075056848*eta**4 - 232.4912812736922*eta**5))
    lam2 = ((delta_lam_t*(-1.5296267736621122e-19 + 3.0592535473242243e-19*delta*eta + 7.342208513578138e-18*eta**2 + 9.789611351437518e-18*eta**3 +
            (0.011550173712335778 - 9.789611351437518e-18*delta)*eta**4 + 0.11646425159938675*eta**5) +
            (-3.8240669341552804e-20 + 3.8240669341552804e-20*delta + (-4.588880320986336e-19 - 6.118507094648449e-19*delta)*eta +
            (9.789611351437518e-18 - 1.2237014189296897e-18*delta)*eta**2 + (-9.789611351437518e-18 + 1.4684417027156276e-17*delta)*eta**3 +
            (-0.0014297928149279568 - 0.007954723326344955*delta)*eta**4 + (0.07386363636363634 - 0.005511061254304498*delta)*eta**5)*lam_t)/
            (eta*(0.0909090909090909 - 0.09090909090909091*delta + (0.6363636363636364 - 0.8181818181818181*delta)*eta +
            (-2.818181818181818 + 1.*delta)*eta**2)*(-3.82026549483612e-18 + 6.112424791737792e-17*eta - 1.2224849583475584e-16*eta**2 -
            0.01935719503287065*eta**3 + 1.*eta**4)))
    return lam1, lam2


#-----overlap function-----------------------------
def scalar_product_integrand(hf, gf, psd):
    temp = np.multiply(hf,np.conj(gf))
    return 2*np.divide(temp+np.conj(temp),psd)

def scalar_product_freq_array(hf, gf, psd, freqs, df=None):
    if df is None:
        if scipy_version==0: return np.real(simps(scalar_product_integrand(hf, gf, psd),freqs))
        else: return np.real(simpson(y=scalar_product_integrand(hf, gf, psd),x=freqs))
    else:
        summ = 2.*np.real(((hf*np.conjugate(gf)+np.conjugate(hf)*gf)/psd).sum())*df
        return summ

#-----SNR function---------------------
def snr_square_integrand(hf, psd):
    a = 4.*np.divide(np.power(np.abs(hf),2),psd)
    a = np.where(a>1e5,0,a)
    return a

def snr_square_freq_array(hf, psd, freqs):
    if scipy_version==0: return simps(snr_square_integrand(hf, psd),freqs)
    else: return simpson(y=snr_square_integrand(hf, psd),x=freqs)

def snr_freq_array(hf, psd, freqs):
    return np.sqrt(snr_square_freq_array(hf, psd, freqs))

def snr_snr_sq_freq_array(hf, psd, freqs, df=None):
    if df is None:
        if scipy_version==0: snr_sq = simps(snr_square_integrand(hf, psd),freqs)
        else: snr_sq = simpson(y=snr_square_integrand(hf, psd),x=freqs)
        return np.sqrt(snr_sq), snr_sq
    else:
        summ = 4.*(np.divide(np.power(np.abs(hf),2),psd)).sum()*df
        summ = 2.*np.real(((hf*np.conjugate(hf)+np.conjugate(hf)*hf)/psd).sum())*df
        return np.sqrt(summ), summ

#-----fft method from Anuradha-------
def rfft_normalized(time_series, dt, n=None):
    return np.fft.rfft(time_series,n)*dt

def fft_normalized(time_series, dt, n=None):
    return np.fft.fft(time_series,n)*dt

# finding the range in f for the stitching region 
#(input: total frequency range, initial and final 
#stitching frequency; output: list of indices of f for which this window is satisfied, and the corresponding
#frequency array  -----  this works for time array as well!)  --------------

def fnew(f,f1,f2):   
    
    f_index = []
    f_new = []
    
    for i1 in range (len(f)):
        if f[i1]>=f1 and f[i1]<=f2: f_index.append(i1)
        if f[i1]>f2: break
    for i2 in range(len(f_index)):
        i3=f_index[i2]
        f_new.append(f[i3])
    
    return np.asarray(f_index), np.asarray(f_new)

# derivative function -----------

def dydx(x,y):
    
    yprime = []
    x2 = []
    for i in range(len(x)-1):
        z = (y[i+1]-y[i])/(x[i+1]-x[i])
        yprime.append(z)
        x2.append(x[i])
    
    interp = interp1d(x2,yprime,fill_value='extrapolate')
    add_value = interp(x[-1])
    yprime.append(add_value)
    x2.append(x[-1])
    return np.asarray(yprime)

def nth_derivative_from_data(x, y, n):
    """Computes the nth numerical derivative from discrete data using finite differences.
    
    Args:
        x: Array of x values.
        y: Array of y values.
        n: The order of the derivative.
    
    Returns:
        Array of nth derivative values.
    """
    if n == 0:
        return y
    dy_dx = np.gradient(y, x)  # First derivative
    for _ in range(n - 1):
        dy_dx = np.gradient(dy_dx, x)  # Higher-order derivatives
        x = x[1:-1]  # Trim edges to avoid redundancy
        dy_dx = dy_dx[1:-1]  # Trim edges
        y = y[1:-1]
    return x, y, dy_dx


## # BLENDING FUNCTION

## linear

def blendlin(f,f1,f2):
    blend_array=[]
    for i in range (len(f)):
        if f[i]<f1: blend_array.append(0.)
        if f[i]>=f1 and f[i]<=f2: blend_array.append((f[i]-f1)/(f2-f1))
        if f[i]>f2: blend_array.append(1.)
    return np.asarray(blend_array)

## cosine

def blendcos(f,f1,f2):
    blend_array=[]
    for i in range (len(f)):
        if f[i]<f1: blend_array.append(0.)
        if f[i]>=f1 and f[i]<=f2: blend_array.append((1+np.cos((np.pi*(f[i]-f2))/(f1-f2)))/2.)
        if f[i]>f2: blend_array.append(1.)
    return np.asarray(blend_array)

## WINDOW FUNCTION (PLANCK TAPER) FOR FOURIER TRANSORM

def planck_taper_window(t,t1=None,t2=None,t3=None,t4=None,only_right=False,only_left=False):
    
    w = []
    def y(t): return (t2-t1)/(t-t1) + (t2-t1)/(t-t2)
    def z(t): return (t3-t4)/(t-t3) + (t3-t4)/(t-t4)
    
    if only_right==True:
        if t3 is None: raise Exception('specify t3')
        if t4 is None: raise Exception('specify t4')
        for i in range(len(t)):
            if t[i]<=t3: w.append(1.)
            if t[i]<t4 and t[i]>t3: w.append(1./(np.exp(z(t[i]))+1))
            if t[i]>=t4: w.append(0.)
    
    if only_left==True:
        if t1 is None: raise Exception('specify t1')
        if t2 is None: raise Exception('specify t2')
        for i in range(len(t)):
            if t[i]<=t1: w.append(0.)
            if t[i]<t2 and t[i]>t1: w.append(1./(np.exp(y(t[i]))+1))
            if t[i]>=t2: w.append(1.)
    
    if only_right==only_left==False:       
        for i in range(len(t)):
            if t[i]<=t1: w.append(0.)
            if t[i]<t2 and t[i]>t1: w.append(1./(np.exp(y(t[i]))+1))
            if t[i]<=t3 and t[i]>=t2: w.append(1.)
            if t[i]<t4 and t[i]>t3: w.append(1./(np.exp(z(t[i]))+1))
            if t[i]>=t4: w.append(0.)
            
    return np.asarray(w)
