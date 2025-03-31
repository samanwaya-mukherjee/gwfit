import numpy as np
from useful_relations import MTsun, time_fac, strain_fac
import waveform_functions as wfn
from pycbc import types, waveform, fft
from scipy.special import psi as digamma

## Nonspinning orbital energy function (4PN) and the GW flux function (4.5PN) (arXiv:2304.11185)

##Eq. (4)
def gw_flux(f,m1,m2):
    M = (m1+m2)*MTsun
    Mf = M*f
    q = m1/m2
    eta = q/(1+q)**2
    v = (np.pi*Mf)**(1./3.)
    prefactor = (32./5.)*eta**2*v**10
    p_v0 = 1.
    p_v2 = -1247./336.-(35./12.)*eta
    p_v3 = 4*np.pi
    p_v4 = -44711./9072. + (9271./504.)*eta + (65./18.)*eta**2
    p_v5 = (-(8191./672.)-(583./24.)*eta)*np.pi
    p_v6 = (6643739519./69854400) + (16./3.)*np.pi**2 - (1712./105.)*np.euler_gamma - (856./105.)*np.log(16*v**2) + (-(134543./7776.)+(41./48.)*np.pi**2)*eta - (94403./3024.) - (775./324.)*eta**2
    p_v7 = (-(16285./504.) + (214745./1728.)*eta + (193385./3024.)*eta**2)*np.pi
    p_v8 = -(323105549467./3178375200.) + (232597./4410)*np.euler_gamma - (1369./126.)*np.pi**2 + (39931./294.)*np.log(2) - (47385./1568.)*np.log(3) + (232597./8820.)*np.log(v**2)\
        + ((-1452202403629./1466942400.) + (41478./245.)*np.euler_gamma - (267127./4608.)*np.pi**2 + (479062./2205.)*np.log(2) + (47385./392.)*np.log(3) + (20739./245.)*np.log(v**2))*eta\
        + ((1607125./6804.) - (3157./384.)*np.pi**2)*eta**2 + (6785./504.)*eta**3 + (5./6.)*eta**4
    p_v9 = ((265978667519./745113600.) - (6848./105.)*np.euler_gamma - (3424./105.)*np.log(16*v**2) + ((2062241./22176.) + (41./12.)*np.pi**2)*eta - (133112905./290304.)*eta**2 - (3719141./38016.)*eta**3)*np.pi

    return prefactor*(p_v0 + p_v2*v**2 + p_v3*v**3 + p_v4*v**4 + p_v5*v**5 + p_v6*v**6 + p_v7*v**7 + p_v8*v**8 + p_v9*v**9)

##Eq. (3)
def E_orb(f,m1,m2):
    M = (m1+m2)*MTsun
    Mf = M*f
    q = m1/m2
    eta = q/(1+q)**2
    eta = q/(1+q)**2
    v = (np.pi*Mf)**(1./3.)
    prefactor = -(1/2)*eta*M*v**2
    p_v0 = 1.
    p_v2 = (9+eta)/12
    p_v4 = (-27/8) + (19/8)*eta - (eta**2/24)
    p_v6 = -(675/64) + (34445/576 - (205/96)*np.pi**2)*eta - (155/96)*eta**2 - (35/5184)*eta**3
    p_v8 = -3969/128 + (-123671/5760 + (9037/1536)*np.pi**2 + (896/15)*np.euler_gamma + (448/15)*np.log(16*v**2))*eta + (-498449/3456 + (3157/576)*np.pi**2)*eta**2 + (301/1728)*eta**3 + (77/31104)*eta**4
    
    return prefactor*(p_v0 + p_v2*v**2 + p_v4*v**4 + p_v6*v**6 + p_v8*v**8)

# TaylorF2 with tidal heating --------------------------------------
####################################################################


# Polygamma function with n=0 (digamma) -------
def B2(x,y):
    z = x+1j*y
    temp = digamma(z)
    return temp.imag

def dmdt(f,m1,m2,chi1,chi2):
    
    Ms = m1+m2
    eta = (m1*m2)/Ms**2
    v = (np.pi*Ms*MTsun*f)**(1./3.)
    dEdt_N = (32./5.)*(eta**2)*(v**10)
    
    dm1dt = dEdt_N*(v**5/4.)*(m1/Ms)**3*(1+3*chi1**2)*(-chi1 + 2*(1+np.sqrt(1-chi1**2))*(m1/Ms)*v**3)
    dm2dt = dEdt_N*(v**5/4.)*(m2/Ms)**3*(1+3*chi2**2)*(-chi2 + 2*(1+np.sqrt(1-chi2**2))*(m2/Ms)*v**3)
    
    return dm1dt, dm2dt

def dmdt_new(f,m1,m2,chi1,chi2):
    
    Ms = m1+m2
    eta = (m1*m2)/Ms**2
    v = (np.pi*Ms*MTsun*f)**(1./3.)
    
    k1 = np.sqrt(1.-chi1**2)
    k2 = np.sqrt(1.-chi2**2)
    X1 = m1/Ms
    X2 = m2/Ms
    chi_tilde = (2*X1**2 + 3.*eta)*chi1 + (2*X2**2 + 3.*eta)*chi2
    chi_bar1 = X1*(1.+X1)*chi1 + 3.*eta*chi2
    chi_bar2 = X2*(1.+X2)*chi2 + 3.*eta*chi1
    
    B21 = B2(3.,2.*chi1/k1)
    B22 = B2(3.,2.*chi2/k2)
        
    omega_H1 = chi1/(2.*m1*MTsun*(1.+k1)) ## angular velocity of horizon for BH 1
    omega_H2 = chi2/(2.*m2*MTsun*(1.+k2)) ## angular velocity of horizon for BH 2
    
    omega1 = (v**3/(Ms*MTsun))*(1.-eta*v**2 + (1./2.)*(chi_tilde-chi_bar1)*v**3)
    omega2 = (v**3/(Ms*MTsun))*(1.-eta*v**2 + (1./2.)*(chi_tilde-chi_bar2)*v**3)
    
    C_x1 = -(16./5.)*(m1*MTsun)**2*(X1**2)*eta**2*(1. + k1)*v**12*(1. + 3.*chi1**2 + (1./4.)*(3.*(2. + chi1**2) + 2*X1*(1. + 3.*chi1**2)*(2. + 3.*X1))*v**2 + v**3*((1./2.)*(-4. + 3.*chi1**2)*chi2 - 2.*X1*(1. + 3.*chi1**2)*(X1*(chi1+chi2) + 4*B21) + X1*(-(2./3.)*(23. + 30.*k1)*chi1 + (7. - 12.*k1)*chi1**3 + 4.*chi2 + (9.*chi1**2*chi2/2.))))
    
    C_x2 = -(16./5.)*(m2*MTsun)**2*(X2**2)*eta**2*(1. + k2)*v**12*(1. + 3.*chi2**2 + (1./4.)*(3.*(2. + chi2**2) + 2*X2*(1. + 3.*chi2**2)*(2. + 3.*X2))*v**2 + v**3*((1./2.)*(-4. + 3.*chi2**2)*chi1 - 2.*X2*(1. + 3.*chi2**2)*(X2*(chi1+chi2) + 4*B22) + X2*(-(2./3.)*(23. + 30.*k2)*chi2 + (7. - 12.*k2)*chi2**3 + 4.*chi1 + (9.*chi2**2*chi1/2.))))
    
    dm1dt = omega1*(omega_H1 - omega1)*C_x1
    dm2dt = omega2*(omega_H2 - omega2)*C_x2
    
    return dm1dt, dm2dt



# phase and amplitude corrections due to tidal heating ---------------

def psiTH(f,m1,m2,chi1,chi2,H1,H2):
    
    m1, m2 = m1*time_fac, m2*time_fac
    eta = (m1*m2)/(m1+m2)**2
    Heff5 = wfn.H_eff5(H1,H2,m1,m2,chi1,chi2)
    Heff8 = wfn.H_eff8(H1,H2,m1,m2,chi1,chi2)
    v = (np.pi*(m1+m2)*f)**(1./3.)

    psi_so1 = (1/6.)*(-56*eta - 73*np.sqrt(1 - 4*eta) + 73)*chi1
    psi_so2 = (1/6.)*(-56*eta + 73*np.sqrt(1 - 4*eta) + 73)*chi2
    psi_so = psi_so1 + psi_so2
    con = 3./(128.*eta)/v**5
    con = np.nan_to_num(con, nan=0)
    term1 = -(10/9.)*(v**5)*Heff5*(3*np.log(v) + 1)
    term2 = -(5/168.)*(v**7)*Heff5*(952*eta + 995)
    term3 = (5/9.)*(v**8)*(3*np.log(v) - 1)*(-4*Heff8 + Heff5*psi_so)

    heated_phase = con*(term1 + term2 + term3)

    return heated_phase

## ------------------------------------- Dephasing expression from Saketh et al (arXiv:2212.13095) ------------------


#Dephasing (Eqn 5.12 - 5.20 of arXiv: 2212.13095) -----------
def psiTH_new(f,m1,m2,chi1,chi2,v0_v3_term=1):

    m1, m2 = m1*time_fac, m2*time_fac
    eta = (m1*m2)/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    chi_s = (chi1+chi2)/2.
    chi_a = (chi1-chi2)/2.
    k1 = np.sqrt(1.-chi1**2)
    k2 = np.sqrt(1.-chi2**2)
    k_s = (k1+k2)/2.
    k_a = (k1-k2)/2.
    B21 = B2(3,2*chi1/k1)
    B22 = B2(3,2*chi2/k2)
    B2_s = (B21+B22)/2.
    B2_a = (B21-B22)/2.
    v = (np.pi*(m1+m2)*f)**(1./3.)
    con = 3./(128.*eta)/v**5
        
    delta_psi_5 = -(10./9.)*((1.- 3.*eta)*chi_s*(1 + 9.*chi_a**2 + 3.*chi_s**2) + delta*(1.-eta)*chi_a*(1 + 3.*chi_a**2 + 9.*chi_s**2))
    
    delta_psi_5_l = 3.*delta_psi_5
    
    delta_psi_7 = (5./168.)*(delta*chi_a*(-1667. - 4371.*chi_a**2 - 13113.*chi_s**2 + 616.*eta**2*(1 + 3.*chi_a**2 + 9.*chi_s**2) + 5.*eta*(311. + 807.*chi_a**2 +2421.*chi_s**2)) + chi_s*(840.*eta**2*(9.*chi_a**2 + 3.*chi_s**2 + 1.) + eta*(38331.*chi_a**2 + 12777.*chi_s**2 + 4889.) - 13113.*chi_a**2 - 4371.*chi_s**2 - 1667.))
    
    delta_psi_8_a = -(5./27.)*(144.*np.pi*delta*(eta-1.)*chi_a**3 + 48.*np.pi*delta*(eta-1)*chi_a + 3.*(278.*eta**2 - 370.*eta + 75.)*chi_a**4 + (-36.*eta**2 + 213.*eta - 67.)*chi_a**2 + chi_s*(-12.*delta*(eta**2 + 190.*eta - 75.)*chi_a**3 + 2.*delta*(10.*eta**2 + 124.*eta - 67.)*chi_a + 432.*np.pi*(3.*eta -1)*chi_a**2 + 48.*np.pi*(3.*eta - 1)) + chi_s**2*(432.*np.pi*delta*(eta -1)*chi_a + 90.*(36.*eta**2 - 62.*eta + 15.)*chi_a**2 - 172.*eta**2 + 303.*eta - 67.) + chi_s**3*(12.*delta*(21.*eta**2 - 130.*eta + 75.)*chi_a + 144.*np.pi*(3.*eta - 1)) + 3.*(82.*eta**2 -250.*eta + 75.)*chi_s**4 - 12.*(2.*eta**2 - 4.*eta + 1.))
    
    delta_psi_8_b = -(20./9.) * ((delta*(2.*eta - 1.)*k_a + (-1. - 2.*eta**2 + 4.*eta)*k_s)*(1. + 6.*chi_a**4 + 13.*chi_s**2 + 6.*chi_s**4 + chi_a**2*(13. + 36.*chi_s**2)) - 2.*(k_a*(1. - 4.*eta + 2.*eta**2) + delta*(1. - 2.*eta)*k_s)*chi_a*chi_s*(13. + 12.*(chi_a**2 + chi_s**2)))
    
    delta_psi_8_c = (80./9.)*(B2_s*((2.*eta**2 - 4.*eta + 1.)*chi_s*(9.*chi_a**2 + 3.*chi_s**2 +1.) - delta*(2.*eta - 1.)*chi_a*(3.*chi_a**2 + 9.*chi_s**2 + 1.)) + B2_a*(3.*(2.*eta**2 - 4.*eta + 1.)*chi_a**3 + 9.*delta*(1. - 2.*eta)*chi_a**2*chi_s + (2.*eta**2 - 4.*eta + 1.)*chi_a*(9.*chi_s**2 + 1.) - delta*(2.*eta - 1.)*chi_s*(3.*chi_s**2 + 1.)))
    
    delta_psi_8 = delta_psi_8_a + delta_psi_8_b + delta_psi_8_c
    
    delta_psi_8_l = -3.*delta_psi_8
    
    if v0_v3_term: delta_psi = con*(delta_psi_5*v**5 + delta_psi_5_l*v**5*np.log(v) + delta_psi_7*v**7 + delta_psi_8*v**8 + delta_psi_8_l*v**8*np.log(v))
    else: delta_psi = con*(delta_psi_5_l*v**5*np.log(v) + delta_psi_7*v**7 + delta_psi_8_l*v**8*np.log(v))
    
    
    return delta_psi

def ampTH(f,m1,m2,chi1,chi2,DL):
    
    m1, m2 = m1*time_fac, m2*time_fac
    M = m1 + m2
    DLt = DL * time_fac/strain_fac
    
    eta = (m1*m2)/(m1+m2)**2
    Heff5 = wfn.H_eff5(1.,1.,m1,m2,chi1,chi2)
    Heff8 = wfn.H_eff8(1.,1.,m1,m2,chi1,chi2)
    v = (np.pi*M*f)**(1./3.)
    
    A_so_1 = chi1*(179.*(np.sqrt(1-4*eta)-1) + 148.*eta)
    A_so_2 = chi2*(-179.*(np.sqrt(1-4*eta)+1) + 148.*eta)
    A_so = A_so_1 + A_so_2
    
    term1 = (Heff5/8.)*v**5
    term2 = ((1079./1792.) + (103.*eta/192.))*Heff5*v**7
    term3 = -(1./384.)*(A_so*Heff5 + 96.*(Heff8 - Heff5*np.pi))*v**8
    
    amp = ((M**(5./6.))*(np.sqrt(5.*eta/24.))*(f**(-7./6.)) / (DLt*np.pi**(2./3.))) * (term1 + term2 + term3)
    
    return amp

## HTF2 from PN
########################################
def HTF2(f, m1, m2, chi1, chi2, DL, tc, phic, iota, amp_phase_only=False, amp_phase_hp_hc=False, TH_phase_corr=True, TH_phase_corr0=False, TH_amp_corr=True, same_as_PhenomD=False):
    '''
    m1,m2 ... in solar mass
    DL ... in mega parsec
    '''
    PI = np.pi
    log = np.log
    sin = np.sin
    cos = np.cos
    GammaE = 0.577215664901532
    
    # convert to sec
    Ms = m1+m2
    M = Ms * time_fac
    DLt = DL * time_fac/strain_fac

    # get sym and asym chi combinations
    chi_s = 0.5*(chi1+chi2)
    chi_a = 0.5*(chi1-chi2)

    '''
    Mc is in sec, e.g., Mc = 10*MTSUN_SI (for 10 solar mass)
    DL is in sec, e.g., DL = 100*1e6*PC_SI/C_SI (for 100 Mpc)
    '''
    
    eta = (m1*m2)/(m1+m2)**2
    delta = (1.-4.*eta)**0.5
    v  = (PI*M*f)**(1./3.)
    flso = wfn.f_isco(M)
    vlso = (PI*M*flso)**(1./3.)
    
    
    beta = (113./12.)*(chi_s + delta*chi_a - (76.*eta/113.)*chi_s)
    
    sigma = chi_a**2*((81./16.) - 20*eta) + (81.*chi_a*chi_s*delta)/8. + chi_s**2*((81./16.) - eta/4.)
    
    epsilon = (502429./16128. - 907.*eta/192.)*delta*chi_a + (5.*eta**2/48. - 73921.*eta/2016. + 502429./16128.)*chi_s
    
    gamma = (732985./2268. - 24260.*eta/81. - 340.*eta**2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*delta*chi_a
    
    
    # amplitude correction (w/o TH)
    
    
    A = wfn.A0(m1,m2,DL)
    
    
    a0 = 1.
    
    a1 = 0
    
    a2 = 11.*eta/8. + 743./672.
    
    a3 = beta/2. - 2.*PI
    
    a4 = 1379.*eta**2/1152. + 18913.*eta/16128. + 7266251./8128512. - sigma/2.
    
    a5 = 57.*PI*eta/16. - 4757.*PI/1344. + epsilon
    
    a6 = 856.*GammaE/105. + 67999.*eta**3/82944. - 1041557.*eta**2/238048. - 451.*PI**2*eta/96. + 10.*PI**2/3. + 3526813753.*eta/27869184. - 29342493702821./500716339200. + 856.*np.log(4*v)/105.
    
    a7 = -1349.*PI*eta**2/24192. - 72221.*PI*eta/24192. - 5111593.*PI/2709504.
    
    # 3.5PN phasing (point particle limit)
    
    p0 = 1.

    p1 = 0

    p2 = (3715./756. + (55.*eta)/9.)

    p3 = (-16.*PI + (113.*delta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s)

    p4 = (15293365./508032. + (27145.*eta)/504.+ (3085.*eta**2)/72. + (-405./8. + 200.*eta)*chi_a**2 - (405.*delta*chi_a*chi_s)/4. + (-405./8. + (5.*eta)/2.)*chi_s**2)


    p5 = (38645.*PI/756. - 65.*PI*eta/9. - gamma)

    p5L = (38645.*PI/756. - 65.*PI*eta/9. - gamma)*3*log(v/vlso)

    p6 = (11583231236531./4694215680. - 640./3.*PI**2 - 6848./21.*GammaE + eta*(-15737765635./3048192. + 2255./12.*PI**2) + eta*eta*76055./1728. - eta*eta*eta*127825./1296. \
         - (6848./21.)*log(4.) + PI*(2270.*delta*chi_a/3. + (2270./3. - 520.*eta)*chi_s) + (75515./144. - 8225.*eta/18.)*delta*chi_a*chi_s \
         + (75515./288. - 263245.*eta/252. - 480.*eta**2)*chi_a**2 + (75515./288. - 232415.*eta/504. + 1255.*eta**2/9.)*chi_s**2)
    ## PhenomD --------------------------------------------
    p6_2 = (11583231236531./4694215680. - 640./3.*PI**2 - 6848./21.*GammaE + eta*(-15737765635./3048192. + 2255./12.*PI**2) + eta*eta*76055./1728. - eta*eta*eta*127825./1296. \
         - (6848./21.)*log(4.) + PI*(2270.*delta*chi_a/3. + (2270./3. - 520.*eta)*chi_s) + (75515./144. - 8225.*eta/18.)*delta*chi_a*chi_s)
    ##-----------------------------------------------------

    p6L = -(6848./21.)*log(v)

    p7 = (((77096675.*PI)/254016. + (378515.*PI*eta)/1512.- (74045.*PI*eta**2)/756. + (-25150083775./3048192. + (10566655595.*eta)/762048. - (1042165.*eta**2)/3024. + (5345.*eta**3)/36.
         + (14585./8. - 7270.*eta + 80.*eta**2)*chi_a**2)*chi_s + (14585./24. - (475.*eta)/6. + (100.*eta**2)/3.)*chi_s**3 + delta*((-25150083775./3048192.
         + (26804935.*eta)/6048. - (1985.*eta**2)/48.)*chi_a + (14585./24. - 2380.*eta)*chi_a**3 + (14585./8. - (215.*eta)/2.)*chi_a*chi_s**2)))
    
    ## PhenomD --------------------------------------------
    p7_2 = ((77096675.*PI)/254016. + (378515.*PI*eta)/1512.- (74045.*PI*eta**2)/756. + (-25150083775./3048192. + (10566655595.*eta)/762048. - (1042165.*eta**2)/3024. + (5345.*eta**3)/36.)*chi_s + delta*((-25150083775./3048192.
         + (26804935.*eta)/6048. - (1985.*eta**2)/48.)*chi_a))
    ##-----------------------------------------------------

    phase = 2*f*PI*tc - phic - PI/4. + (3./(128.*v**5*eta))*(p0 + v*p1 + v**2*p2 + v**3*p3+ v**4*p4 + v**5*(p5+p5L) + v**6*(p6+p6L) + v**7*p7)
    if same_as_PhenomD==True: phase = 2*f*PI*tc - phic - PI/4. + (3./(128.*v**5*eta))*(p0 + v*p1 + v**2*p2 + v**3*p3+ v**4*p4 + v**5*(p5+p5L) + v**6*(p6_2+p6L) + v**7*p7_2)

    if TH_phase_corr==True: phase += psiTH_new(f,m1,m2,chi1,chi2)
    if TH_phase_corr0==True: phase += psiTH(f,m1,m2,chi1,chi2,1.,1.)
    
    amp = A*f**(-7./6.)*(a0 + v*a1 + v**2*a2 + v**3*a3 + v**4*a4 + v**5*a5 + v**6*a6 + v**7*a7)
    if TH_amp_corr==True: amp += ampTH(f,m1,m2,chi1,chi2,DL)
    
    hp = 0.5*(1+(cos(iota))**2)*amp*(cos(phase) - 1j*sin(phase))
    hc = -1j*cos(iota)*amp*(cos(phase) - 1j*sin(phase))

    if amp_phase_hp_hc == True: return amp, phase, hp, hc
    elif amp_phase_only == True: return amp, phase
    else: return hp, hc
    

## HTF2 with 4.5PN phase (arXiv: 2304.11185)
################################################
def HTF2_45(f_ini=10.,f_final=None, delta_f=None, m1=10, m2=10, chi1=0, chi2=0, H1=1, H2=1, DL=200, tc=0, phic=0, iota=0, amp_phase_only=False, amp_phase_hp_hc=False):

    if f_final is None: f_final = wfn.f_isco_Msolar(m1+m2)
    if delta_f is None: raise Exception('Specify delta_f')
    f = np.arange(f_ini,f_final,delta_f)

    PI = np.pi
    log = np.log
    sin = np.sin
    cos = np.cos
    GammaE = 0.577215664901532
    
    # convert to sec
    Ms = m1+m2
    M = Ms * time_fac
    DLt = DL * time_fac/strain_fac

    # get sym and asym chi combinations
    chi_s = 0.5*(chi1+chi2)
    chi_a = 0.5*(chi1-chi2)

    '''
    Mc is in sec, e.g., Mc = 10*MTSUN_SI (for 10 solar mass)
    DL is in sec, e.g., DL = 100*1e6*PC_SI/C_SI (for 100 Mpc)
    '''
    
    eta = (m1*m2)/(m1+m2)**2
    delta = (1.-4.*eta)**0.5
    v  = (PI*M*f)**(1./3.)
    flso = wfn.f_isco(M)
    vlso = (PI*M*flso)**(1./3.)
    
    beta = (113./12.)*(chi_s + delta*chi_a - (76.*eta/113.)*chi_s)
    
    sigma = chi_a**2*((81./16.) - 20*eta) + (81.*chi_a*chi_s*delta)/8. + chi_s**2*((81./16.) - eta/4.)
    
    epsilon = (502429./16128. - 907.*eta/192.)*delta*chi_a + (5.*eta**2/48. - 73921.*eta/2016. + 502429./16128.)*chi_s
    
    gamma = (732985./2268. - 24260.*eta/81. - 340.*eta**2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*delta*chi_a
    
    
    # amplitude correction (w/o TH)
    
    
    A = wfn.A0(m1,m2,DL)
    
    
    a0 = 1.
    
    a1 = 0
    
    a2 = 11.*eta/8. + 743./672.
    
    a3 = beta/2. - 2.*PI
    
    a4 = 1379.*eta**2/1152. + 18913.*eta/16128. + 7266251./8128512. - sigma/2.
    
    a5 = 57.*PI*eta/16. - 4757.*PI/1344. + epsilon
    
    a6 = 856.*GammaE/105. + 67999.*eta**3/82944. - 1041557.*eta**2/238048. - 451.*PI**2*eta/96. + 10.*PI**2/3. + 3526813753.*eta/27869184. - 29342493702821./500716339200. + 856.*np.log(4*v)/105.
    
    a7 = -1349.*PI*eta**2/24192. - 72221.*PI*eta/24192. - 5111593.*PI/2709504.
    
    # 4.5PN phasing (point particle limit)
    
    p0 = 1.

    p1 = 0

    p2 = (3715./756. + (55.*eta)/9.)

    p3 = (-16.*PI + (113.*delta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s)

    p4 = (15293365./508032. + (27145.*eta)/504.+ (3085.*eta**2)/72. + (-405./8. + 200.*eta)*chi_a**2 - (405.*delta*chi_a*chi_s)/4. + (-405./8. + (5.*eta)/2.)*chi_s**2)


    p5 = (38645.*PI/756. - 65.*PI*eta/9. - gamma)

    p5L = (38645.*PI/756. - 65.*PI*eta/9. - gamma)*3*log(v/vlso)

    p6 = (11583231236531./4694215680. - 640./3.*PI**2 - 6848./21.*GammaE + eta*(-15737765635./3048192. + 2255./12.*PI**2) + eta*eta*76055./1728. - eta*eta*eta*127825./1296. \
         - (6848./21.)*log(4.) + PI*(2270.*delta*chi_a/3. + (2270./3. - 520.*eta)*chi_s) + (75515./144. - 8225.*eta/18.)*delta*chi_a*chi_s \
         + (75515./288. - 263245.*eta/252. - 480.*eta**2)*chi_a**2 + (75515./288. - 232415.*eta/504. + 1255.*eta**2/9.)*chi_s**2)

    p6L = -(6848./21.)*log(v)

    p7 = (((77096675.*PI)/254016. + (378515.*PI*eta)/1512.- (74045.*PI*eta**2)/756. + (-25150083775./3048192. + (10566655595.*eta)/762048. - (1042165.*eta**2)/3024. + (5345.*eta**3)/36.
         + (14585./8. - 7270.*eta + 80.*eta**2)*chi_a**2)*chi_s + (14585./24. - (475.*eta)/6. + (100.*eta**2)/3.)*chi_s**3 + delta*((-25150083775./3048192.
         + (26804935.*eta)/6048. - (1985.*eta**2)/48.)*chi_a + (14585./24. - 2380.*eta)*chi_a**3 + (14585./8. - (215.*eta)/2.)*chi_a*chi_s**2)))

    # 4PN + 4.5PN nonspinning contribution (arXiv:2304.11185)

    p8L = (-2550713843998885153./276808510218240. + 90490.*PI**2/189. + 36812.*GammaE/63. + 1011020.*log(2.)/1323. + 78975.*log(3.)/196. + 18406.*log(v)/63. \
            +(680712846248317./42247941120. - 109295.*PI**2/224. + 3911888.*GammaE/1323. + 9964112.*log(2.)/1323. - 78975.*log(3.)/49. + 1955944.*log(v)/1323.)*eta \
            +(-7510073635./3048192. + 11275.*PI**2/144.)*eta**2 - 1292395.*eta**3/12096. + 5975.*eta**4/96.)*log(v)

    p9 = (105344279473163./18776862720. - 640.*PI**2/3. - 13696.*GammaE/21. - 13696.*log(4*v)/21. \
            + (-1492917260735./134120448. + 2255.*PI**2/6.)*eta + 45293335.*eta**2/127008. + 10323755.*eta**3/199584.)*PI

    phase = 2*f*PI*tc - phic - PI/4. + (3./(128.*v**5*eta))*(p0 + v*p1 + v**2*p2 + v**3*p3+ v**4*p4 + v**5*(p5+p5L) + v**6*(p6+p6L) + v**7*p7 + v**8*p8L + v**9*p9)

    phase += psiTH_new(f,m1,m2,chi1,chi2)
    
    amp = A*f**(-7./6.)*(a0 + v*a1 + v**2*a2 + v**3*a3 + v**4*a4 + v**5*a5 + v**6*a6 + v**7*a7)
    amp += ampTH(f,m1,m2,chi1,chi2,DL)
    
    hp = 0.5*(1+(cos(iota))**2)*amp*(cos(phase) - 1j*sin(phase))
    hc = -1j*cos(iota)*amp*(cos(phase) - 1j*sin(phase))

    if amp_phase_hp_hc == True: return amp, phase, hp, hc
    elif amp_phase_only == True: return amp, phase
    else: return hp, hc
    
    
def inspiral_TH(f_ini=10.,f_final=None, delta_t=None, m1=10, m2=10, chi1=0, chi2=0, H1=1, H2=1, DL=200, tc=0, phic=0, iota=0, amp_phase_only=False, amp_phase_hp_hc=False,approximant='SEOBNRv2',TH1=False,TH2=False,amp_corr=False):
    
    # convert to sec
    Ms = m1+m2
    M = Ms * time_fac
    DLt = DL * time_fac/strain_fac
#     f_isco_KBH = wfn.f_isco_Msolar_KBH(m1,m2,chi1,chi2)
    
#     if f_final is None: f_final = 0.03/M #f_isco_KBH
    if delta_t is None: raise Exception('Specify delta_t')
    
    hps, hcs = waveform.get_td_waveform(approximant=approximant,
                             mass1=m1, mass2=m2, spin1z=chi1, spin2z=chi2, distance = DL, 
                                  delta_t=delta_t, f_lower = f_ini-2, f_final = f_final)
    
    f0 = waveform.utils.frequency_from_polarizations(hps,hcs)
    
    ## Find the TD tapering needed
    index = np.where(f0>f_ini)[0][0]
    tini = hps.sample_times[0]
    tfin = hps.sample_times[index]
    
    
    
    ## Apply tapering to TD data
    hps = waveform.utils.td_taper(hps,tini,tfin)
    hcs = waveform.utils.td_taper(hcs,tini,tfin)
#     if f_final is not None:
#         index2 = np.where(f0>(f_final-2))[0][0]
#         index3 = np.where(f0>f_final)[0][0]
#         tini2 = hps.sample_times[index2]
#         tfin2 = hps.sample_times[index3]
#         hps = waveform.utils.td_taper(hps,tini2,tfin2,side='right')
#         hcs = waveform.utils.td_taper(hcs,tini2,tfin2,side='right')
    
    ## Take FFT
    hpstilde = hps.to_frequencyseries()
    hcstilde = hcs.to_frequencyseries()
    
    f = hpstilde.sample_frequencies
    index0 = np.where(f>f_ini-2)[0][0]
    
    finsp = f[index0:]
    patch = np.zeros(index0)
    
    ## Extract amplitude & phase
    amps = waveform.utils.amplitude_from_frequencyseries(hpstilde)
    amps=np.asarray(amps)
    phases = waveform.utils.phase_from_frequencyseries(hpstilde)
    phases=np.asarray(phases)
    
    ## Compute TH corrections
    if TH1==1: psi_TH = psiTH(finsp,m1,m2,chi1,chi2,H1,H2)
    if TH2==1: psi_TH = psiTH_new(finsp,m1,m2,chi1,chi2)
    if TH1==0 and TH2==0: psi_TH = np.zeros_like(finsp)
    amp_TH = ampTH(finsp,m1,m2,chi1,chi2,DL,H1,H2)
    
    psi_TH = np.concatenate((patch,psi_TH))
    amp_TH = np.concatenate((patch,amp_TH))
    
    psi_TH = types.FrequencySeries(psi_TH,delta_f=hpstilde.delta_f)
    amp_TH = types.FrequencySeries(amp_TH,delta_f=hpstilde.delta_f)
    
    ## Tapering starts from f_ini-2 to f_ini; ends from f_peak to 1.2f_peak
    psi_TH = waveform.utils.fd_taper(psi_TH,f_ini-2,f_ini)
    amp_TH = waveform.utils.fd_taper(amp_TH,f_ini-2,f_ini)
    
    fpeak = wfn.f_peak(m1,m2,chi1,chi2)
    amp_TH = waveform.utils.fd_taper(amp_TH,fpeak,1.2*fpeak,side='right')
    
    if amp_corr==True:
        hptilde = 0.5*(1+np.cos(iota)**2) * (amps+amp_TH) * np.exp(1j*(phases + psi_TH + 2.*np.pi*f*tc + phic))
        hctilde = -1j*np.cos(iota) * (amps+amp_TH) * np.exp(1j*(phases + psi_TH + 2.*np.pi*f*tc + phic)) 
    else:
        hptilde = 0.5*(1+np.cos(iota)**2) * amps * np.exp(1j*(phases + psi_TH + 2.*np.pi*f*tc + phic))
        hctilde = -1j*np.cos(iota) * amps * np.exp(1j*(phases + psi_TH + 2.*np.pi*f*tc + phic))
    
    hptilde = types.FrequencySeries(hptilde,delta_f=hpstilde.delta_f)
    hctilde = types.FrequencySeries(hctilde,delta_f=hpstilde.delta_f)
    
    hp = hptilde.to_timeseries(delta_t=delta_t)
    hc = hctilde.to_timeseries(delta_t=delta_t)
    
    hp.start_time = 0
    hc.start_time = 0
    
    return hp, hc


def inspiral_TH_fd(f_ini=10.,f_final=None, delta_t=None, m1=10, m2=10, chi1=0, chi2=0, H1=1, H2=1, DL=200, tc=0, phic=0, iota=0, amp_phase_only=False, amp_phase_hp_hc=False,approximant='SEOBNRv2',TH1=True,TH2=False,amp_corr=True):
    
    # convert to sec
    Ms = m1+m2
    M = Ms * time_fac
    DLt = DL * time_fac/strain_fac
#     f_isco_KBH = wfn.f_isco_Msolar_KBH(m1,m2,chi1,chi2)
    
#     if f_final is None: f_final = 0.03/M #f_isco_KBH
    if delta_t is None: raise Exception('Specify delta_t')
    
    hps, hcs = waveform.get_td_waveform(approximant=approximant,
                             mass1=m1, mass2=m2, spin1z=chi1, spin2z=chi2, distance = DL, 
                                  delta_t=delta_t, f_lower = f_ini-2, f_final = f_final)
    
    f0 = waveform.utils.frequency_from_polarizations(hps,hcs)
    
    ## Find the TD tapering needed
    index = np.where(f0>f_ini)[0][0]
    tini = hps.sample_times[0]
    tfin = hps.sample_times[index]
    
    
    
    ## Apply tapering to TD data
    hps = waveform.utils.td_taper(hps,tini,tfin)
    hcs = waveform.utils.td_taper(hcs,tini,tfin)
    if f_final is not None:
        index2 = np.where(f0>(f_final-2))[0][0]
        index3 = np.where(f0>f_final)[0][0]
        tini2 = hps.sample_times[index2]
        tfin2 = hps.sample_times[index3]
        hps = waveform.utils.td_taper(hps,tini2,tfin2,side='right')
        hcs = waveform.utils.td_taper(hcs,tini2,tfin2,side='right')
    
    ## Take FFT
    hpstilde = hps.to_frequencyseries()
    hcstilde = hcs.to_frequencyseries()
    
    f = hpstilde.sample_frequencies
    index0 = np.where(f>f_ini-2)[0][0]
    
    finsp = f[index0:]
    patch = np.zeros(index0)
    
    ## Extract amplitude & phase
    amps = waveform.utils.amplitude_from_frequencyseries(hpstilde)
    amps=np.asarray(amps)
    phases = waveform.utils.phase_from_frequencyseries(hpstilde)
    phases=np.asarray(phases)
    
    ## Compute TH corrections
    if TH1==True: psi_TH = psiTH(finsp,m1,m2,chi1,chi2,H1,H2)
    if TH2==True: psi_TH = psiTH_new(finsp,m1,m2,chi1,chi2,H1,H2)
    else: psi_TH = np.zeros_like(finsp)
    amp_TH = ampTH(finsp,m1,m2,chi1,chi2,DL,H1,H2)
    
    psi_TH = np.concatenate((patch,psi_TH))
    amp_TH = np.concatenate((patch,amp_TH))
    
    psi_TH = types.FrequencySeries(psi_TH,delta_f=hpstilde.delta_f)
    amp_TH = types.FrequencySeries(amp_TH,delta_f=hpstilde.delta_f)
    
    ## Tapering starts from f_ini-2 to f_ini; ends from f_peak to 1.2f_peak
    psi_TH = waveform.utils.fd_taper(psi_TH,f_ini-2,f_ini)
    amp_TH = waveform.utils.fd_taper(amp_TH,f_ini-2,f_ini)
    
    fpeak = wfn.f_peak(m1,m2,chi1,chi2)
    amp_TH = waveform.utils.fd_taper(amp_TH,fpeak,1.2*fpeak,side='right')
    
    ## Add TH corrections
    if TH1 or TH2 is True:
        if amp_corr==True:
            hptilde = 0.5*(1+np.cos(iota)**2) * (amps+amp_TH) * np.exp(1j*(phases + psi_TH + 2.*np.pi*f*tc + phic))
            hctilde = -1j*np.cos(iota) * (amps+amp_TH) * np.exp(1j*(phases + psi_TH + 2.*np.pi*f*tc + phic)) 
        else:
            hptilde = 0.5*(1+np.cos(iota)**2) * amps * np.exp(1j*(phases + psi_TH + 2.*np.pi*f*tc + phic))
            hctilde = -1j*np.cos(iota) * amps * np.exp(1j*(phases + psi_TH + 2.*np.pi*f*tc + phic)) 
    else:
        hptilde = 0.5*(1+np.cos(iota)**2)*hpstilde
        hctilde = -1j*np.cos(iota)*hpstilde
        
    hptilde = types.FrequencySeries(hptilde,delta_f=hpstilde.delta_f)
    hctilde = types.FrequencySeries(hctilde,delta_f=hpstilde.delta_f)
    
    
    return hptilde, hctilde
