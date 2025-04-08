import numpy as np
import inspiral_model as insp 
import imrphenomd_horizon as phor 
import useful_relations as urel
import waveform_functions as wfn
import model_functions as mfn
from scipy.interpolate import interp1d

# beta = [
# 				[  -8287.60083325,  102471.0357223,  -272927.44643436],
#                 [  51760.41259419, -654282.80673584, 1758244.08930711],
#                 [ -108300.75797795,  1395202.69711306, -3777615.90379674],
#                 [  75636.04409168, -992413.78926107, 2706250.44035738]
# 			]


def fit_parameters_from_nr(m1,m2):
	
    beta = [
                [161351.04834113352,	-1531185.526043311,	3633803.558901329],
                [-688223.9302013989,	6520488.603991435,	-15460193.181716902],
                [734430.8757712821,	-6946505.253327953,	16454233.381099897],
                [-38.45357648457703,	236.22824093593755,	-449.9997634938205]
            ]

    eta = m1*m2/(m1+m2)**2
    a = np.zeros(4)
    for i in range(4):
        for j in range(3):
            a[i] += beta[i][j]*eta**j

    return a

def fit_parameters_from_nr_all(m1,m2):
	
    beta = [
                [  -5989.09984752,   73305.34023504, -192223.76459304],
                [  37476.05376404, -470160.45819447, 1245036.30432933],
                [  -78783.61539266,  1008135.98212086, -2690297.36213718],
                [  55360.59921672, -721467.3597681,  1938754.06140394]
            ]

    eta = m1*m2/(m1+m2)**2
    a = np.zeros(4)
    for i in range(4):
        for j in range(3):
            a[i] += beta[i][j]*eta**j

    return a

def Psi_1(f,m1,m2):
    return insp.psiTH_new(f,m1,m2,0,0)

def Psi_2(f,m1,m2,coeff,Mf_transition1=0.02,Mf_transition2=0.03):
    a_func = []
    Mf = (m1+m2)*urel.MTsun*f
    eta = m1*m2/(m1+m2)**2
    v=(np.pi*Mf)**(1/3)
    for i in range(4):
        tmp = coeff[i]*((Mf-np.ones_like(Mf)*Mf_transition1)/(Mf_transition2-Mf_transition1))
        a_func.append(tmp)
    a1 = a_func[0]
    a2 = a_func[1]
    a3 = a_func[2]
    a4 = a_func[3]
    
    psi = (3/(128*eta))*(a1*v**4 + a2*v**5 + a3*v**6 + a4*v**7)
    return Psi_1(f,m1,m2) + psi

def Psi_3(f,m1,m2,coeff):
    eta = m1*m2/(m1+m2)**2
    M = (m1+m2)*urel.MTsun
    v = (np.pi*M*f)**(1./3.)
    psi = (3/(128*eta))*(coeff[0]*v**4 + coeff[1]*v**5 + coeff[2]*v**6 + coeff[3]*v**7)
    return Psi_1(f,m1,m2) + psi



def psiTH_from_nr(f,m1,m2,intermediate=0,Mf_transition1=0.022,Mf_transition2=0.032,PN_order=6.):
	
    eta = m1*m2/(m1+m2)**2
    Mf = (m1+m2)*urel.MTsun*f
    v = (np.pi*Mf)**(1./3.)
    
    a = fit_parameters_from_nr_all(m1,m2)
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    b1 = a[3]
    
    if intermediate:
        a_func = []
        for i in range(4):
            tmp = a[i]*((Mf-np.ones_like(Mf)*Mf_transition1)/(Mf_transition2-Mf_transition1))
            a_func.append(tmp)
    
        a1 = a_func[0]
        a2 = a_func[1]
        a3 = a_func[2]
        b1 = a_func[3]
    
    x1=m1/(m1+m2)
    x2=m2/(m1+m2)
    A = (32./5.)*(x1**4 + x2**4)

    psi = np.zeros(5)

    psi[0] = (25./36.)*A
    # psi[1] = -3*psi[0]
    psi[1] = -(25./16.)*A*(a1 - b1)*eta
    psi[2] = -(5./1344.)*A*(168.*(a2*eta - a1*b1*eta**2 + b1**2*eta**2) + 995. + 952.*eta)
    psi[3] = (25./12096.)*A*eta*(-168.*a3-a1*(995.+952*eta+168*b1**2*eta**2)\
                                +b1*(995.+56.*(17+3*a2)*eta+168.*b1**2*eta**2)+1344.*np.pi/eta)
    psi[4] = -(25./12644352.)*A*(3408051. \
                                 +56.*eta*(-16128.*np.pi*a1+11940.*a2+16128.*np.pi*b1+110801.)\
                                +112.*eta**2*(-5970.*a1*b1+5712.*a2-1008.*a3*b1+5970.*b1**2+24745.)\
                                +37632*b1*eta**3*((3*a2+17)*b1-17*a1)\
                                -112896*b1**3*eta**4*(a1-b1))

    func = np.zeros_like(f)

    v_power = int(2*PN_order)
     
    for i in range(v_power-7):
        power = float(i)+8.
        if i==0: func += psi[i]*v**power*(1-3.*np.log(v))
        else: func += psi[i]*v**power
    
    # psi_8 = (25./36.)*A
    # psi_8_l = -3.*psi_8
    # psi_9 = -(25./16.)*A*(a1 - b1)
    # psi_10 = -(5./1344.)*A*(168.*(a2*eta - a1*b1*eta**2 + b1**2*eta**2) + 995. + 952.*eta)
    # psi_11 = (25./12096.)*A*eta*(-168.*a3-a1*(995.+952*eta+168*b1**2*eta**2)\
    #                              +b1*(995.+56.*(17+3*a2)*eta+168.*b1**2*eta**2)+1344.*np.pi/eta)
    # psi_12 = -(25./12644352.)*A*(3408051. \
    #                              +56.*eta*(-16128.*np.pi*a1+11940.*a2+16128.*np.pi*b1+110801.)\
    #                             +112.*eta**2*(-5970.*a1*b1+5712.*a2-1008.*a3*b1+5970.*b1**2+24745.)\
    #                             +37632*b1*eta**3*((3*a2+17)*b1-17*a1)\
    #                             -112896*b1**3*eta**4*(a1-b1))
    # psi_12 = -(25./113799168.)*A*(30400075.-1016064.*a3*a4\
    # 			+6017760.*a4**2+1016064*a4**4+8128512.*a4*np.pi\
    # 			+25702488.*eta+5757696.*a4**2*eta+17477712.*eta**2\
    # 			+6048.*a2*(995.+168.*a4**2+952.*eta)\
    # 			-6048.*a1*(995.*a4+168.*a4**3+1344.*np.pi+952.*a4*eta))
    
    # psi_10 = -(5./1344.)*A*(168.*(a2 - a1*a4 + a4**2))
    # psi_11 = -(25./12096.)*A*((995.+952.*eta+168.*eta**2)*(a1-a4)+168*(a3-a2*a4))
    # psi_12 = -(25./113799168.)*A*(-1016064.*a3*a4\
    #             +6017760.*a4**2+1016064*a4**4+8128512.*a4*np.pi\
    #             +5757696.*a4**2*eta\
    #             +6048.*a2*(995.+168.*a4**2+952.*eta)\
    #             -6048.*a1*(995.*a4+168.*a4**3+952.*a4*eta))
    
    # psi_TH = (3./(128.*eta*v**5))*(psi_8*v**8 + psi_8_l*v**8*np.log(v) + psi_9*v**9 + psi_10*v**10 + psi_11*v**11 + psi_12*v**12)
    psi_TH = (3./(128.*eta*v**5))*func
    
    return psi_TH


def imr_tidal_heating_approximant(f,m1,m2,Mf_transition1=0.022,Mf_transition2=0.032):
	
	# psiTH_pn = insp.psiTH_new(f,m1,m2,0,0) #spins are zero

	# psiTH_intermediate = psiTH_from_nr(f,m1,m2,intermediate=1,Mf_transition1=Mf_transition1,Mf_transition2=Mf_transition2)

	# psiTH_nr = psiTH_from_nr(f,m1,m2)

    coeff = fit_parameters_from_nr_all(m1,m2)
    
    psiTH_pn = Psi_1(f,m1,m2) #spins are zero
    
    psiTH_intermediate = Psi_2(f,m1,m2,coeff,Mf_transition1=Mf_transition1,Mf_transition2=Mf_transition2)
    
    psiTH_nr = Psi_3(f,m1,m2,coeff)
    
    Mf = (m1+m2)*urel.MTsun*f
		
    psi_prime_nr = urel.dydx(Mf,psiTH_nr)
    psi_prime_intermediate = urel.dydx(Mf,psiTH_intermediate)
    psi_prime_pn = urel.dydx(Mf,psiTH_pn)
    
    phase_prime_interpolate_nr = interp1d(Mf,psi_prime_nr)
    phase_prime_interpolate_intermediate = interp1d(Mf,psi_prime_intermediate)
    phase_prime_interpolate_pn = interp1d(Mf,psi_prime_pn)
    
    tc1 =  phase_prime_interpolate_intermediate(Mf_transition1) - phase_prime_interpolate_pn(Mf_transition1)
    
    psiTH_pn += Mf*tc1
    
    phase_interpolate_intermediate = interp1d(Mf,psiTH_intermediate)
    phase_interpolate_pn = interp1d(Mf,psiTH_pn)
    phi1 = phase_interpolate_intermediate(Mf_transition1) - phase_interpolate_pn(Mf_transition1)
    
    psiTH_pn += phi1
    
    
    tc2 =  phase_prime_interpolate_nr(Mf_transition2) - phase_prime_interpolate_intermediate(Mf_transition2)
    
    psiTH_nr -= Mf*tc2
    
    phase_interpolate_nr = interp1d(Mf,psiTH_nr)
    phase_interpolate_intermediate = interp1d(Mf,psiTH_intermediate)
    
    phi2 = phase_interpolate_nr(Mf_transition2) - phase_interpolate_intermediate(Mf_transition2)
    psiTH_nr -= phi2
    
    
    theta1plus, theta1minus = mfn.theta_pm(Mf,Mf_transition1)
    theta2plus, theta2minus = mfn.theta_pm(Mf,Mf_transition2)
    
    imr_phase = psiTH_pn*theta1minus + theta1plus*psiTH_intermediate*theta2minus + theta2plus*psiTH_nr
    
    return imr_phase





	