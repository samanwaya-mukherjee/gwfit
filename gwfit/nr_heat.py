import numpy as np
import inspiral_model as insp 
import imrphenomd_horizon as phor 
import useful_relations as urel

beta = [
		[161351.04834113352,	-1531185.526043311,	3633803.558901329],
		[-688223.9302013989,	6520488.603991435,	-15460193.181716902],
		[734430.8757712821,	-6946505.253327953,	16454233.381099897],
		[-38.45357648457703,	236.22824093593755,	-449.9997634938205]
		]


def fit_parameters_from_nr(m1,m2):
	eta = m1*m2/(m1+m2)**2
	a = np.zeros(4)
	for i in range(4):
		for j in range(3):
			a[i] += beta[i][j]*eta**j

	return a



def psiTH_from_nr(f,m1,m2):
	
	eta = m1*m2/(m1+m2)**2
	v = (np.pi*(m1+m2)*urel.MTsun*f)**(1./3.)
	
	a = fit_parameters_from_nr(m1,m2)
	a1 = a[0]
	a2 = a[1]
	a3 = a[2]
	a4 = a[3]
	
	x1=m1/(m1+m2)
	x2=m2/(m1+m2)
	A = (32./5.)*(x1**4 + x2**4)

	psi_8 = (25./36.)*A*v**8
	psi_8_l = -3.*psi_8
	psi_9 = -(25./16.)*A*(a1 - a4)
	psi_10 = -(5./1344.)*A*(168.*(a2 - a1*a4 + a4**2) + 995. + 952.*eta)
	psi_11 = -(25./12096.)*A*((995.+952.*eta+168.*eta**2)*(a1-a4)+168*(a3-a2*a4)-1344.*np.pi)
	psi_12 = -(25./113799168.)*A*(30400075.-1016064.*a3*a4\
				+6017760.*a4**2+1016064*a4**4+8128512.*a4*np.pi\
				+25702488.*eta+5757696.*a4**2*eta+17477712.*eta**2\
				+6048.*a2*(995.+168.*a4**2+952.*eta)\
				-6048.*a1*(995.*a4+168.*a4**3+1344.*np.pi+952.*a4*eta))

	psi_TH = (3./(128.*eta*v**5))*(psi_8*v**8 + psi_8_l*v**8*np.log(v) + psi_9*v**9 + psi_10*v**10 + psi_11*v**11 + psi_12*v**12)

	return psi_TH


# def imr_heating_approximant(f,m1,m2):

	