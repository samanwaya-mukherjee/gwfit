import numpy as np
import inspiral_model as insp
import useful_relations as urel
import constants as const


def fit_parameters_from_nr_dmdt(m1, m2):

    alpha = [
        [-328.05974594, 2210.47296328, -4205.7900661],
        [1510.16178859, -10140.38185907, 19288.62179549],
        [-1768.94820504, 11818.963894, -22445.76331149],
    ]

    eta = m1 * m2 / (m1 + m2) ** 2
    a = np.zeros(3)
    for i in range(3):
        for j in range(3):
            a[i] += alpha[i][j] * eta**j

    return a


def fit_parameters_from_nr_phase(m1, m2):

    beta = [
        [-85.68941085, 6787.85060859, -43705.19843872, 75611.03119954],
        [723.8457672, -57045.8117683, 367908.08785425, -637826.4070581],
        [-2898.27539825, 160880.80193527, -1017873.8686362, 1764975.53017488],
        [3318.41937528, -148686.90933617, 925937.44261813, -1605648.23730778],
    ]

    eta = m1 * m2 / (m1 + m2) ** 2
    b = np.zeros(4)
    for i in range(4):
        for j in range(4):
            b[i] += beta[i][j] * eta**j

    return b


def dmdt_nr(f, m1, m2, coeff):
    dm1dt_pn, dm2dt_pn = insp.dmdt_new(f, m1, m2, 0, 0)
    dmdt_pn = dm1dt_pn + dm2dt_pn
    eta = m1 * m2 / (m1 + m2) ** 2
    M = (m1 + m2) * const.MTsun
    v = (np.pi * M * f) ** (1.0 / 3.0)
    log_dmdt = np.log(dmdt_pn) * (
        1 + eta * (coeff[0] * v + coeff[1] * v**2 + coeff[2] * v**3)
    )
    return np.exp(log_dmdt)


def Psi_1(f, m1, m2):
    return insp.psiTH_new(f, m1, m2, 0, 0)


def Psi_2(f, m1, m2, coeff, Mf_transition1=0.02, Mf_transition2=0.03):
    a_func = []
    Mf = (m1 + m2) * const.MTsun * f
    eta = m1 * m2 / (m1 + m2) ** 2
    v = (np.pi * Mf) ** (1 / 3)
    for i in range(4):
        tmp = coeff[i] * (
            (Mf - np.ones_like(Mf) * Mf_transition1) / (Mf_transition2 - Mf_transition1)
        )
        a_func.append(tmp)
    a1 = a_func[0]
    a2 = a_func[1]
    a3 = a_func[2]
    a4 = a_func[3]

    psi = (3 / (128 * eta)) * (a1 * v**4 + a2 * v**5 + a3 * v**6 + a4 * v**7)
    return Psi_1(f, m1, m2) + psi


def Psi_3(f, m1, m2, coeff):
    eta = m1 * m2 / (m1 + m2) ** 2
    M = (m1 + m2) * const.MTsun
    v = (np.pi * M * f) ** (1.0 / 3.0)
    psi = (3 / (128 * eta)) * (
        coeff[0] * v**4 + coeff[1] * v**5 + coeff[2] * v**6 + coeff[3] * v**7
    )
    return Psi_1(f, m1, m2) + psi


def Psi_fit(f, m1, m2, coeff):
    eta = m1 * m2 / (m1 + m2) ** 2
    M = (m1 + m2) * const.MTsun
    v = (np.pi * M * f) ** (1.0 / 3.0)
    psi = (3 / (128 * eta)) * (
        coeff[0] * v**4 + coeff[1] * v**5 + coeff[2] * v**6 + coeff[3] * v**7
    )
    return psi


def imr_dmdt(f, m1, m2, Mf_transition1=0.019, Mf_transition2=0.032):
    M = (m1 + m2) * const.MTsun
    dm1dt_pn, dm2dt_pn = insp.dmdt_new(f, m1, m2, 0, 0)
    dmdt_1 = dm1dt_pn + dm2dt_pn
    coeff = fit_parameters_from_nr_dmdt(m1, m2)
    dmdt_3 = dmdt_nr(f, m1, m2, coeff)
    w = urel.planck_taper_window(
        f, t1=Mf_transition1 / M, t2=Mf_transition2 / M, only_left=True
    )
    complete_dmdt = (1.0 - w) * dmdt_1 + w * dmdt_3
    return complete_dmdt


def imr_tidal_heating_approximant(f, m1, m2):
    coeff = fit_parameters_from_nr_phase(m1, m2)
    psi_pn = Psi_1(f, m1, m2)
    psi_pn -= psi_pn[0]
    delta_psi = Psi_fit(f, m1, m2, coeff)
    delta_psi -= delta_psi[0]
    psiTH = psi_pn + delta_psi
    return psiTH
