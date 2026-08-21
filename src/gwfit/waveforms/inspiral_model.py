import numpy as np
from ..physics.constants import MTsun, time_fac, strain_fac
from . import waveform_functions as wfn
from scipy.special import psi as digamma

## Nonspinning orbital energy function (4PN) and the GW flux function (4.5PN) (arXiv:2304.11185)


##Eq. (4)
def gw_flux(f, m1, m2):
    M = (m1 + m2) * MTsun
    Mf = M * f
    q = m1 / m2
    eta = q / (1 + q) ** 2
    v = (np.pi * Mf) ** (1.0 / 3.0)
    prefactor = (32.0 / 5.0) * eta**2 * v**10
    p_v0 = 1.0
    p_v2 = -1247.0 / 336.0 - (35.0 / 12.0) * eta
    p_v3 = 4 * np.pi
    p_v4 = -44711.0 / 9072.0 + (9271.0 / 504.0) * eta + (65.0 / 18.0) * eta**2
    p_v5 = (-(8191.0 / 672.0) - (583.0 / 24.0) * eta) * np.pi
    p_v6 = (
        (6643739519.0 / 69854400)
        + (16.0 / 3.0) * np.pi**2
        - (1712.0 / 105.0) * np.euler_gamma
        - (856.0 / 105.0) * np.log(16 * v**2)
        + (-(134543.0 / 7776.0) + (41.0 / 48.0) * np.pi**2) * eta
        - (94403.0 / 3024.0)
        - (775.0 / 324.0) * eta**2
    )
    p_v7 = (
        -(16285.0 / 504.0) + (214745.0 / 1728.0) * eta + (193385.0 / 3024.0) * eta**2
    ) * np.pi
    p_v8 = (
        -(323105549467.0 / 3178375200.0)
        + (232597.0 / 4410) * np.euler_gamma
        - (1369.0 / 126.0) * np.pi**2
        + (39931.0 / 294.0) * np.log(2)
        - (47385.0 / 1568.0) * np.log(3)
        + (232597.0 / 8820.0) * np.log(v**2)
        + (
            (-1452202403629.0 / 1466942400.0)
            + (41478.0 / 245.0) * np.euler_gamma
            - (267127.0 / 4608.0) * np.pi**2
            + (479062.0 / 2205.0) * np.log(2)
            + (47385.0 / 392.0) * np.log(3)
            + (20739.0 / 245.0) * np.log(v**2)
        )
        * eta
        + ((1607125.0 / 6804.0) - (3157.0 / 384.0) * np.pi**2) * eta**2
        + (6785.0 / 504.0) * eta**3
        + (5.0 / 6.0) * eta**4
    )
    p_v9 = (
        (265978667519.0 / 745113600.0)
        - (6848.0 / 105.0) * np.euler_gamma
        - (3424.0 / 105.0) * np.log(16 * v**2)
        + ((2062241.0 / 22176.0) + (41.0 / 12.0) * np.pi**2) * eta
        - (133112905.0 / 290304.0) * eta**2
        - (3719141.0 / 38016.0) * eta**3
    ) * np.pi

    return prefactor * (
        p_v0
        + p_v2 * v**2
        + p_v3 * v**3
        + p_v4 * v**4
        + p_v5 * v**5
        + p_v6 * v**6
        + p_v7 * v**7
        + p_v8 * v**8
        + p_v9 * v**9
    )


##Eq. (3)
def E_orb(f, m1, m2):
    M = (m1 + m2) * MTsun
    Mf = M * f
    q = m1 / m2
    eta = q / (1 + q) ** 2
    eta = q / (1 + q) ** 2
    v = (np.pi * Mf) ** (1.0 / 3.0)
    prefactor = -(1 / 2) * eta * M * v**2
    p_v0 = 1.0
    p_v2 = (9 + eta) / 12
    p_v4 = (-27 / 8) + (19 / 8) * eta - (eta**2 / 24)
    p_v6 = (
        -(675 / 64)
        + (34445 / 576 - (205 / 96) * np.pi**2) * eta
        - (155 / 96) * eta**2
        - (35 / 5184) * eta**3
    )
    p_v8 = (
        -3969 / 128
        + (
            -123671 / 5760
            + (9037 / 1536) * np.pi**2
            + (896 / 15) * np.euler_gamma
            + (448 / 15) * np.log(16 * v**2)
        )
        * eta
        + (-498449 / 3456 + (3157 / 576) * np.pi**2) * eta**2
        + (301 / 1728) * eta**3
        + (77 / 31104) * eta**4
    )

    return prefactor * (p_v0 + p_v2 * v**2 + p_v4 * v**4 + p_v6 * v**6 + p_v8 * v**8)


# TaylorF2 with tidal heating --------------------------------------
####################################################################


# Polygamma function with n=0 (digamma) -------
def B2(x, y):
    z = x + 1j * y
    temp = digamma(z)
    return temp.imag


def dmdt(f, m1, m2, chi1, chi2):

    Ms = m1 + m2
    eta = (m1 * m2) / Ms**2
    v = (np.pi * Ms * MTsun * f) ** (1.0 / 3.0)
    dEdt_N = (32.0 / 5.0) * (eta**2) * (v**10)

    dm1dt = (
        dEdt_N
        * (v**5 / 4.0)
        * (m1 / Ms) ** 3
        * (1 + 3 * chi1**2)
        * (-chi1 + 2 * (1 + np.sqrt(1 - chi1**2)) * (m1 / Ms) * v**3)
    )
    dm2dt = (
        dEdt_N
        * (v**5 / 4.0)
        * (m2 / Ms) ** 3
        * (1 + 3 * chi2**2)
        * (-chi2 + 2 * (1 + np.sqrt(1 - chi2**2)) * (m2 / Ms) * v**3)
    )

    return dm1dt, dm2dt


def dmdt_new(f, m1, m2, chi1, chi2):

    Ms = m1 + m2
    eta = (m1 * m2) / Ms**2
    v = (np.pi * Ms * MTsun * f) ** (1.0 / 3.0)

    k1 = np.sqrt(1.0 - chi1**2)
    k2 = np.sqrt(1.0 - chi2**2)
    X1 = m1 / Ms
    X2 = m2 / Ms
    chi_tilde = (2 * X1**2 + 3.0 * eta) * chi1 + (2 * X2**2 + 3.0 * eta) * chi2
    chi_bar1 = X1 * (1.0 + X1) * chi1 + 3.0 * eta * chi2
    chi_bar2 = X2 * (1.0 + X2) * chi2 + 3.0 * eta * chi1

    B21 = B2(3.0, 2.0 * chi1 / k1)
    B22 = B2(3.0, 2.0 * chi2 / k2)

    omega_H1 = chi1 / (
        2.0 * m1 * MTsun * (1.0 + k1)
    )  ## angular velocity of horizon for BH 1
    omega_H2 = chi2 / (
        2.0 * m2 * MTsun * (1.0 + k2)
    )  ## angular velocity of horizon for BH 2

    omega1 = (v**3 / (Ms * MTsun)) * (
        1.0 - eta * v**2 + (1.0 / 2.0) * (chi_tilde - chi_bar1) * v**3
    )
    omega2 = (v**3 / (Ms * MTsun)) * (
        1.0 - eta * v**2 + (1.0 / 2.0) * (chi_tilde - chi_bar2) * v**3
    )

    C_x1 = (
        -(16.0 / 5.0)
        * (m1 * MTsun) ** 2
        * (X1**2)
        * eta**2
        * (1.0 + k1)
        * v**12
        * (
            1.0
            + 3.0 * chi1**2
            + (1.0 / 4.0)
            * (
                3.0 * (2.0 + chi1**2)
                + 2 * X1 * (1.0 + 3.0 * chi1**2) * (2.0 + 3.0 * X1)
            )
            * v**2
            + v**3
            * (
                (1.0 / 2.0) * (-4.0 + 3.0 * chi1**2) * chi2
                - 2.0 * X1 * (1.0 + 3.0 * chi1**2) * (X1 * (chi1 + chi2) + 4 * B21)
                + X1
                * (
                    -(2.0 / 3.0) * (23.0 + 30.0 * k1) * chi1
                    + (7.0 - 12.0 * k1) * chi1**3
                    + 4.0 * chi2
                    + (9.0 * chi1**2 * chi2 / 2.0)
                )
            )
        )
    )

    C_x2 = (
        -(16.0 / 5.0)
        * (m2 * MTsun) ** 2
        * (X2**2)
        * eta**2
        * (1.0 + k2)
        * v**12
        * (
            1.0
            + 3.0 * chi2**2
            + (1.0 / 4.0)
            * (
                3.0 * (2.0 + chi2**2)
                + 2 * X2 * (1.0 + 3.0 * chi2**2) * (2.0 + 3.0 * X2)
            )
            * v**2
            + v**3
            * (
                (1.0 / 2.0) * (-4.0 + 3.0 * chi2**2) * chi1
                - 2.0 * X2 * (1.0 + 3.0 * chi2**2) * (X2 * (chi1 + chi2) + 4 * B22)
                + X2
                * (
                    -(2.0 / 3.0) * (23.0 + 30.0 * k2) * chi2
                    + (7.0 - 12.0 * k2) * chi2**3
                    + 4.0 * chi1
                    + (9.0 * chi2**2 * chi1 / 2.0)
                )
            )
        )
    )

    dm1dt = omega1 * (omega_H1 - omega1) * C_x1
    dm2dt = omega2 * (omega_H2 - omega2) * C_x2

    return dm1dt, dm2dt


# phase and amplitude corrections due to tidal heating ---------------


def psiTH(f, m1, m2, chi1, chi2, H1, H2):

    m1, m2 = m1 * time_fac, m2 * time_fac
    eta = (m1 * m2) / (m1 + m2) ** 2
    Heff5 = wfn.H_eff5(H1, H2, m1, m2, chi1, chi2)
    Heff8 = wfn.H_eff8(H1, H2, m1, m2, chi1, chi2)
    v = (np.pi * (m1 + m2) * f) ** (1.0 / 3.0)

    psi_so1 = (1 / 6.0) * (-56 * eta - 73 * np.sqrt(1 - 4 * eta) + 73) * chi1
    psi_so2 = (1 / 6.0) * (-56 * eta + 73 * np.sqrt(1 - 4 * eta) + 73) * chi2
    psi_so = psi_so1 + psi_so2
    con = 3.0 / (128.0 * eta) / v**5
    con = np.nan_to_num(con, nan=0)
    term1 = -(10 / 9.0) * (v**5) * Heff5 * (3 * np.log(v) + 1)
    term2 = -(5 / 168.0) * (v**7) * Heff5 * (952 * eta + 995)
    term3 = (5 / 9.0) * (v**8) * (3 * np.log(v) - 1) * (-4 * Heff8 + Heff5 * psi_so)

    heated_phase = con * (term1 + term2 + term3)

    return heated_phase


## ------------------------------------- Dephasing expression from Saketh et al (arXiv:2212.13095) ------------------


# Dephasing (Eqn 5.12 - 5.20 of arXiv: 2212.13095) -----------
def psiTH_new(f, m1, m2, chi1, chi2, v0_v3_term=1):

    m1, m2 = m1 * time_fac, m2 * time_fac
    eta = (m1 * m2) / (m1 + m2) ** 2
    delta = (m1 - m2) / (m1 + m2)
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    k1 = np.sqrt(1.0 - chi1**2)
    k2 = np.sqrt(1.0 - chi2**2)
    k_s = (k1 + k2) / 2.0
    k_a = (k1 - k2) / 2.0
    B21 = B2(3, 2 * chi1 / k1)
    B22 = B2(3, 2 * chi2 / k2)
    B2_s = (B21 + B22) / 2.0
    B2_a = (B21 - B22) / 2.0
    v = (np.pi * (m1 + m2) * f) ** (1.0 / 3.0)
    con = 3.0 / (128.0 * eta) / v**5

    delta_psi_5 = -(10.0 / 9.0) * (
        (1.0 - 3.0 * eta) * chi_s * (1 + 9.0 * chi_a**2 + 3.0 * chi_s**2)
        + delta * (1.0 - eta) * chi_a * (1 + 3.0 * chi_a**2 + 9.0 * chi_s**2)
    )

    delta_psi_5_l = 3.0 * delta_psi_5

    delta_psi_7 = (5.0 / 168.0) * (
        delta
        * chi_a
        * (
            -1667.0
            - 4371.0 * chi_a**2
            - 13113.0 * chi_s**2
            + 616.0 * eta**2 * (1 + 3.0 * chi_a**2 + 9.0 * chi_s**2)
            + 5.0 * eta * (311.0 + 807.0 * chi_a**2 + 2421.0 * chi_s**2)
        )
        + chi_s
        * (
            840.0 * eta**2 * (9.0 * chi_a**2 + 3.0 * chi_s**2 + 1.0)
            + eta * (38331.0 * chi_a**2 + 12777.0 * chi_s**2 + 4889.0)
            - 13113.0 * chi_a**2
            - 4371.0 * chi_s**2
            - 1667.0
        )
    )

    delta_psi_8_a = -(5.0 / 27.0) * (
        144.0 * np.pi * delta * (eta - 1.0) * chi_a**3
        + 48.0 * np.pi * delta * (eta - 1) * chi_a
        + 3.0 * (278.0 * eta**2 - 370.0 * eta + 75.0) * chi_a**4
        + (-36.0 * eta**2 + 213.0 * eta - 67.0) * chi_a**2
        + chi_s
        * (
            -12.0 * delta * (eta**2 + 190.0 * eta - 75.0) * chi_a**3
            + 2.0 * delta * (10.0 * eta**2 + 124.0 * eta - 67.0) * chi_a
            + 432.0 * np.pi * (3.0 * eta - 1) * chi_a**2
            + 48.0 * np.pi * (3.0 * eta - 1)
        )
        + chi_s**2
        * (
            432.0 * np.pi * delta * (eta - 1) * chi_a
            + 90.0 * (36.0 * eta**2 - 62.0 * eta + 15.0) * chi_a**2
            - 172.0 * eta**2
            + 303.0 * eta
            - 67.0
        )
        + chi_s**3
        * (
            12.0 * delta * (21.0 * eta**2 - 130.0 * eta + 75.0) * chi_a
            + 144.0 * np.pi * (3.0 * eta - 1)
        )
        + 3.0 * (82.0 * eta**2 - 250.0 * eta + 75.0) * chi_s**4
        - 12.0 * (2.0 * eta**2 - 4.0 * eta + 1.0)
    )

    delta_psi_8_b = -(20.0 / 9.0) * (
        (delta * (2.0 * eta - 1.0) * k_a + (-1.0 - 2.0 * eta**2 + 4.0 * eta) * k_s)
        * (
            1.0
            + 6.0 * chi_a**4
            + 13.0 * chi_s**2
            + 6.0 * chi_s**4
            + chi_a**2 * (13.0 + 36.0 * chi_s**2)
        )
        - 2.0
        * (k_a * (1.0 - 4.0 * eta + 2.0 * eta**2) + delta * (1.0 - 2.0 * eta) * k_s)
        * chi_a
        * chi_s
        * (13.0 + 12.0 * (chi_a**2 + chi_s**2))
    )

    delta_psi_8_c = (80.0 / 9.0) * (
        B2_s
        * (
            (2.0 * eta**2 - 4.0 * eta + 1.0)
            * chi_s
            * (9.0 * chi_a**2 + 3.0 * chi_s**2 + 1.0)
            - delta
            * (2.0 * eta - 1.0)
            * chi_a
            * (3.0 * chi_a**2 + 9.0 * chi_s**2 + 1.0)
        )
        + B2_a
        * (
            3.0 * (2.0 * eta**2 - 4.0 * eta + 1.0) * chi_a**3
            + 9.0 * delta * (1.0 - 2.0 * eta) * chi_a**2 * chi_s
            + (2.0 * eta**2 - 4.0 * eta + 1.0) * chi_a * (9.0 * chi_s**2 + 1.0)
            - delta * (2.0 * eta - 1.0) * chi_s * (3.0 * chi_s**2 + 1.0)
        )
    )

    delta_psi_8 = delta_psi_8_a + delta_psi_8_b + delta_psi_8_c

    delta_psi_8_l = -3.0 * delta_psi_8

    if v0_v3_term:
        delta_psi = con * (
            delta_psi_5 * v**5
            + delta_psi_5_l * v**5 * np.log(v)
            + delta_psi_7 * v**7
            + delta_psi_8 * v**8
            + delta_psi_8_l * v**8 * np.log(v)
        )
    else:
        delta_psi = con * (
            delta_psi_5_l * v**5 * np.log(v)
            + delta_psi_7 * v**7
            + delta_psi_8_l * v**8 * np.log(v)
        )

    return delta_psi


def ampTH(f, m1, m2, chi1, chi2, DL):

    m1, m2 = m1 * time_fac, m2 * time_fac
    M = m1 + m2
    DLt = DL * time_fac / strain_fac

    eta = (m1 * m2) / (m1 + m2) ** 2
    Heff5 = wfn.H_eff5(1.0, 1.0, m1, m2, chi1, chi2)
    Heff8 = wfn.H_eff8(1.0, 1.0, m1, m2, chi1, chi2)
    v = (np.pi * M * f) ** (1.0 / 3.0)

    A_so_1 = chi1 * (179.0 * (np.sqrt(1 - 4 * eta) - 1) + 148.0 * eta)
    A_so_2 = chi2 * (-179.0 * (np.sqrt(1 - 4 * eta) + 1) + 148.0 * eta)
    A_so = A_so_1 + A_so_2

    term1 = (Heff5 / 8.0) * v**5
    term2 = ((1079.0 / 1792.0) + (103.0 * eta / 192.0)) * Heff5 * v**7
    term3 = -(1.0 / 384.0) * (A_so * Heff5 + 96.0 * (Heff8 - Heff5 * np.pi)) * v**8

    amp = (
        (M ** (5.0 / 6.0))
        * (np.sqrt(5.0 * eta / 24.0))
        * (f ** (-7.0 / 6.0))
        / (DLt * np.pi ** (2.0 / 3.0))
    ) * (term1 + term2 + term3)

    return amp


## HTF2 from PN
########################################
def HTF2(
    f,
    m1,
    m2,
    chi1,
    chi2,
    DL,
    tc,
    phic,
    iota,
    amp_phase_only=False,
    amp_phase_hp_hc=False,
    TH_phase_corr=True,
    TH_phase_corr0=False,
    TH_amp_corr=True,
    same_as_PhenomD=False,
):
    """
    m1,m2 ... in solar mass
    DL ... in mega parsec
    """
    PI = np.pi
    log = np.log
    sin = np.sin
    cos = np.cos
    GammaE = 0.577215664901532

    # convert to sec
    Ms = m1 + m2
    M = Ms * time_fac
    DLt = DL * time_fac / strain_fac

    # get sym and asym chi combinations
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)

    """
    Mc is in sec, e.g., Mc = 10*MTSUN_SI (for 10 solar mass)
    DL is in sec, e.g., DL = 100*1e6*PC_SI/C_SI (for 100 Mpc)
    """

    eta = (m1 * m2) / (m1 + m2) ** 2
    delta = (1.0 - 4.0 * eta) ** 0.5
    v = (PI * M * f) ** (1.0 / 3.0)
    flso = wfn.f_isco(M)
    vlso = (PI * M * flso) ** (1.0 / 3.0)

    beta = (113.0 / 12.0) * (chi_s + delta * chi_a - (76.0 * eta / 113.0) * chi_s)

    sigma = (
        chi_a**2 * ((81.0 / 16.0) - 20 * eta)
        + (81.0 * chi_a * chi_s * delta) / 8.0
        + chi_s**2 * ((81.0 / 16.0) - eta / 4.0)
    )

    epsilon = (502429.0 / 16128.0 - 907.0 * eta / 192.0) * delta * chi_a + (
        5.0 * eta**2 / 48.0 - 73921.0 * eta / 2016.0 + 502429.0 / 16128.0
    ) * chi_s

    gamma = (
        732985.0 / 2268.0 - 24260.0 * eta / 81.0 - 340.0 * eta**2 / 9.0
    ) * chi_s + (732985.0 / 2268.0 + 140.0 * eta / 9.0) * delta * chi_a

    # amplitude correction (w/o TH)

    A = wfn.A0(m1, m2, DL)

    a0 = 1.0

    a1 = 0

    a2 = 11.0 * eta / 8.0 + 743.0 / 672.0

    a3 = beta / 2.0 - 2.0 * PI

    a4 = (
        1379.0 * eta**2 / 1152.0
        + 18913.0 * eta / 16128.0
        + 7266251.0 / 8128512.0
        - sigma / 2.0
    )

    a5 = 57.0 * PI * eta / 16.0 - 4757.0 * PI / 1344.0 + epsilon

    a6 = (
        856.0 * GammaE / 105.0
        + 67999.0 * eta**3 / 82944.0
        - 1041557.0 * eta**2 / 238048.0
        - 451.0 * PI**2 * eta / 96.0
        + 10.0 * PI**2 / 3.0
        + 3526813753.0 * eta / 27869184.0
        - 29342493702821.0 / 500716339200.0
        + 856.0 * np.log(4 * v) / 105.0
    )

    a7 = (
        -1349.0 * PI * eta**2 / 24192.0
        - 72221.0 * PI * eta / 24192.0
        - 5111593.0 * PI / 2709504.0
    )

    # 3.5PN phasing (point particle limit)

    p0 = 1.0

    p1 = 0

    p2 = 3715.0 / 756.0 + (55.0 * eta) / 9.0

    p3 = (
        -16.0 * PI
        + (113.0 * delta * chi_a) / 3.0
        + (113.0 / 3.0 - (76.0 * eta) / 3.0) * chi_s
    )

    p4 = (
        15293365.0 / 508032.0
        + (27145.0 * eta) / 504.0
        + (3085.0 * eta**2) / 72.0
        + (-405.0 / 8.0 + 200.0 * eta) * chi_a**2
        - (405.0 * delta * chi_a * chi_s) / 4.0
        + (-405.0 / 8.0 + (5.0 * eta) / 2.0) * chi_s**2
    )

    p5 = 38645.0 * PI / 756.0 - 65.0 * PI * eta / 9.0 - gamma

    p5L = (38645.0 * PI / 756.0 - 65.0 * PI * eta / 9.0 - gamma) * 3 * log(v / vlso)

    p6 = (
        11583231236531.0 / 4694215680.0
        - 640.0 / 3.0 * PI**2
        - 6848.0 / 21.0 * GammaE
        + eta * (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * PI**2)
        + eta * eta * 76055.0 / 1728.0
        - eta * eta * eta * 127825.0 / 1296.0
        - (6848.0 / 21.0) * log(4.0)
        + PI * (2270.0 * delta * chi_a / 3.0 + (2270.0 / 3.0 - 520.0 * eta) * chi_s)
        + (75515.0 / 144.0 - 8225.0 * eta / 18.0) * delta * chi_a * chi_s
        + (75515.0 / 288.0 - 263245.0 * eta / 252.0 - 480.0 * eta**2) * chi_a**2
        + (75515.0 / 288.0 - 232415.0 * eta / 504.0 + 1255.0 * eta**2 / 9.0) * chi_s**2
    )
    ## PhenomD --------------------------------------------
    p6_2 = (
        11583231236531.0 / 4694215680.0
        - 640.0 / 3.0 * PI**2
        - 6848.0 / 21.0 * GammaE
        + eta * (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * PI**2)
        + eta * eta * 76055.0 / 1728.0
        - eta * eta * eta * 127825.0 / 1296.0
        - (6848.0 / 21.0) * log(4.0)
        + PI * (2270.0 * delta * chi_a / 3.0 + (2270.0 / 3.0 - 520.0 * eta) * chi_s)
        + (75515.0 / 144.0 - 8225.0 * eta / 18.0) * delta * chi_a * chi_s
    )
    ##-----------------------------------------------------

    p6L = -(6848.0 / 21.0) * log(v)

    p7 = (
        (77096675.0 * PI) / 254016.0
        + (378515.0 * PI * eta) / 1512.0
        - (74045.0 * PI * eta**2) / 756.0
        + (
            -25150083775.0 / 3048192.0
            + (10566655595.0 * eta) / 762048.0
            - (1042165.0 * eta**2) / 3024.0
            + (5345.0 * eta**3) / 36.0
            + (14585.0 / 8.0 - 7270.0 * eta + 80.0 * eta**2) * chi_a**2
        )
        * chi_s
        + (14585.0 / 24.0 - (475.0 * eta) / 6.0 + (100.0 * eta**2) / 3.0) * chi_s**3
        + delta
        * (
            (
                -25150083775.0 / 3048192.0
                + (26804935.0 * eta) / 6048.0
                - (1985.0 * eta**2) / 48.0
            )
            * chi_a
            + (14585.0 / 24.0 - 2380.0 * eta) * chi_a**3
            + (14585.0 / 8.0 - (215.0 * eta) / 2.0) * chi_a * chi_s**2
        )
    )

    ## PhenomD --------------------------------------------
    p7_2 = (
        (77096675.0 * PI) / 254016.0
        + (378515.0 * PI * eta) / 1512.0
        - (74045.0 * PI * eta**2) / 756.0
        + (
            -25150083775.0 / 3048192.0
            + (10566655595.0 * eta) / 762048.0
            - (1042165.0 * eta**2) / 3024.0
            + (5345.0 * eta**3) / 36.0
        )
        * chi_s
        + delta
        * (
            (
                -25150083775.0 / 3048192.0
                + (26804935.0 * eta) / 6048.0
                - (1985.0 * eta**2) / 48.0
            )
            * chi_a
        )
    )
    ##-----------------------------------------------------

    phase = (
        2 * f * PI * tc
        - phic
        - PI / 4.0
        + (3.0 / (128.0 * v**5 * eta))
        * (
            p0
            + v * p1
            + v**2 * p2
            + v**3 * p3
            + v**4 * p4
            + v**5 * (p5 + p5L)
            + v**6 * (p6 + p6L)
            + v**7 * p7
        )
    )
    if same_as_PhenomD == True:
        phase = (
            2 * f * PI * tc
            - phic
            - PI / 4.0
            + (3.0 / (128.0 * v**5 * eta))
            * (
                p0
                + v * p1
                + v**2 * p2
                + v**3 * p3
                + v**4 * p4
                + v**5 * (p5 + p5L)
                + v**6 * (p6_2 + p6L)
                + v**7 * p7_2
            )
        )

    if TH_phase_corr == True:
        phase += psiTH_new(f, m1, m2, chi1, chi2)
    if TH_phase_corr0 == True:
        phase += psiTH(f, m1, m2, chi1, chi2, 1.0, 1.0)

    amp = (
        A
        * f ** (-7.0 / 6.0)
        * (
            a0
            + v * a1
            + v**2 * a2
            + v**3 * a3
            + v**4 * a4
            + v**5 * a5
            + v**6 * a6
            + v**7 * a7
        )
    )
    if TH_amp_corr == True:
        amp += ampTH(f, m1, m2, chi1, chi2, DL)

    hp = 0.5 * (1 + (cos(iota)) ** 2) * amp * (cos(phase) - 1j * sin(phase))
    hc = -1j * cos(iota) * amp * (cos(phase) - 1j * sin(phase))

    if amp_phase_hp_hc == True:
        return amp, phase, hp, hc
    elif amp_phase_only == True:
        return amp, phase
    else:
        return hp, hc


## HTF2 with 4.5PN phase (arXiv: 2304.11185)
################################################
def HTF2_45(
    f_ini=10.0,
    f_final=None,
    delta_f=None,
    m1=10,
    m2=10,
    chi1=0,
    chi2=0,
    H1=1,
    H2=1,
    DL=200,
    tc=0,
    phic=0,
    iota=0,
    amp_phase_only=False,
    amp_phase_hp_hc=False,
):

    if f_final is None:
        f_final = wfn.f_isco_Msolar(m1 + m2)
    if delta_f is None:
        raise Exception("Specify delta_f")
    f = np.arange(f_ini, f_final, delta_f)

    PI = np.pi
    log = np.log
    sin = np.sin
    cos = np.cos
    GammaE = 0.577215664901532

    # convert to sec
    Ms = m1 + m2
    M = Ms * time_fac
    DLt = DL * time_fac / strain_fac

    # get sym and asym chi combinations
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)

    """
    Mc is in sec, e.g., Mc = 10*MTSUN_SI (for 10 solar mass)
    DL is in sec, e.g., DL = 100*1e6*PC_SI/C_SI (for 100 Mpc)
    """

    eta = (m1 * m2) / (m1 + m2) ** 2
    delta = (1.0 - 4.0 * eta) ** 0.5
    v = (PI * M * f) ** (1.0 / 3.0)
    flso = wfn.f_isco(M)
    vlso = (PI * M * flso) ** (1.0 / 3.0)

    beta = (113.0 / 12.0) * (chi_s + delta * chi_a - (76.0 * eta / 113.0) * chi_s)

    sigma = (
        chi_a**2 * ((81.0 / 16.0) - 20 * eta)
        + (81.0 * chi_a * chi_s * delta) / 8.0
        + chi_s**2 * ((81.0 / 16.0) - eta / 4.0)
    )

    epsilon = (502429.0 / 16128.0 - 907.0 * eta / 192.0) * delta * chi_a + (
        5.0 * eta**2 / 48.0 - 73921.0 * eta / 2016.0 + 502429.0 / 16128.0
    ) * chi_s

    gamma = (
        732985.0 / 2268.0 - 24260.0 * eta / 81.0 - 340.0 * eta**2 / 9.0
    ) * chi_s + (732985.0 / 2268.0 + 140.0 * eta / 9.0) * delta * chi_a

    # amplitude correction (w/o TH)

    A = wfn.A0(m1, m2, DL)

    a0 = 1.0

    a1 = 0

    a2 = 11.0 * eta / 8.0 + 743.0 / 672.0

    a3 = beta / 2.0 - 2.0 * PI

    a4 = (
        1379.0 * eta**2 / 1152.0
        + 18913.0 * eta / 16128.0
        + 7266251.0 / 8128512.0
        - sigma / 2.0
    )

    a5 = 57.0 * PI * eta / 16.0 - 4757.0 * PI / 1344.0 + epsilon

    a6 = (
        856.0 * GammaE / 105.0
        + 67999.0 * eta**3 / 82944.0
        - 1041557.0 * eta**2 / 238048.0
        - 451.0 * PI**2 * eta / 96.0
        + 10.0 * PI**2 / 3.0
        + 3526813753.0 * eta / 27869184.0
        - 29342493702821.0 / 500716339200.0
        + 856.0 * np.log(4 * v) / 105.0
    )

    a7 = (
        -1349.0 * PI * eta**2 / 24192.0
        - 72221.0 * PI * eta / 24192.0
        - 5111593.0 * PI / 2709504.0
    )

    # 4.5PN phasing (point particle limit)

    p0 = 1.0

    p1 = 0

    p2 = 3715.0 / 756.0 + (55.0 * eta) / 9.0

    p3 = (
        -16.0 * PI
        + (113.0 * delta * chi_a) / 3.0
        + (113.0 / 3.0 - (76.0 * eta) / 3.0) * chi_s
    )

    p4 = (
        15293365.0 / 508032.0
        + (27145.0 * eta) / 504.0
        + (3085.0 * eta**2) / 72.0
        + (-405.0 / 8.0 + 200.0 * eta) * chi_a**2
        - (405.0 * delta * chi_a * chi_s) / 4.0
        + (-405.0 / 8.0 + (5.0 * eta) / 2.0) * chi_s**2
    )

    p5 = 38645.0 * PI / 756.0 - 65.0 * PI * eta / 9.0 - gamma

    p5L = (38645.0 * PI / 756.0 - 65.0 * PI * eta / 9.0 - gamma) * 3 * log(v / vlso)

    p6 = (
        11583231236531.0 / 4694215680.0
        - 640.0 / 3.0 * PI**2
        - 6848.0 / 21.0 * GammaE
        + eta * (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * PI**2)
        + eta * eta * 76055.0 / 1728.0
        - eta * eta * eta * 127825.0 / 1296.0
        - (6848.0 / 21.0) * log(4.0)
        + PI * (2270.0 * delta * chi_a / 3.0 + (2270.0 / 3.0 - 520.0 * eta) * chi_s)
        + (75515.0 / 144.0 - 8225.0 * eta / 18.0) * delta * chi_a * chi_s
        + (75515.0 / 288.0 - 263245.0 * eta / 252.0 - 480.0 * eta**2) * chi_a**2
        + (75515.0 / 288.0 - 232415.0 * eta / 504.0 + 1255.0 * eta**2 / 9.0) * chi_s**2
    )

    p6L = -(6848.0 / 21.0) * log(v)

    p7 = (
        (77096675.0 * PI) / 254016.0
        + (378515.0 * PI * eta) / 1512.0
        - (74045.0 * PI * eta**2) / 756.0
        + (
            -25150083775.0 / 3048192.0
            + (10566655595.0 * eta) / 762048.0
            - (1042165.0 * eta**2) / 3024.0
            + (5345.0 * eta**3) / 36.0
            + (14585.0 / 8.0 - 7270.0 * eta + 80.0 * eta**2) * chi_a**2
        )
        * chi_s
        + (14585.0 / 24.0 - (475.0 * eta) / 6.0 + (100.0 * eta**2) / 3.0) * chi_s**3
        + delta
        * (
            (
                -25150083775.0 / 3048192.0
                + (26804935.0 * eta) / 6048.0
                - (1985.0 * eta**2) / 48.0
            )
            * chi_a
            + (14585.0 / 24.0 - 2380.0 * eta) * chi_a**3
            + (14585.0 / 8.0 - (215.0 * eta) / 2.0) * chi_a * chi_s**2
        )
    )

    # 4PN + 4.5PN nonspinning contribution (arXiv:2304.11185)

    p8L = (
        -2550713843998885153.0 / 276808510218240.0
        + 90490.0 * PI**2 / 189.0
        + 36812.0 * GammaE / 63.0
        + 1011020.0 * log(2.0) / 1323.0
        + 78975.0 * log(3.0) / 196.0
        + 18406.0 * log(v) / 63.0
        + (
            680712846248317.0 / 42247941120.0
            - 109295.0 * PI**2 / 224.0
            + 3911888.0 * GammaE / 1323.0
            + 9964112.0 * log(2.0) / 1323.0
            - 78975.0 * log(3.0) / 49.0
            + 1955944.0 * log(v) / 1323.0
        )
        * eta
        + (-7510073635.0 / 3048192.0 + 11275.0 * PI**2 / 144.0) * eta**2
        - 1292395.0 * eta**3 / 12096.0
        + 5975.0 * eta**4 / 96.0
    ) * log(v)

    p9 = (
        105344279473163.0 / 18776862720.0
        - 640.0 * PI**2 / 3.0
        - 13696.0 * GammaE / 21.0
        - 13696.0 * log(4 * v) / 21.0
        + (-1492917260735.0 / 134120448.0 + 2255.0 * PI**2 / 6.0) * eta
        + 45293335.0 * eta**2 / 127008.0
        + 10323755.0 * eta**3 / 199584.0
    ) * PI

    phase = (
        2 * f * PI * tc
        - phic
        - PI / 4.0
        + (3.0 / (128.0 * v**5 * eta))
        * (
            p0
            + v * p1
            + v**2 * p2
            + v**3 * p3
            + v**4 * p4
            + v**5 * (p5 + p5L)
            + v**6 * (p6 + p6L)
            + v**7 * p7
            + v**8 * p8L
            + v**9 * p9
        )
    )

    phase += psiTH_new(f, m1, m2, chi1, chi2)

    amp = (
        A
        * f ** (-7.0 / 6.0)
        * (
            a0
            + v * a1
            + v**2 * a2
            + v**3 * a3
            + v**4 * a4
            + v**5 * a5
            + v**6 * a6
            + v**7 * a7
        )
    )
    amp += ampTH(f, m1, m2, chi1, chi2, DL)

    hp = 0.5 * (1 + (cos(iota)) ** 2) * amp * (cos(phase) - 1j * sin(phase))
    hc = -1j * cos(iota) * amp * (cos(phase) - 1j * sin(phase))

    if amp_phase_hp_hc == True:
        return amp, phase, hp, hc
    elif amp_phase_only == True:
        return amp, phase
    else:
        return hp, hc
