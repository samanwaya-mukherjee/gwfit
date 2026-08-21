import numpy as np
from ..physics.constants import MTsun, time_fac, strain_fac
from ..waveforms import waveform_functions as wfn
from pycbc import types, waveform, fft
from ..waveforms import inspiral_model as insp


def inspiral_TH(
    f_ini=10.0,
    f_final=None,
    delta_t=None,
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
    approximant="SEOBNRv2",
    TH1=False,
    TH2=False,
    amp_corr=False,
):

    # convert to sec
    Ms = m1 + m2
    M = Ms * time_fac
    DLt = DL * time_fac / strain_fac
    #     f_isco_KBH = wfn.f_isco_Msolar_KBH(m1,m2,chi1,chi2)

    #     if f_final is None: f_final = 0.03/M #f_isco_KBH
    if delta_t is None:
        raise Exception("Specify delta_t")

    hps, hcs = waveform.get_td_waveform(
        approximant=approximant,
        mass1=m1,
        mass2=m2,
        spin1z=chi1,
        spin2z=chi2,
        distance=DL,
        delta_t=delta_t,
        f_lower=f_ini - 2,
        f_final=f_final,
    )

    f0 = waveform.utils.frequency_from_polarizations(hps, hcs)

    ## Find the TD tapering needed
    index = np.where(f0 > f_ini)[0][0]
    tini = hps.sample_times[0]
    tfin = hps.sample_times[index]

    ## Apply tapering to TD data
    hps = waveform.utils.td_taper(hps, tini, tfin)
    hcs = waveform.utils.td_taper(hcs, tini, tfin)
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
    index0 = np.where(f > f_ini - 2)[0][0]

    finsp = f[index0:]
    patch = np.zeros(index0)

    ## Extract amplitude & phase
    amps = waveform.utils.amplitude_from_frequencyseries(hpstilde)
    amps = np.asarray(amps)
    phases = waveform.utils.phase_from_frequencyseries(hpstilde)
    phases = np.asarray(phases)

    ## Compute TH corrections
    if TH1 == 1:
        psi_TH = insp.psiTH(finsp, m1, m2, chi1, chi2, H1, H2)
    if TH2 == 1:
        psi_TH = insp.psiTH_new(finsp, m1, m2, chi1, chi2)
    if TH1 == 0 and TH2 == 0:
        psi_TH = np.zeros_like(finsp)
    amp_TH = insp.ampTH(finsp, m1, m2, chi1, chi2, DL, H1, H2)

    psi_TH = np.concatenate((patch, psi_TH))
    amp_TH = np.concatenate((patch, amp_TH))

    psi_TH = types.FrequencySeries(psi_TH, delta_f=hpstilde.delta_f)
    amp_TH = types.FrequencySeries(amp_TH, delta_f=hpstilde.delta_f)

    ## Tapering starts from f_ini-2 to f_ini; ends from f_peak to 1.2f_peak
    psi_TH = waveform.utils.fd_taper(psi_TH, f_ini - 2, f_ini)
    amp_TH = waveform.utils.fd_taper(amp_TH, f_ini - 2, f_ini)

    fpeak = wfn.f_peak(m1, m2, chi1, chi2)
    amp_TH = waveform.utils.fd_taper(amp_TH, fpeak, 1.2 * fpeak, side="right")

    if amp_corr == True:
        hptilde = (
            0.5
            * (1 + np.cos(iota) ** 2)
            * (amps + amp_TH)
            * np.exp(1j * (phases + psi_TH + 2.0 * np.pi * f * tc + phic))
        )
        hctilde = (
            -1j
            * np.cos(iota)
            * (amps + amp_TH)
            * np.exp(1j * (phases + psi_TH + 2.0 * np.pi * f * tc + phic))
        )
    else:
        hptilde = (
            0.5
            * (1 + np.cos(iota) ** 2)
            * amps
            * np.exp(1j * (phases + psi_TH + 2.0 * np.pi * f * tc + phic))
        )
        hctilde = (
            -1j
            * np.cos(iota)
            * amps
            * np.exp(1j * (phases + psi_TH + 2.0 * np.pi * f * tc + phic))
        )

    hptilde = types.FrequencySeries(hptilde, delta_f=hpstilde.delta_f)
    hctilde = types.FrequencySeries(hctilde, delta_f=hpstilde.delta_f)

    hp = hptilde.to_timeseries(delta_t=delta_t)
    hc = hctilde.to_timeseries(delta_t=delta_t)

    hp.start_time = 0
    hc.start_time = 0

    return hp, hc


def inspiral_TH_fd(
    f_ini=10.0,
    f_final=None,
    delta_t=None,
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
    approximant="SEOBNRv2",
    TH1=True,
    TH2=False,
    amp_corr=True,
):

    # convert to sec
    Ms = m1 + m2
    M = Ms * time_fac
    DLt = DL * time_fac / strain_fac
    #     f_isco_KBH = wfn.f_isco_Msolar_KBH(m1,m2,chi1,chi2)

    #     if f_final is None: f_final = 0.03/M #f_isco_KBH
    if delta_t is None:
        raise Exception("Specify delta_t")

    hps, hcs = waveform.get_td_waveform(
        approximant=approximant,
        mass1=m1,
        mass2=m2,
        spin1z=chi1,
        spin2z=chi2,
        distance=DL,
        delta_t=delta_t,
        f_lower=f_ini - 2,
        f_final=f_final,
    )

    f0 = waveform.utils.frequency_from_polarizations(hps, hcs)

    ## Find the TD tapering needed
    index = np.where(f0 > f_ini)[0][0]
    tini = hps.sample_times[0]
    tfin = hps.sample_times[index]

    ## Apply tapering to TD data
    hps = waveform.utils.td_taper(hps, tini, tfin)
    hcs = waveform.utils.td_taper(hcs, tini, tfin)
    if f_final is not None:
        index2 = np.where(f0 > (f_final - 2))[0][0]
        index3 = np.where(f0 > f_final)[0][0]
        tini2 = hps.sample_times[index2]
        tfin2 = hps.sample_times[index3]
        hps = waveform.utils.td_taper(hps, tini2, tfin2, side="right")
        hcs = waveform.utils.td_taper(hcs, tini2, tfin2, side="right")

    ## Take FFT
    hpstilde = hps.to_frequencyseries()
    hcstilde = hcs.to_frequencyseries()

    f = hpstilde.sample_frequencies
    index0 = np.where(f > f_ini - 2)[0][0]

    finsp = f[index0:]
    patch = np.zeros(index0)

    ## Extract amplitude & phase
    amps = waveform.utils.amplitude_from_frequencyseries(hpstilde)
    amps = np.asarray(amps)
    phases = waveform.utils.phase_from_frequencyseries(hpstilde)
    phases = np.asarray(phases)

    ## Compute TH corrections
    if TH1 == True:
        psi_TH = insp.psiTH(finsp, m1, m2, chi1, chi2, H1, H2)
    if TH2 == True:
        psi_TH = insp.psiTH_new(finsp, m1, m2, chi1, chi2, H1, H2)
    else:
        psi_TH = np.zeros_like(finsp)
    amp_TH = insp.ampTH(finsp, m1, m2, chi1, chi2, DL, H1, H2)

    psi_TH = np.concatenate((patch, psi_TH))
    amp_TH = np.concatenate((patch, amp_TH))

    psi_TH = types.FrequencySeries(psi_TH, delta_f=hpstilde.delta_f)
    amp_TH = types.FrequencySeries(amp_TH, delta_f=hpstilde.delta_f)

    ## Tapering starts from f_ini-2 to f_ini; ends from f_peak to 1.2f_peak
    psi_TH = waveform.utils.fd_taper(psi_TH, f_ini - 2, f_ini)
    amp_TH = waveform.utils.fd_taper(amp_TH, f_ini - 2, f_ini)

    fpeak = wfn.f_peak(m1, m2, chi1, chi2)
    amp_TH = waveform.utils.fd_taper(amp_TH, fpeak, 1.2 * fpeak, side="right")

    ## Add TH corrections
    if TH1 or TH2 is True:
        if amp_corr == True:
            hptilde = (
                0.5
                * (1 + np.cos(iota) ** 2)
                * (amps + amp_TH)
                * np.exp(1j * (phases + psi_TH + 2.0 * np.pi * f * tc + phic))
            )
            hctilde = (
                -1j
                * np.cos(iota)
                * (amps + amp_TH)
                * np.exp(1j * (phases + psi_TH + 2.0 * np.pi * f * tc + phic))
            )
        else:
            hptilde = (
                0.5
                * (1 + np.cos(iota) ** 2)
                * amps
                * np.exp(1j * (phases + psi_TH + 2.0 * np.pi * f * tc + phic))
            )
            hctilde = (
                -1j
                * np.cos(iota)
                * amps
                * np.exp(1j * (phases + psi_TH + 2.0 * np.pi * f * tc + phic))
            )
    else:
        hptilde = 0.5 * (1 + np.cos(iota) ** 2) * hpstilde
        hctilde = -1j * np.cos(iota) * hpstilde

    hptilde = types.FrequencySeries(hptilde, delta_f=hpstilde.delta_f)
    hctilde = types.FrequencySeries(hctilde, delta_f=hpstilde.delta_f)

    return hptilde, hctilde
