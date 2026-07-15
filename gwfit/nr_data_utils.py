# Copyright (c) 2026, Samanwaya Mukherjee

import sxs
import numpy as np 
import constants as const 
from scipy.interpolate import interp1d

def filter_simulations(
    df,
    *,
    object_type="BHBH",
    q=(None, None),
    chi1=(None, None),
    chi2=(None, None),
    eccentricity=(None, 1e-3),
    number_of_orbits=(27, None),
    spin_xy_tol=None,
    allow_deprecated=False,
):
    """
    Filter an SXS simulation dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        SXS simulations dataframe.

    q : tuple(float | None, float | None)
        Allowed range of reference mass ratio.

    chi1, chi2 : tuple(float | None, float | None)
        Allowed ranges of dimensionless spin magnitudes.

    eccentricity : tuple(float | None, float | None)
        Allowed range of reference eccentricity.

    number_of_orbits : tuple(float | None, float | None)
        Allowed range of inspiral orbits.

    spin_xy_tol : float | None
        If not None, require spins to be aligned/anti-aligned with the z-axis within this tolerance.

    allow_deprecated : bool
        Include deprecated simulations.
    """

    def in_range(x, bounds):
        lower, upper = bounds
        mask = np.ones(len(x), dtype=bool)
        if lower is not None:
            mask &= x >= lower
        if upper is not None:
            mask &= x <= upper
        return mask

    spin1 = np.vstack(df["reference_dimensionless_spin1"])
    spin2 = np.vstack(df["reference_dimensionless_spin2"])

    chi1_mag = np.linalg.norm(spin1, axis=1)
    chi2_mag = np.linalg.norm(spin2, axis=1)

    mask = (
        (df["object_types"] == object_type)
        & in_range(df["reference_mass_ratio"].to_numpy(), q)
        & in_range(df["reference_eccentricity"].to_numpy(), eccentricity)
        & in_range(df["number_of_orbits"].to_numpy(), number_of_orbits)
        & in_range(chi1_mag, chi1)
        & in_range(chi2_mag, chi2)
    )

    if spin_xy_tol is not None:
        mask &= (
            (np.abs(spin1[:, 0]) < spin_xy_tol)
            & (np.abs(spin1[:, 1]) < spin_xy_tol)
            & (np.abs(spin2[:, 0]) < spin_xy_tol)
            & (np.abs(spin2[:, 1]) < spin_xy_tol)
        )

    if not allow_deprecated:
        mask &= ~df["deprecated"].to_numpy()

    return df[mask]

def match_time_grid(source_time, source_data, target_time, kind="cubic"):
    """
    Interpolate source_data onto the portion of target_time that overlaps
    the source_time range.

    Returns
    -------
    matched_time : ndarray
        Portion of target_time lying within source_time.
    matched_data : ndarray
        source_data interpolated onto matched_time.
    """

    mask = (
        (target_time >= source_time[0]) &
        (target_time <= source_time[-1])
    )

    matched_time = target_time[mask]

    interp = interp1d(source_time, source_data, kind=kind)

    return matched_time, interp(matched_time)

def resample(time, data, delta_t=None, sampling="coarse", kind="cubic"):
    """
    Resample a time series onto a uniform grid.

    Parameters
    ----------
    time : ndarray
        Monotonically increasing time array.
    data : ndarray
        Data values (real or complex).
    delta_t : float, optional
        Desired time spacing.
    sampling : {"coarse", "fine"}
        Used only if delta_t is None.
    kind : str
        Interpolation kind passed to interp1d.

    Returns
    -------
    time_resampled, data_resampled
    """
    if delta_t is None:
        dt = np.diff(time)
        if sampling == "fine":
            delta_t = dt.min()
        elif sampling == "coarse":
            delta_t = dt.max()
        else:
            raise ValueError("sampling must be 'fine' or 'coarse'.")

    interp = interp1d(time, data, kind=kind)
    time_resampled = np.arange(time[0], time[-1], delta_t)

    return time_resampled, interp(time_resampled)

class Horizon:

    def __init__(self, parent):
        self.parent = parent
        self.metadata = parent.metadata
        self.horizons = parent.sim.horizons

    def component_christodoulou_masses(self, 
                         times_to_interpolate=None,
                         delta_t=None,
                         ):
        massA = self.horizons.A.christodoulou_mass * self.parent.mass_scale
        massB = self.horizons.B.christodoulou_mass * self.parent.mass_scale
        source_time = self.horizons.A.time * self.parent.time_scale

        time_grid = source_time.copy()
        
        if times_to_interpolate is not None:
            time_grid, massA = match_time_grid(source_time, massA, times_to_interpolate)
            _, massB, _ = match_time_grid(source_time, massB, times_to_interpolate)
        else:
            time_grid, massA = resample(time_grid, massA, delta_t=delta_t)
            _, massB = resample(time_grid, massB, delta_t=delta_t)
        
        return time_grid, massA, massB

    def common_christodoulou_mass(self, 
                         times_to_interpolate=None,
                         delta_t=None,
                         ):
        massC = self.horizons.C.christodoulou_mass * self.parent.mass_scale
        source_time = self.horizons.C.time * self.parent.time_scale

        time_grid = source_time.copy()
        
        if times_to_interpolate is not None:
            time_grid, massC = match_time_grid(source_time, massC, times_to_interpolate, delta_t=delta_t)
        else:
            time_grid, massC = resample(time_grid, massC, delta_t=delta_t)
        
        return time_grid, massC

    def component_horizon_fluxes(self,
                                 times_to_interpolate=None,
                                 delta_t=None):
        if times_to_interpolate is not None:
            time, massA, massB = self.component_christodoulou_masses(times_to_interpolate=times_to_interpolate, delta_t=delta_t)
            
        else:
            time, massA, massB = self.component_christodoulou_masses(delta_t=delta_t)
        
        fluxA = np.gradient(massA, time)
        fluxB = np.gradient(massB, time)
        return time, fluxA, fluxB
    
    def common_horizon_flux(self,
                              times_to_interpolate=None,
                              delta_t=None):
        time, massC = self.common_christodoulou_mass(times_to_interpolate=times_to_interpolate, delta_t=delta_t)
        fluxC = np.gradient(massC, time)
        return time, fluxC

    def component_areal_masses(self, 
                         times_to_interpolate=None,
                         scale=True,
                         delta_t=None,
                         ):
        massA = self.horizons.A.areal_mass
        massB = self.horizons.B.areal_mass
        source_time = self.horizons.A.time

        time_grid = source_time.copy()
        
        if times_to_interpolate is not None:
            time_grid, massA = match_time_grid(source_time, massA, times_to_interpolate, delta_t=delta_t)
            _, massB = match_time_grid(source_time, massB, times_to_interpolate, delta_t=delta_t)
        else:
            time_grid, massA = resample(time_grid, massA, delta_t=delta_t)
            _, massB = resample(time_grid, massB, delta_t=delta_t)

        if scale:
            time_grid *= self.parent.time_scale
            massA *= self.parent.mass_scale
            massB *= self.parent.mass_scale

        return time_grid, massA, massB
    
    def common_areal_mass(self, 
                         times_to_interpolate=None,
                         scale=True,
                         delta_t=None
                         ):
        massC = self.horizons.C.areal_mass
        source_time = self.horizons.C.time

        time_grid = source_time.copy()
        
        if times_to_interpolate is not None:
            time_grid, massC = match_time_grid(source_time, massC, times_to_interpolate, delta_t=delta_t)
            return time_grid, massC
        if scale:
            time_grid *= self.parent.time_scale
            massC *= self.parent.mass_scale
            return time_grid, massC

    def component_areas(self, 
                         times_to_interpolate=None,
                         delta_t=None
                         ):
        time, massA, massB = self.component_areal_masses(times_to_interpolate=times_to_interpolate,delta_t=delta_t,scale=False)
        areaA = 16 * np.pi * massA**2 * self.parent.area_scale
        areaB = 16 * np.pi * massB**2 * self.parent.area_scale
        time *= self.parent.time_scale

        return time, areaA, areaB
    
    def common_area(self, 
                         times_to_interpolate=None,
                         delta_t=None
                         ):
        time, massC = self.common_areal_mass(times_to_interpolate=times_to_interpolate,delta_t=delta_t,scale=False)
        areaC = 16 * np.pi * massC**2 * self.parent.area_scale
        time *= self.parent.time_scale

        return time, areaC
    
    def component_entropies(self, 
                         times_to_interpolate=None,
                         ):
        time, areaA, areaB = self.component_areas(times_to_interpolate=times_to_interpolate)
        entropyA = areaA / 4.0 * self.parent.entropy_scale
        entropyB = areaB / 4.0 * self.parent.entropy_scale

        return time, entropyA , entropyB 
    
    def common_entropy(self, 
                         times_to_interpolate=None,
                         ):
        time, areaC = self.common_area(times_to_interpolate=times_to_interpolate)
        entropyC = areaC / 4.0 * self.parent.entropy_scale
        return time, entropyC 

    def component_chis(self, 
                         times_to_interpolate=None,
                         delta_t=None,
                         ):
        chiA = self.horizons.A.chi_inertial_mag
        chiB = self.horizons.B.chi_inertial_mag
        source_time = self.horizons.A.time

        time_grid = source_time.copy()
        
        if times_to_interpolate is not None:
            time_grid, chiA = match_time_grid(source_time, chiA, times_to_interpolate)
            _, chiB = match_time_grid(source_time, chiB, times_to_interpolate)
        else:
            time_grid, chiA = resample(time_grid, chiA, delta_t=delta_t)
            _, chiB = resample(time_grid, chiB, delta_t=delta_t)
        
        return time_grid * self.parent.time_scale, chiA , chiB

    def common_chi(self, 
                    times_to_interpolate=None,
                    delta_t=None
                    ):
        chiC = self.horizons.C.chi_inertial_mag
        source_time = self.horizons.C.time

        time_grid = source_time.copy()
        
        if times_to_interpolate is not None:
            time_grid, chiC = match_time_grid(source_time, chiC, times_to_interpolate)
        else:
            time_grid, chiC = resample(time_grid, chiC, delta_t=delta_t)

        return time_grid * self.parent.time_scale, chiC  


class SXSSimulation:

    def __init__(self,
                 simulation_label=None,
                 units="NR",
                 total_mass=1,
                 distance=200,
                 initial_cutoff='reference',
                 final_cutoff='common_horizon',
                 ):
        if simulation_label is None:
            raise ValueError("Please provide an SXS simulation label.")
        self.simulation_label = simulation_label
        self.units = units
        self.Ms = total_mass
        self.MT = self.Ms * const.MTsun
        self.DL = distance
        
        self.sim = sxs.load(simulation_label)
        self.waveform_modes = self.sim.h
        self.metadata = self.sim.metadata
        self.initial_cutoff = initial_cutoff
        self.final_cutoff = final_cutoff

        self.horizon = Horizon(self)

        if units == "NR":
            self.time_scale = 1.0
            self.length_scale = 1.0
            self.mass_scale = 1.0
            self.area_scale = 1.0
            self.angular_momentum_scale = 1.0
            self.strain_scale = 1.0
            self.entropy_scale = 1.0

        elif units == "Geometric":
            self.time_scale = self.MT
            self.length_scale = self.MT
            self.mass_scale = self.MT
            self.area_scale = self.MT**2
            self.angular_momentum_scale = self.MT**2
            self.strain_scale = (self.Ms / self.DL) * const.strain_fac
            self.entropy_scale = 1.0

        elif units == "SI":
            self.time_scale = self.MT
            self.length_scale = const.GNewton * self.Ms * const.Msun / const.cLight**2
            self.mass_scale = self.Ms * const.Msun
            self.area_scale = self.length_scale**2
            self.angular_momentum_scale = (
                            const.GNewton
                            * (self.Ms * const.Msun)**2
                            / const.cLight
                        )
            self.strain_scale = (self.Ms / self.DL) * const.strain_fac
            self.entropy_scale = const.kBoltzmann * const.cLight**3 / (const.GNewton * const.hbar)

        else:
            raise ValueError("units must be 'NR', 'Geometric', or 'SI'.") 
        
    @property
    def remnant_mass(self):
        return self.metadata.remnant_mass * self.mass_scale

    @property
    def remnant_chi(self):
        return self.metadata.remnant_dimensionless_spin

    def _resolve_time(self, cutoff, peak_time=None):
        """
        Convert a cutoff specification into a physical time.

        Parameters
        ----------
        cutoff : None, str, or float
            None      -> no cutoff
            str       -> named event
            float/int -> explicit time
        """

        if cutoff is None:
            return None

        if isinstance(cutoff, (int, float)):
            return float(cutoff)

        if isinstance(cutoff, str):
            lookup = {
                "reference": self.metadata.reference_time * self.time_scale,
                "common_horizon": self.metadata.common_horizon_time * self.time_scale,
                "peak_amplitude": peak_time,
            }

            try:
                return lookup[cutoff]
            except KeyError:
                available = ", ".join(f"'{key}'" for key in lookup)
                raise ValueError(
                    f"Unknown cutoff '{cutoff}'. "
                    f"Available options are: {available}."
                )

        raise TypeError("cutoff must be None, a string, or a number.")

    def waveform_mode(
                self,
                mode=(2, 2),
                delta_t=None,
                sampling="coarse",
                ):
        """
        Get the resampled complex waveform mode h_lm.

        Parameters
        ----------
        mode : tuple(int, int)
            Spherical harmonic mode (l, m).

        delta_t : float, optional
            Time step for resampling. If None, it is chosen according to
            `sampling`.

        sampling : {"coarse", "fine"}
            If `delta_t` is not provided, use either the coarsest or
            finest native time spacing.

        Returns
        -------
        time_resampled : ndarray
            Resampled time array.

        hlm_resampled : ndarray (complex)
            Resampled complex waveform mode.
        """
        hlm = self.waveform_modes[:, self.waveform_modes.index(*mode)] * self.strain_scale
        time = self.waveform_modes.t * self.time_scale

        peak_time = time[np.argmax(np.abs(hlm))]

        t0 = self._resolve_time(self.initial_cutoff, peak_time)
        tf = self._resolve_time(self.final_cutoff, peak_time)

        start = 0 if t0 is None else np.searchsorted(time, t0)
        stop = None if tf is None else np.searchsorted(time, tf)

        time = time[start:stop]
        hlm = hlm[start:stop]

        # Shift merger to t = 0
        # time -= time[np.argmax(np.abs(hlm))]

        time_resampled, hlm_resampled = resample(time, hlm, delta_t=delta_t, sampling=sampling)

        return time_resampled, hlm_resampled
    
    def gw_flux(self,
                delta_t=None,
                ell_min=2,ell_max=8,
                ):
        """
        Compute the gravitational wave flux from the SXS simulation."""
        gw_flux = None
        for ell in range(ell_min,ell_max+1):
            for em in range(-ell,ell+1):
                time, hlm = self.waveform_mode(mode=(ell,em),
                                        delta_t=delta_t)
                if gw_flux is None: 
                    gw_flux = np.zeros_like(time)
                    t_nr = time.copy()
                
                hlm_dot = np.gradient(hlm, time)

                gw_flux += np.abs(hlm_dot)**2                
                                
        gw_flux *= (1./(16.*np.pi)) 

        if self.units == "SI":
            gw_flux *= const.cLight**5 / const.GNewton  # Convert to Watts
        
        return t_nr, gw_flux
    
    def mode_phase(
                self,
                mode=(2, 2),
                delta_t=None,
                sampling="coarse",
                ):
        """
        Return the instantaneous phase of a waveform mode.

        Returns
        -------
        time : ndarray
            Resampled time array.

        phase : ndarray
            Instantaneous phase of the (l,m) mode.
        """
        time, hlm = self.waveform_mode(
            mode=mode,
            delta_t=delta_t,
            sampling=sampling,
        )

        phase = - np.unwrap(np.angle(hlm))

        return time, phase

    def mode_frequency(
                self,
                mode=(2, 2),
                delta_t=None,
                sampling="coarse",
                ):
        """
        Return the instantaneous frequency of a waveform mode.

        Returns
        -------
        time : ndarray
            Resampled time array.

        frequency : ndarray
            Instantaneous frequency of the (l,m) mode.
        """
        time, phase = self.mode_phase(
            mode=mode,
            delta_t=delta_t,
            sampling=sampling,
        )

        frequency = np.gradient(phase, time) / (2 * np.pi)

        return time, frequency
    
    def mode_amplitude(
                self,
                mode=(2, 2),
                delta_t=None,
                sampling="coarse",
                ):
        """
        Return the instantaneous amplitude of a waveform mode.

        Returns
        -------
        time : ndarray
            Resampled time array.

        amplitude : ndarray
            Instantaneous amplitude of the (l,m) mode.
        """
        time, hlm = self.waveform_mode(
            mode=mode,
            delta_t=delta_t,
            sampling=sampling,
        )

        amplitude = np.abs(hlm)

        return time, amplitude