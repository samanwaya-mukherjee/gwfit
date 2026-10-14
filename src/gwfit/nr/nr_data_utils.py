# Copyright (c) 2026, Samanwaya Mukherjee

import sxs
import numpy as np
from ..physics import constants as const
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


def save_sxs_data_to_h5(labels, file_path):
    """
    Extracts numerical relativity data from SXS simulations and saves
    it to an HDF5 file.
    """
    import h5py

    with h5py.File(file_path, "w") as f:
        for sim_label in labels:
            sim_prem = SXSSimulation(
                simulation_label=sim_label,
                initial_cutoff="reference",
                final_cutoff="common_horizon",
            )
            print(
                f"Processing {sim_label} (q = {sim_prem.metadata.reference_mass_ratio:.2f})"
            )

            # Extract frequencies and fluxes
            twf, gw_freq = sim_prem.mode_frequency(mode=(2, 2))
            _, gw_flux = sim_prem.gw_flux(ell_max=8)
            t, fA, fB = sim_prem.horizon.component_horizon_fluxes(
                times_to_interpolate=twf
            )

            dMdt = fA + fB
            mask = (twf >= t[0]) & (twf <= t[-1])

            # Create group for current simulation
            group = f.create_group(sim_label)

            # Write compressed datasets
            group.create_dataset("gw_freq", data=gw_freq[mask], compression="gzip")
            group.create_dataset("dMdt", data=dMdt, compression="gzip")
            group.create_dataset("gw_flux", data=gw_flux[mask], compression="gzip")

            # Save mass ratio as attribute
            group.attrs["q"] = sim_prem.metadata.reference_mass_ratio

    print(f"Successfully saved data to {file_path}")


def load_sxs_data_from_h5(labels, file_path):
    """
    Loads saved SXS simulation data from an HDF5 file into 4 dictionaries.
    """
    import h5py

    freq_data = {}
    dMdt_data = {}
    gw_flux_data = {}
    q_values = {}

    with h5py.File(file_path, "r") as f:
        for sim_label in labels:
            group = f[sim_label]

            freq_data[sim_label] = group["gw_freq"][:]
            dMdt_data[sim_label] = group["dMdt"][:]
            gw_flux_data[sim_label] = group["gw_flux"][:]
            q_values[sim_label] = group.attrs["q"]

    return freq_data, dMdt_data, gw_flux_data, q_values


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

    mask = (target_time >= source_time[0]) & (target_time <= source_time[-1])

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
    """
    Apparent-horizon data for an SXS binary-black-hole simulation.

    Wraps the ``horizons`` object of an :class:`SXSSimulation` (as loaded by
    the ``sxs`` package), exposing the trajectories of the two initial
    black-hole horizons ("component" quantities, labelled A and B) and the
    post-merger remnant horizon ("common" quantities, labelled C). All
    quantities are returned either on their native (possibly irregular) NR
    time sampling resampled to a uniform grid via `delta_t`, or interpolated
    onto an arbitrary set of times via `times_to_interpolate`, and are scaled
    according to the unit system (`"NR"`, `"Geometric"`, or `"SI"`) of the
    parent :class:`SXSSimulation`.

    Parameters
    ----------
    parent : SXSSimulation
        The simulation this horizon data belongs to. Used both to access the
        raw ``sxs`` horizons object (`parent.sim.horizons`) and to read the
        unit-dependent scale factors (`time_scale`, `mass_scale`,
        `area_scale`, `entropy_scale`, ...) applied to the quantities
        returned by this class's methods.

    Attributes
    ----------
    parent : SXSSimulation
        Back-reference to the owning simulation wrapper.

    metadata : sxs.metadata.Metadata
        Alias of `parent.metadata`.

    horizons : sxs.horizons.HorizonQuantities
        The raw apparent-horizon data from the ``sxs`` package, with
        sub-objects `.A`, `.B` (the two initial black holes) and `.C`
        (the common/remnant horizon formed after merger).

    Notes
    -----
    Methods come in "component" (returning both A and B quantities) and
    "common" (returning the C quantity) variants, covering Christodoulou
    mass, horizon mass flux, areal mass, area, entropy (area / 4), and
    dimensionless spin magnitude (`chi`).
    """

    def __init__(self, parent):
        self.parent = parent
        self.metadata = parent.metadata
        self.horizons = parent.sim.horizons

    @property
    def start_time(self):
        return self.horizons.A.time[0] * self.parent.time_scale

    @property
    def end_time(self):
        return self.horizons.C.time[-1] * self.parent.time_scale

    def component_christodoulou_masses(
        self,
        times_to_interpolate=None,
        delta_t=None,
    ):
        """
        Christodoulou mass of each initial black hole (A and B) over time.

        The two horizons are returned on a common time grid, sampled either
        at `delta_t` (resampling horizon A's native time stamps) or at
        `times_to_interpolate`. Masses and time are scaled by the parent
        simulation's `mass_scale` / `time_scale`.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Times (already in the parent's unit system) at which to
            interpolate both masses. If given, `delta_t` is ignored.
        delta_t : float, optional
            Uniform time spacing to resample onto when
            `times_to_interpolate` is not given. If neither is given, the
            horizon's native (irregular) time sampling is used.

        Returns
        -------
        time_grid : ndarray
            Common time array, in the parent's unit system.
        massA, massB : ndarray
            Christodoulou mass of black holes A and B, scaled by
            `mass_scale`.
        """
        massA = self.horizons.A.christodoulou_mass * self.parent.mass_scale
        massB = self.horizons.B.christodoulou_mass * self.parent.mass_scale
        source_time = self.horizons.A.time * self.parent.time_scale

        if times_to_interpolate is not None:
            time_grid, massA = match_time_grid(source_time, massA, times_to_interpolate)
            _, massB = match_time_grid(source_time, massB, times_to_interpolate)
        else:
            time_grid, massA = resample(source_time, massA, delta_t=delta_t)
            _, massB = resample(source_time, massB, delta_t=delta_t)

        return time_grid, massA, massB

    def common_christodoulou_mass(
        self,
        times_to_interpolate=None,
        delta_t=None,
    ):
        """
        Christodoulou mass of the post-merger remnant horizon (C) over time.

        Sampled either at `delta_t` (resampling horizon C's native time
        stamps) or at `times_to_interpolate`. Mass and time are scaled by
        the parent simulation's `mass_scale` / `time_scale`.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Times (already in the parent's unit system) at which to
            interpolate the mass. If given, `delta_t` is ignored.
        delta_t : float, optional
            Uniform time spacing to resample onto when
            `times_to_interpolate` is not given. If neither is given, the
            horizon's native (irregular) time sampling is used.

        Returns
        -------
        time_grid : ndarray
            Time array, in the parent's unit system.
        massC : ndarray
            Christodoulou mass of the remnant horizon, scaled by
            `mass_scale`.
        """
        massC = self.horizons.C.christodoulou_mass * self.parent.mass_scale
        source_time = self.horizons.C.time * self.parent.time_scale

        if times_to_interpolate is not None:
            time_grid, massC = match_time_grid(source_time, massC, times_to_interpolate)
        else:
            time_grid, massC = resample(source_time, massC, delta_t=delta_t)

        return time_grid, massC

    def component_horizon_fluxes(self, times_to_interpolate=None, delta_t=None):
        """
        Rate of change of each initial black hole's Christodoulou mass.

        Computed as the numerical time-derivative (`numpy.gradient`) of
        `component_christodoulou_masses`, i.e. the mass flux absorbed by
        each horizon (e.g. from tidal/horizon heating) over the inspiral.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Forwarded to `component_christodoulou_masses`.
        delta_t : float, optional
            Forwarded to `component_christodoulou_masses`; ignored if
            `times_to_interpolate` is given.

        Returns
        -------
        time : ndarray
            Common time array, in the parent's unit system.
        fluxA, fluxB : ndarray
            d(mass)/dt for black holes A and B.
        """
        if times_to_interpolate is not None:
            time, massA, massB = self.component_christodoulou_masses(
                times_to_interpolate=times_to_interpolate
            )

        else:
            time, massA, massB = self.component_christodoulou_masses(delta_t=delta_t)

        fluxA = np.gradient(massA, time)
        fluxB = np.gradient(massB, time)
        return time, fluxA, fluxB

    def common_horizon_flux(self, times_to_interpolate=None, delta_t=None):
        """
        Rate of change of the remnant horizon's Christodoulou mass.

        Computed as the numerical time-derivative (`numpy.gradient`) of
        `common_christodoulou_mass`, i.e. the mass flux absorbed by the
        merged remnant horizon after merger (e.g. from ringdown radiation).

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Forwarded to `common_christodoulou_mass`.
        delta_t : float, optional
            Forwarded to `common_christodoulou_mass`; ignored if
            `times_to_interpolate` is given.

        Returns
        -------
        time : ndarray
            Time array, in the parent's unit system.
        fluxC : ndarray
            d(mass)/dt for the remnant horizon.
        """
        time, massC = self.common_christodoulou_mass(
            times_to_interpolate=times_to_interpolate
        )
        fluxC = np.gradient(massC, time)
        return time, fluxC

    def component_areal_masses(
        self,
        times_to_interpolate=None,
        scale=True,
        delta_t=None,
    ):
        """
        Areal (irreducible) mass of each initial black hole over time.

        The areal mass is derived from the horizon's surface area rather
        than its Christodoulou mass; see `component_areas` for the area
        itself. The two horizons are returned on a common time grid, sampled
        either at `delta_t` or at `times_to_interpolate`.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Times at which to interpolate both masses. If given, `delta_t`
            is ignored. Interpreted in the parent's unit system if `scale`
            is True, otherwise in native NR units.
        scale : bool, default True
            If True, scale time and mass by the parent simulation's
            `time_scale` / `mass_scale`. If False, return raw NR-unit
            values.
        delta_t : float, optional
            Uniform time spacing to resample onto when
            `times_to_interpolate` is not given.

        Returns
        -------
        time_grid : ndarray
            Common time array.
        massA, massB : ndarray
            Areal mass of black holes A and B.
        """
        massA = self.horizons.A.areal_mass
        massB = self.horizons.B.areal_mass
        source_time = self.horizons.A.time

        if times_to_interpolate is not None:
            time_grid, massA = match_time_grid(source_time, massA, times_to_interpolate)
            _, massB = match_time_grid(source_time, massB, times_to_interpolate)
        else:
            time_grid, massA = resample(source_time, massA, delta_t=delta_t)
            _, massB = resample(source_time, massB, delta_t=delta_t)

        if scale:
            time_grid *= self.parent.time_scale
            massA *= self.parent.mass_scale
            massB *= self.parent.mass_scale

        return time_grid, massA, massB

    def common_areal_mass(self, times_to_interpolate=None, scale=True, delta_t=None):
        """
        Areal (irreducible) mass of the post-merger remnant horizon over time.

        The areal mass is derived from the horizon's surface area rather
        than its Christodoulou mass; see `common_area` for the area itself.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Times at which to interpolate the mass. If given, `delta_t` is
            ignored, and the result is returned in native NR units
            regardless of `scale` (unlike the `delta_t`/native-sampling
            path below).
        scale : bool, default True
            If True, scale time and mass by the parent simulation's
            `time_scale` / `mass_scale`. Only applied when
            `times_to_interpolate` is not given; see note above.
        delta_t : float, optional
            Uniform time spacing to resample onto when
            `times_to_interpolate` is not given.

        Returns
        -------
        time_grid : ndarray
            Time array.
        massC : ndarray
            Areal mass of the remnant horizon.
        """
        massC = self.horizons.C.areal_mass
        source_time = self.horizons.C.time

        time_grid = source_time.copy()

        if times_to_interpolate is not None:
            time_grid, massC = match_time_grid(source_time, massC, times_to_interpolate)
            return time_grid, massC
        else:
            time_grid, massC = resample(source_time, massC, delta_t=delta_t)
        if scale:
            time_grid *= self.parent.time_scale
            massC *= self.parent.mass_scale

        return time_grid, massC

    def component_areas(self, times_to_interpolate=None, delta_t=None):
        """
        Surface area of each initial black hole's apparent horizon.

        Computed from `component_areal_masses` via the areal-mass/area
        relation `area = 16 * pi * mass**2`, then scaled by the parent
        simulation's `area_scale` / `time_scale`.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Forwarded to `component_areal_masses`.
        delta_t : float, optional
            Forwarded to `component_areal_masses`; ignored if
            `times_to_interpolate` is given.

        Returns
        -------
        time : ndarray
            Common time array, in the parent's unit system.
        areaA, areaB : ndarray
            Horizon area of black holes A and B, scaled by `area_scale`.
        """
        time, massA, massB = self.component_areal_masses(
            times_to_interpolate=times_to_interpolate, delta_t=delta_t, scale=False
        )
        areaA = 16 * np.pi * massA**2 * self.parent.area_scale
        areaB = 16 * np.pi * massB**2 * self.parent.area_scale
        time *= self.parent.time_scale

        return time, areaA, areaB

    def common_area(self, times_to_interpolate=None, delta_t=None):
        """
        Surface area of the post-merger remnant apparent horizon.

        Computed from `common_areal_mass` via the areal-mass/area relation
        `area = 16 * pi * mass**2`, then scaled by the parent simulation's
        `area_scale` / `time_scale`.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Forwarded to `common_areal_mass`.
        delta_t : float, optional
            Forwarded to `common_areal_mass`; ignored if
            `times_to_interpolate` is given.

        Returns
        -------
        time : ndarray
            Time array, in the parent's unit system.
        areaC : ndarray
            Horizon area of the remnant, scaled by `area_scale`.
        """
        time, massC = self.common_areal_mass(
            times_to_interpolate=times_to_interpolate, delta_t=delta_t, scale=False
        )
        areaC = 16 * np.pi * massC**2 * self.parent.area_scale
        time *= self.parent.time_scale

        return time, areaC

    def component_entropies(
        self,
        times_to_interpolate=None,
    ):
        """
        Bekenstein-Hawking entropy of each initial black hole's horizon.

        Computed from `component_areas` as `area / 4`, then scaled by the
        parent simulation's `entropy_scale` (e.g. to convert to units of
        `k_B` in `"SI"` units).

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Forwarded to `component_areas`.

        Returns
        -------
        time : ndarray
            Common time array, in the parent's unit system.
        entropyA, entropyB : ndarray
            Entropy of black holes A and B.
        """
        time, areaA, areaB = self.component_areas(
            times_to_interpolate=times_to_interpolate
        )
        entropyA = areaA / 4.0 * self.parent.entropy_scale
        entropyB = areaB / 4.0 * self.parent.entropy_scale

        return time, entropyA, entropyB

    def common_entropy(
        self,
        times_to_interpolate=None,
    ):
        """
        Bekenstein-Hawking entropy of the post-merger remnant horizon.

        Computed from `common_area` as `area / 4`, then scaled by the
        parent simulation's `entropy_scale` (e.g. to convert to units of
        `k_B` in `"SI"` units).

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Forwarded to `common_area`.

        Returns
        -------
        time : ndarray
            Time array, in the parent's unit system.
        entropyC : ndarray
            Entropy of the remnant horizon.
        """
        time, areaC = self.common_area(times_to_interpolate=times_to_interpolate)
        entropyC = areaC / 4.0 * self.parent.entropy_scale
        return time, entropyC

    def component_chis(
        self,
        times_to_interpolate=None,
        delta_t=None,
    ):
        """
        Dimensionless spin magnitude of each initial black hole over time.

        Taken from the horizon's inertial-frame spin vector magnitude
        (`chi_inertial_mag`), which is already dimensionless and so is not
        rescaled, only the time axis is scaled by the parent simulation's
        `time_scale`.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Times at which to interpolate both spins. If given, `delta_t`
            is ignored.
        delta_t : float, optional
            Uniform time spacing to resample onto when
            `times_to_interpolate` is not given.

        Returns
        -------
        time_grid : ndarray
            Common time array, in the parent's unit system.
        chiA, chiB : ndarray
            Dimensionless spin magnitude of black holes A and B.
        """
        chiA = self.horizons.A.chi_inertial_mag
        chiB = self.horizons.B.chi_inertial_mag
        source_time = self.horizons.A.time

        if times_to_interpolate is not None:
            time_grid, chiA = match_time_grid(source_time, chiA, times_to_interpolate)
            _, chiB = match_time_grid(source_time, chiB, times_to_interpolate)
        else:
            time_grid, chiA = resample(source_time, chiA, delta_t=delta_t)
            _, chiB = resample(source_time, chiB, delta_t=delta_t)

        return time_grid * self.parent.time_scale, chiA, chiB

    def common_chi(self, times_to_interpolate=None, delta_t=None):
        """
        Dimensionless spin magnitude of the post-merger remnant over time.

        Taken from the horizon's inertial-frame spin vector magnitude
        (`chi_inertial_mag`), which is already dimensionless and so is not
        rescaled, only the time axis is scaled by the parent simulation's
        `time_scale`.

        Parameters
        ----------
        times_to_interpolate : ndarray, optional
            Times at which to interpolate the spin. If given, `delta_t` is
            ignored.
        delta_t : float, optional
            Uniform time spacing to resample onto when
            `times_to_interpolate` is not given.

        Returns
        -------
        time_grid : ndarray
            Time array, in the parent's unit system.
        chiC : ndarray
            Dimensionless spin magnitude of the remnant.
        """
        chiC = self.horizons.C.chi_inertial_mag
        source_time = self.horizons.C.time

        time_grid = source_time.copy()

        if times_to_interpolate is not None:
            time_grid, chiC = match_time_grid(source_time, chiC, times_to_interpolate)
        else:
            time_grid, chiC = resample(time_grid, chiC, delta_t=delta_t)

        return time_grid * self.parent.time_scale, chiC


class SXSSimulation:
    """
    Convenience wrapper around a single SXS catalog simulation.

    Loads a numerical-relativity simulation from the `sxs` package by its
    catalog label and exposes its waveform modes, metadata, and apparent
    horizon data (via a nested :class:`Horizon` instance at `self.horizon`)
    with a consistent, configurable unit system and a flexible way to pick
    the time window of the waveform to work with.

    Parameters
    ----------
    simulation_label : str
        SXS catalog label of the simulation to load, e.g. `"SXS:BBH:0305"`.
        Passed directly to `sxs.load`. Required; raises `ValueError` if
        left as `None`.

    units : {"NR", "Geometric", "SI"}, default "NR"
        Unit system used to scale all derived quantities (time, mass, area,
        strain, angular momentum, entropy):

        - `"NR"`: no rescaling; quantities are in the code units natively
          used by the SXS catalog (total Christodoulou mass = 1).
        - `"Geometric"`: quantities are rescaled to physical geometric units
          (G = c = 1) set by `total_mass` (and `distance` for strain).
        - `"SI"`: quantities are rescaled to SI units (seconds, kilograms,
          metres, ...) using `total_mass` and `distance`.

        Raises `ValueError` for any other value.

    total_mass : float, default 1
        Total binary mass in solar masses, `Ms`. Used to set the mass-to-time
        conversion `MT = Ms * MTsun` and all scale factors when `units` is
        `"Geometric"` or `"SI"`. Ignored when `units="NR"`.

    distance : float, default 200
        Luminosity distance to the source in Mpc, `DL`. Used together with
        `total_mass` to set the strain scale factor. Ignored when
        `units="NR"`.

    initial_cutoff, final_cutoff : None, str, or float, default "reference" / "common_horizon"
        Specification of the start/end of the time window used by
        `waveform_mode` (and everything built on it). Resolved to a physical
        time by `_resolve_time`:

        - `None`: no cutoff (use the full available waveform).
        - `float`/`int`: an explicit time, in the units set by `units`.
        - `str`: a named event looked up from simulation metadata/waveform:
          `"reference"` (reference time), `"common_horizon"` (time the
          common apparent horizon forms), `"peak_amplitude"` (time of peak
          waveform amplitude for the mode being requested), or
          `"last_common_time"` (the earlier of the waveform's and the
          common horizon's final times).

    Attributes
    ----------
    sim : sxs.Simulation
        The underlying simulation object returned by `sxs.load`.

    waveform_modes : sxs.waveforms.WaveformModes
        Alias of `sim.h`, the full set of spin-weighted spherical-harmonic
        waveform modes.

    metadata : sxs.metadata.Metadata
        Alias of `sim.metadata`.

    horizon : Horizon
        Apparent-horizon data for this simulation; see :class:`Horizon`.

    time_scale, length_scale, mass_scale, area_scale, angular_momentum_scale, strain_scale, entropy_scale : float
        Multiplicative factors converting native NR-unit quantities into the
        requested `units` system. All equal to `1.0` when `units="NR"`.

    remnant_mass : float
        Property; final remnant mass, scaled by `mass_scale`.

    remnant_chi : float
        Property; final remnant dimensionless spin magnitude (unscaled, as
        it is already dimensionless).
    """

    def __init__(
        self,
        simulation_label=None,
        units="NR",
        total_mass=1,
        distance=200,
        initial_cutoff="reference",
        final_cutoff="common_horizon",
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
                const.GNewton * (self.Ms * const.Msun) ** 2 / const.cLight
            )
            self.strain_scale = (self.Ms / self.DL) * const.strain_fac
            self.entropy_scale = (
                const.kBoltzmann * const.cLight**3 / (const.GNewton * const.hbar)
            )

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
        last_wf_time = self.waveform_modes.t[-1] * self.time_scale
        last_horizon_time = self.horizon.end_time * self.time_scale
        last_common_time = min(last_wf_time, last_horizon_time)

        if cutoff is None:
            return None

        if isinstance(cutoff, (int, float)):
            return float(cutoff)

        if isinstance(cutoff, str):
            lookup = {
                "reference": self.metadata.reference_time * self.time_scale,
                "common_horizon": self.metadata.common_horizon_time * self.time_scale,
                "peak_amplitude": peak_time,
                "last_common_time": last_common_time,
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
        hlm = (
            self.waveform_modes[:, self.waveform_modes.index(*mode)] * self.strain_scale
        )
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

        time_resampled, hlm_resampled = resample(
            time, hlm, delta_t=delta_t, sampling=sampling
        )

        return time_resampled, hlm_resampled

    def gw_flux(
        self,
        delta_t=None,
        ell_min=2,
        ell_max=8,
    ):
        """
        Compute the gravitational wave flux from the SXS simulation."""
        gw_flux = None
        for ell in range(ell_min, ell_max + 1):
            for em in range(-ell, ell + 1):
                time, hlm = self.waveform_mode(mode=(ell, em), delta_t=delta_t)
                if gw_flux is None:
                    gw_flux = np.zeros_like(time)
                    t_nr = time.copy()

                hlm_dot = np.gradient(hlm, time)

                gw_flux += np.abs(hlm_dot) ** 2

        gw_flux *= 1.0 / (16.0 * np.pi)

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

        phase = -np.unwrap(np.angle(hlm))

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
