# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `gw_energy_flux` and `gw_angular_momentum_flux` applied the unit scale on
  top of already-scaled strain and time, giving wrong values in `"SI"` and
  `"Geometric"` units. The 1.1.1 note that the result is numerically
  identical to the previous behaviour does not hold for those units.
- Horizon areal masses, areas, entropies and spins were interpolated on the
  unscaled NR time axis against scaled target times, and the returned time
  grid was scaled a second time. Time is now scaled up front everywhere.
- `Horizon` docstrings now describe the default sampling correctly: with
  neither `delta_t` nor `times_to_interpolate`, quantities are returned on a
  uniform grid at the coarsest native time step (behaviour unchanged).
- `SXSSimulation._resolve_time` scaled the horizon end time twice, breaking
  `"last_common_time"` in non-NR units.
- `final_cutoff="peak_amplitude"` gave each mode its own window, breaking
  the flux mode sums. The flux methods take a new `peak_mode` (default
  `(2, 2)`) shared by all modes; `waveform_mode` gains the same argument.
- `save_sxs_data_to_h5` and the examples called the removed `gw_flux`.
- `delta_t` is now honoured by the common-horizon flux methods and accepted
  by `component_entropies` and `common_entropy`.
- `remnant_chi` returns the spin magnitude, not the spin vector.
- Horizon B is interpolated on its own time stamps rather than horizon A's.

## [1.1.1] - 2026-10-15

### Added

- `Horizon.component_angular_momenta`, `common_angular_momentum`,
  `component_angular_momentum_fluxes`, and `common_angular_momentum_flux`.
- `SXSSimulation.gw_angular_momentum_flux`, computing the z-component of
  the gravitational-wave angular momentum flux via the standard
  multipolar expression (Ruiz, Alcubierre, Bruegmann & Takahashi 2008).
- `SXSSimulation.energy_scale` unit-conversion attribute.
- Detailed docstrings for `Horizon` and `SXSSimulation` and their methods.

### Changed

- `SXSSimulation.gw_energy_flux` (renamed from `gw_flux`) now scales by
  `energy_scale / time_scale` instead of a special-cased SI conversion;
  numerically identical to the previous behavior.

### Fixed

- `Horizon.common_areal_mass` now respects the `scale` argument when
  `times_to_interpolate` is given (previously returned unscaled values
  regardless of `scale` in that code path).

## [1.0.1] - 2026-09-15

### Added

- Changelog file.
- GitLab CI/CD pipeline to build and publish releases to PyPI.

### Changed

- Version is now derived dynamically from git tags via `setuptools_scm` instead of being hardcoded in `pyproject.toml`.

## [1.0.0] - 2026-08-23

### Changed

- Package version is now read from installed metadata in `__init__.py`.

## [0.1.1] - 2026-08-23

### Changed

- Style improvements.

[1.1.1]: https://gitlab.com/samanwaya-mukherjee/gwfit/-/compare/v1.0.1...v1.1.1
[1.0.1]: https://gitlab.com/samanwaya-mukherjee/gwfit/-/compare/v1.0.0...v1.0.1
[1.0.0]: https://gitlab.com/samanwaya-mukherjee/gwfit/-/compare/v0.1.1...v1.0.0
[0.1.1]: https://gitlab.com/samanwaya-mukherjee/gwfit/-/releases/v0.1.1
