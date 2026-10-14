# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
