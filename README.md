# GWFIT

**Gravitational Waveform Integration Tools**

GWFIT is a Python package for gravitational-wave modelling and analysis, with utilities for waveform generation, inspiral modelling, numerical-relativity data analysis, horizon fluxes, and neutron-star calculations.

## Features

* Gravitational-waveform and inspiral modelling
* Numerical-relativity data handling
* Horizon flux and tidal-heating calculations
* Neutron-star utilities
* Physical constants and useful relations

## Installation

Clone the repository:

```bash
git clone https://gitlab.com/samanwaya-mukherjee/gwfit.git
cd gwfit
```

Install GWFIT in editable mode:

```bash
python -m pip install -e .
```

For development, the editable installation is recommended so that changes to the source code are immediately available.

### Legacy functionality

Some older implementations are included under `gwfit.legacy`. Their additional dependencies can be installed with:

```bash
python -m pip install -e ".[legacy]"
```

## Basic usage

After installation, GWFIT can be imported as:

```python
import gwfit
```

The main components are organized into subpackages:

```python
from gwfit import physics
from gwfit import waveforms
from gwfit import nr
from gwfit import neutron_star
```

For example, specific modules can be imported with:

```python
from gwfit.waveforms import inspiral_model
from gwfit.nr import nr_data_utils
from gwfit.physics import constants
```

## Project structure

```text
gwfit/
├── LICENSE
├── pyproject.toml
├── README.md
└── src/
    └── gwfit/
        ├── __init__.py
        ├── legacy/
        ├── neutron_star/
        ├── nr/
        ├── physics/
        └── waveforms/
```

## Dependencies

The core package currently depends on:

* NumPy
* SciPy
* SXS

Additional dependencies for legacy functionality are:

* h5py
* PyCBC

## Development

GWFIT is under active development. Contributions, suggestions, and bug reports are welcome.

## License

GWFIT is distributed under the [MIT License](LICENSE).

## Author

**Samanwaya Mukherjee**
