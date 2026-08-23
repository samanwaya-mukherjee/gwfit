# GWFIT

<p align="center">
  <img src="assets/gwfit_logo.png" alt="gwfit logo" width="220">
</p>

<p align="center">
  <b>Gravitational WaveForm Integration Tools</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/gwfit/">
    <img src="https://img.shields.io/pypi/v/gwfit.svg" alt="PyPI version">
  </a>
  <a href="https://gitlab.com/samanwaya-mukherjee/gwfit">
    <img src="https://img.shields.io/badge/source-GitLab-black.svg" alt="GitLab">
  </a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>


GWFIT is a Python package for gravitational-wave modelling and analysis, with utilities for waveform generation, inspiral modelling, numerical-relativity data analysis, horizon fluxes, and neutron-star calculations.

## Features

* Gravitational-waveform and inspiral modelling
* Numerical-relativity data handling
* Horizon flux and tidal-heating calculations
* Neutron-star utilities
* Physical constants and useful relations

## Installation

Create and activate a conda environment
```bash
conda create -n gwfit-env python=3.11 -y
conda activate gwfit
```

There are several ways to install GWFIT depending on whether you want the latest stable release or the development version.

### 1. Install the latest stable release from PyPI

For most users, the recommended method is to install GWFIT directly from PyPI:

```bash
python -m pip install gwfit
```

### 2. Install the latest stable release from git repository

```bash
python -m pip install git+https://gitlab.com/samanwaya-mukherjee/gwfit.git
```

### 2. Install by cloning the source

Clone the repository:

```bash
git clone https://gitlab.com/samanwaya-mukherjee/gwfit.git
cd gwfit
python -m pip install .
```

[
    For development, the editable installation is recommended so that changes to the source code are immediately available. 

Install GWFIT in editable mode:

```bash
python -m pip install -e .
```
]


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
        │   ├── nr_data_utils_legacy.py
        │   ├── nr_heat_legacy.py
        │   └── inspiral_th_model_legacy.py
        ├── neutron_star/
        │   ├── __init__.py
        │   └── neutron_star_functions.py
        ├── nr/
        │   ├── __init__.py
        │   ├── nr_data_utils.py
        │   └── nr_heat.py
        ├── physics/
        │   ├── __init__.py
        │   ├── constants.py
        │   └── useful_relations.py
        └── waveforms/
            ├── __init__.py
            ├── inspiral_model.py
            └── waveform_functions.py
```

## Dependencies

The core package currently depends on:

* NumPy
* SciPy
* SXS
* Matplotlib 

Additional dependencies for legacy functionality are:

* h5py
* PyCBC

## Development

GWFIT is under active development. Contributions, suggestions, and bug reports are welcome.

## License

GWFIT is distributed under the [MIT License](LICENSE).

## Author

**Samanwaya Mukherjee**
