"""GWFIT: Gravitational Waveform Integration Tools."""

from importlib.metadata import version

__version__ = version("gwfit")

from .nr import nr_data_utils
from .physics import constants, useful_relations
from .waveforms import inspiral_model, waveform_functions