# Copyright (c) 2026 Samanwaya Mukherjee

from scipy import constants

GNewton = constants.G
cLight  = constants.c
kBoltzmann = constants.k
hPlanck = constants.h
hbar = constants.hbar

REarth = 6378136.6
AU = 1.4959787066e11
year = 3.1536e7
Msun = 1.9884099021470415e30
Mpc  = 3.085677581491367e22

MTsun = Msun * GNewton / cLight**3
time_fac = MTsun
strain_fac = GNewton * Msun / (cLight**2 * Mpc)