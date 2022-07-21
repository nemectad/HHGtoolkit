#!/usr/bin/python
import numpy as np

### Constants
c = 2.9979e08           # speed of light in a vacuum
eps_0 = 8.8542e-12      # permitivity of a vacuum
h = 6.62607e-34         # Planck constant
e = 1.602176e-19        # elementary charge
nu_0 = 4.13e16          # atomic unit of frequency
I_0 = 3.50944758e16     # atomic unit of intensity


### Conversion to atomic units: m = \hbar = e = 1 [a.u.]

### Functions

### Full Width at Half Maximum (FWHM) for sin^2 field
def FWHM_cycles(N_cycles):
    return 2 * N_cycles


### Conversion of intensity to the electric field amplitude
def intensity_to_E(I, mode="au"):
    if mode == "au":
        return np.round(np.sqrt(I / I_0), decimals=3)
    elif mode == "SI":
        return np.sqrt(2 * I / (eps_0 * c))

def E_to_intensity(E):
    return np.power(E,2) * I_0

### Secondary field parameters with respect to the first field
def secondary_field(omega_0, rel_omega, A_0, A_rel, N_cycles, phi, cosine_pulse):
    if (not cosine_pulse):
        if rel_omega == 2:
            CEP_2 = -0.5
        elif rel_omega == 3:
            CEP_2 = -1
        else:
            CEP_2 = 0
    else:
        CEP_2 = 0.5
    return [rel_omega * omega_0, A_0 * A_rel, rel_omega * N_cycles, CEP_2 + phi]


### Conversion of wavelength in [nm] to frequency [a.u.]
def lambda_to_omega(lamb, mode="au", units="n"):
    if mode == "au":
        ### Frequency over lambda in [a.u.]
        lambda_0 = 45.5633526
        return np.round(lambda_0 / lamb, decimals=4)
    elif mode == "SI":
        return 2 * np.pi * c / (lamb * SI_prefix("n"))


### Conversion of pulse length to N cycles of field (with respect
### to FWHM)
def pulse_length_to_cycles(omega_0, t, FWHM=True):
    T = 2 * np.pi / omega_0
    if FWHM == True:
        return np.round(FWHM_cycles(t / T), decimals=0)
    else:
        return np.round(t / T, decimals = 0)


### Conversion of ionization potential to [a.u.]
def Ip_to_au(Ip):
    return np.round(Ip / 27.2113962, decimals=3)


def SI_prefix(prefix):
    if (prefix == "mu"):
        return 1E-6
    elif (prefix == "n"):
        return 1E-9
    elif (prefix == "p"):
        return 1E-12
    elif (prefix == "f"):
        return 1E-15
    elif (prefix == "a"):
        return 1E-18
    else:
        print(
            "Undefined SI prefix '{}', "
            "returning 1.".format(prefix)
        )
        return 1

def compute_Up(E0, omega0):
    return E0 ** 2 / (4 * (omega0 * 2 * np.pi) ** 2)

def eV_per_harmonic(freq_0):
    return nu_0*freq_0*h/e