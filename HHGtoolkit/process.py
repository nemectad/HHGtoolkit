import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal.windows import *
from scipy.signal import hilbert
import HHGtoolkit.utilities as util
import mynumerics as mn
import HHGtoolkit.Hfn2 as hfn

### Issue: for arrays of different lengths but same data, i.e. different T_max
### we need to multiply the result by T_max 
def spectrum(arr, t_step):
    N = np.shape(arr)[-1]

    #tf = np.linspace(0.0, 1.0 / (2.0 * t_step), N // 2)
    tf = fftfreq(N, t_step)[0 : N//2]

    if (len(arr) == N):
        return 2 / N * np.abs(fft(arr)[0 : N // 2]), tf
    else:
        return 2 / N * np.abs(fft(arr)[:, 0 : N // 2]), tf


def window(arr, *args, win = blackman, **kwargs):
    N = len(arr)
    ### Window weights
    w = np.array(win(N, *args, **kwargs))

    ### element-by-element multiplication
    return np.multiply(arr, w)


def gabor_transf(arr, t, t_min, t_max, N_steps = 400, a = 8):
    '''
    Gabor transform
    ===============


    gabor_transf(arr, t, t_min, t_max, N_steps = 400, a = 8)


    Computes Gabor transform for arbitrary number array 'arr' of 
    lenght N. 

        Parameters:
            arr (np.array or list of floats): input data for Gabor transform, length N
            t (np.array or list of floats): time domain for data, length N
            t_min (float): minimum time for Gabor transform domain
            t_max (float): maximum time for Gabor transform domain

        Optional parameters:
            N_steps (int): number of discretization points
            a (float): Gabor window parameter
    
        Returns:
            gabor_transf (np.array(N_steps, N)): Gabor transform of arr 

    Note: time domain 't' must correspond to the array 'arr'.

    Example:
        import numpy as np
        import random

        t_min = 0
        t_max = 1
        N_steps = 100

        x = np.linspace(0,1,100)
        y = [np.cos(2*np.pi*t) + np.sin(np.pi*t) + 0.1*random.randrange(-1,1) for t in x]

        z = gabor_transform(y, x, t_min, t_max, N_steps = N_steps)

    '''
    if (len(arr) != len(t)):
        raise ValueError('Arrays must have same dimension.')

    if (t_max < t_min):
        raise ValueError('Maximum time must be larger than minimum time.')

    ### Time domain for Gabor transform
    t_0 = np.linspace(t_min, t_max, N_steps)

    ### Number of sample points for fft
    N = len(arr)

    ### Init np.array
    gabor_transf = np.zeros((np.shape(t_0)[0], np.shape(t)[0]))

    ### Compute Gabor transform
    for i in range(0, N_steps):
        gabor_transf[i, :] = 2.0 / N * np.abs(fft(np.exp(-np.power((t-t_0[i])/a, 2))*arr[:]))
    return gabor_transf


def linear_to_log(arr):
    min_ = np.log10(min(arr))
    max_ = np.log10(max(arr))
    return max(arr)*(np.log10(arr)-min_)/(max_-min_)

def sigmoid(len): 
    x=np.linspace(-6,+6,len) 
    y = 1/(1 + np.exp(-x)) 
    return -y + 1

def harmonic_filter(arr, t_max, field_freq, harmonic_freq, filter_type = "bandpass", *args, **kwargs):
    N_steps_per_freq = int(field_freq*t_max) + 1

    if filter_type == "bandpass":
        range_ = [int(N_steps_per_freq*harmonic_freq) - N_steps_per_freq//2,
                    int(N_steps_per_freq*harmonic_freq) + N_steps_per_freq//2]
    elif filter_type == "lowpass":
        range_ = [0, int(N_steps_per_freq*(harmonic_freq+1/2))]
    else:
        raise ValueError("Unknown filter '" + filter_type + "', "
            "allowed filters are: 'bandpass' and 'lowpass'.")
    ### Apply spectral filter
    return spectral_filter(arr, range_, filter_type, *args, **kwargs)

def spectral_filter(arr, range_, filter_type, win = blackman, len_ = 20, *args, **kwargs):
    N = np.shape(arr)[-1]

    ### Compute FFT
    spectrum = fft(arr)
    
    ### Create corresponding kernel
    if filter_type == 'bandpass':
        kernel = np.zeros(N)
        kernel[range_[0]:range_[1]] = win(range_[1]-range_[0], *args, **kwargs)
    elif filter_type == 'lowpass':
        kernel = np.ones(N)
        kernel[range_[1]:(range_[1]+len_)] = sigmoid(len_)
        kernel[range_[1]+len_:] = 0
    else:
        raise ValueError("Unknown filter '" + filter_type + "', "
            "allowed filters are: 'bandpass' and 'lowpass'.")
 
    ### Apply filter and compute IFFT
    return ifft(np.multiply(spectrum, kernel)).real


def stokes_params(E_1, E_2, t, norm = True):
    ### Hilbert transform of a real signal to a complex analytical signal
    E_1 = hilbert(E_1)
    E_2 = hilbert(E_2)

    ### Computation of time mean value and coherency matrix
    J_11 = mean(np.multiply(E_1, np.conjugate(E_1)), t)
    J_12 = mean(np.multiply(E_1, np.conjugate(E_2)), t)
    J_21 = mean(np.multiply(E_2, np.conjugate(E_1)), t)
    J_22 = mean(np.multiply(E_2, np.conjugate(E_2)), t)


    ### Stokes parameters from coherency matrix
    S_0 = np.real(J_11 + J_22)
    S_1 = np.real(J_11 - J_22)
    S_2 = np.real(J_12 + J_21)
    S_3 = np.real(1j*(J_21 - J_12))

    if norm:
        return S_0/S_0, S_1/S_0, S_2/S_0, S_3/S_0
    else:
        return S_0, S_1, S_2, S_3
    

def stokes_params_from_file(h5_file, norm = True, filter_harmonics = None):
    data = util.load_data(h5_file)

    t = data.t
    dipole_z = data.d_z
    dipole_x = data.d_x
    freq_0 = data.omega_1 / (2 * np.pi)

    if filter_harmonics != None:
        dipole_z, dipole_x = harmonic_filter([dipole_z,dipole_x], (t[-1]-t[0]), freq_0,  
                        filter_harmonics, 'bandpass')

    return stokes_params(dipole_z, dipole_x, t, norm = norm)

### Trapezoid rule integration
def trapezoid(arr, dt):
    N = len(arr)

    #I = 0
    #for i in range(0, N-1):
    #    I = I + dt*(arr[i+1] + arr[i])/2

    integral = (arr[0] + arr[-1])/2
    for i in range(1, N-1):
        integral += arr[i]
    
    #assert(I == integral)

    return dt*integral
    #return I


def mean(arr, t):
    T = t[-1]-t[0]
    dt = t[1]-t[0]
    return trapezoid(arr, dt)/T


def integrate_spectrum(data, harmonics = [0, 50], component = 'z'):
    data = util.load_data(data)

    t = data.t
    dip_z = data.d_z
    dip_x = data.d_x
    freq_0 = data.omega_1 / (2 * np.pi)
    T_step = t[1]-t[0]

    ### Compute spectrum and time domain for spectrum
    spectrum_, tf = spectrum([dip_z, dip_x], T_step)

    spectrum_z, spectrum_x = spectrum_

    ### Set values for omega
    omegas = np.linspace(0, tf[-1], int(round(tf[-1] / freq_0))+1)
    omegas = omegas[0:int(harmonics[1])+1]

    N_steps_per_freq = int(t[-1]*freq_0)+1

    range_ = slice(int(N_steps_per_freq*harmonics[0]), int(N_steps_per_freq*harmonics[1]))

    if component == 'z':
        return trapezoid(spectrum_z[range_], tf[1]-tf[0])
    else:
        return trapezoid(spectrum_x[range_], tf[1]-tf[0])

def distribution(amplitude, w_0 = 1, N_pts = 100):
    '''
    Parameters:
        amplitude: float
            field amplitude
        w_0: float
            waist radius
        N_pts: int
            number of points for discretization
    Returns:
        Float array of r-values and intensities given by gaussian distribution
    
    '''
    ### Field threshold: E \sim 1/e^2 E_0
    rs = np.linspace(0, np.sqrt(2)*w_0, N_pts)
    Es = np.array([amplitude * np.exp(- (r/w_0)**2) for r in rs])
    return np.array([rs, Es])

def set_omegas(tf, freq_0):
    '''
    Set of omegas for plotting and computation

    Parameters:
        tf: float
            Frequency axis from FFT.
        freq_0: float
            Frequency (i.e. omega/(2\pi)) of the fundamental field. 
    Returns:
        Float array of frequencies corresponding to harmonic spectrum.
    '''
    return np.linspace(0, tf[-1], int(round(tf[-1] / freq_0))+1)


def get_Hankel_transform(
        Field,
        omega_0, 
        rgrid, 
        distance, 
        rgrid_FF, 
        near_field_factor, 
        t,
        omega_max = -1
    ):
    
    freq_0 = omega_0/(2*np.pi)
    N_steps_per_freq = int(freq_0*t[-1]) + 1
    T_step = t[1]-t[0]
    
    ### Compute spectrum
    N = np.shape(Field)[-1]
    FField = fft(Field)[0:N//2]
    ogrid = fftfreq(N, T_step)[0:N//2]
    FField = np.transpose(FField)

    if omega_max == -1:
        rng = slice(0, -1)
    else:
        rng = slice(0, N_steps_per_freq*omega_max)

    omegaSI = mn.ConvertPhoton(omega_0, 'omegaau', 'omegaSI')
    omeg_0 = (ogrid[N_steps_per_freq-1] - ogrid[0])
    z = hfn.HankelTransform(omegaSI*ogrid[rng]/\
                                   omeg_0, rgrid, FField, distance, rgrid_FF, 
                                   near_field_factor = near_field_factor)
    return np.transpose(z)

