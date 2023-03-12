import h5py
import numpy as np
import os
import re
import HHGtoolkit.conversions as conv
import matplotlib.pyplot as plt


class Data:
    def __init__(self, path) -> None:
        self.path = path

        load_data = h5py.File(path, "r")
        results = load_data["outputs"]
        input_data = load_data["inputs"]

        ### Output values
        self.t = np.array(results["tgrid"][:])
        self.E_z = np.array(results["Efield"][:, 0])
        self.E_x = np.array(results["Efield"][:, 1])
        try:
            self.d_z = np.array(results["dipole"][:, 0])
            self.d_x = np.array(results["dipole"][:, 1])
            self.A_z = np.array(results["Afield"][:, 0])
            self.A_x = np.array(results["Afield"][:, 1])
        except KeyError:
            pass
        
        try:
            self.I_z = np.array(results["integrand"][0, :, 0]) + 1j*np.array(results["integrand"][1, :, 0])
            self.I_x = np.array(results["integrand"][0, :, 1]) + 1j*np.array(results["integrand"][1, :, 1])
        except KeyError:
            pass

        try:
            self.j = results["Current"][()]
            self.p = results["Population"][()]
        except KeyError:
            #print("Warning: no time current and population information in the data.")
            pass

        ### Input parameters
        self.omega_1 = input_data["omega_1"][()]
        self.omega_2 = input_data["omega_2"][()]
        self.I_p = input_data["ground_state_energy"][()]
        self.E0_1 = input_data["E0_1"][()]
        self.E0_2 = input_data["E0_2"][()]
        self.N_cycl_1 = input_data["number_of_cycles_1"][()]
        self.N_cycl_2 = input_data["number_of_cycles_2"][()]
        self.CEP_1 = input_data['CEP_1'][()]
        self.CEP_2 = input_data['CEP_2'][()]
        try:
            self.dz = input_data["dz"][()]
        except KeyError:
            pass

        try:
            self.delay = input_data['delay_between_pulses'][()]
        except KeyError:
            #print("Warning: no time delay information in the data.")
            pass

        try:
            self.theta_1 = input_data['theta_1'][()]
            self.theta_2 = input_data['theta_2'][()]
        except KeyError:
            #print("Warning: no time theta information in the data.")
            pass

        try:
            self.eps_1 = input_data['eps_1'][()]
            self.eps_2 = input_data['eps_2'][()]
        except KeyError:
            #print("Warning: no time ellipticity information in the data.")
            pass
        
        try:
            self.eps_1 = input_data['tau_1'][()]
            self.eps_2 = input_data['tau_2'][()]
        except KeyError:
            #print("Warning: no time pulse width information in the data.")
            pass

        try:
            self.w0_1 = input_data["waist_1"][()]
            self.w0_2 = input_data["waist_2"][()]
            self.z0_1 = input_data["focus_1"][()]
            self.z0_2 = input_data["focus_2"][()]
            self.z_trg = input_data["z_target"][()]
            self.N_distr = input_data["points_per_field_distribution"][()]
        except KeyError:
            pass

        try: 
            self.x = np.array(results["xgrid"][:])
            psi = np.array(results["psi"][:])
            self.psi = np.array([psi[2*i] + 1j*psi[2*i+1] for i in range(len(self.x))], dtype=complex)
            psi0 = np.array(results["ground_state"][:])
            self.psi0 = np.array([psi0[2*i] + 1j*psi0[2*i+1] for i in range(len(self.x))], dtype=complex)
            self.pop = np.array(results["population"][:])
            self.j = np.array(results["current"][:])
            self.pot = np.array(results["potential"][:])
            self.E_ground = results["E_ground_state"][()]
        except KeyError:
            pass

        try:
            grad_pot_re = np.array(results["grad_pot"][:,0])
            grad_pot_im = np.array(results["grad_pot"][:,1])
            self.grad_pot = np.array(grad_pot_re + 1j*grad_pot_im, dtype=complex)
        except KeyError:
            pass

        ### Numerics parameters        
        self.N_int = input_data["points_per_cycle_for_integration"][()]
        self.N_pts = input_data["points_per_cycle_for_evaluation"][()]

        load_data.close()

class Dataset(list):
    def __init__(self, args = None) -> list:
        if args is None:
            super(Dataset, self).__init__()
        elif type(args) is list:
            files = [Data(file) for file in args]
            super(Dataset, self).__init__(files)
        elif type(args) is str:
            super(Dataset, self).__init__([Data(args)])
        else:
            super(Dataset, self).__init__()
  
    def append(self, item) -> None:
        if type(item) is str:
            super(Dataset, self).append(Data(item))
        elif type(item) is Data:
            super(Dataset, self).append(item)
        else:
            raise TypeError("Item is not of type 'Data' or 'str'.")

    def find_data(self, phi = None, intensity = None, amplitude = None) -> Data:
        if phi == None and intensity == None and amplitude == None:
            raise ValueError("One of the values 'phi', 'intensity' or 'amplitude'"
                            " must be selected!")

        if phi != None:
            phases = [data_.CEP_2 for data_ in self]

            omega_1 = self[0].omega_1
            omega_2 = self[0].omega_2

            rel_omega = int(np.round(omega_2/omega_1))
            if self[0].CEP_1 == 0.5:
                cosine_pulse = True
            else:
                cosine_pulse = False
            CEP_2 = conv.secondary_field(omega_1, rel_omega, 0, 0, 0, 0, cosine_pulse)[3]

            delta_phis = np.abs(np.array(phases) - CEP_2 - phi)

            N_min = delta_phis.argmin()

            print("Plotting for {:.2f} π rad".format(phases[N_min] - CEP_2))

            return self[N_min]
        elif amplitude != None:
            Es = [data_.E0_2 for data_ in self]
            E_0 = self.E0_1
            delta_amplitude = np.abs(np.array(Es)-amplitude*E_0)

            N_min = delta_amplitude.argmin()

            print("Plotting for {:.2f}E_0".format(Es[N_min]/E_0))

            return self[N_min]

def load_dataset(dir) -> Dataset:
    if type(dir) is Dataset:
        return dir
    elif type(dir) is str:
        ds = Dataset()
        arr = os.listdir(dir)
        file_paths = [dir + x for x in arr if x.endswith('.h5')]
        sort_nicely(file_paths)

        for path in file_paths:
            ds.append(Data(path))

        return ds
    else:
        raise TypeError("Input is not of type 'Dataset' or 'str'.")

def load_data(data) -> Data:
    if type(data) is str:
        return Data(data)
    elif type(data) is Data:
        return data
    else:
        raise TypeError("Data is not of type 'Data' or 'str'.")

### Methods for obtaining ordered list of h5 files
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect. """
    l.sort(key=alphanum_key)

### Methods for Gaussian beam plots
class Beam:
    """
    Beam
    ====
    Class containing info about the irradiating beam.

    Methods:
    --------
    * __init__(self, lamb, amplitude, z_0 = 0., w_0 = 1e-4, N_pts = 100)

    """
    def __init__(self, lamb, amplitude, z_0 = 0., w_0 = 1e-4, N_pts = 100):
        """
        __init__
        ========
        Initialization of a single Beam.

        Parameters:
        -----------
        lamb: float
            Wavelength in [SI].
        amplitude: float
            Electric field amplitude in [a.u.].
        z_0: float, optional, default: 0.
            Focus of the beam on the z-axis in [SI].
        w_0: float, optional, default: 1e-4
            Beam waist at focus in [SI].
        N_pts: int, optional, default: 100
            Number of points for evaluation.
        """
        self.lamb = lamb
        self.z_0 = z_0
        self.w_0 = w_0
        self.amplitude = amplitude
        self.N_pts = N_pts
        self.w = None
        self.profile = None

class Beams:
    def __init__(self):
        self.beams = []
    
    def create_beam(self, lamb, amplitude, **kwargs):
        self.beams.append(Beam(lamb, amplitude, **kwargs))

    def add_beam(self, beam):
        if not isinstance(beam, Beam):
            raise TypeError("Beam must be of type Beam.")
        self.beams.append(beam)
        
    def plot_beams(self, z_target = 0., profile = True, from_file = False):
        ### Get beam profiles for plotting
        rho, z = self.get_beam_profiles(z_target)

        fig, ax = plt.subplots(figsize=(8, 4))

        ### Colors corresponding to fundamental 'red' and 2nd harmonic 'blue'
        colors = ['red', 'blue']

        for i, b in enumerate(self.beams):
            if len(self.beams) <= len(colors):
                ax.plot(z, b.w, color = colors[i], label=r"$\lambda$ = {} nm".format(int(np.round(b.lamb*1e9, decimals=-2))))
                ax.plot(z, -b.w, color = colors[i])
            else:
                ax.plot(z, b.w, label=r"$\lambda$ = {} nm".format(int(np.round(b.lamb))))
        
        ### Plot Gaussian profile alongside waist
        if profile:
            ax2 = ax.twiny()
            
            for i, b in enumerate(self.beams):
                if len(self.beams) <= len(colors):
                    ### Plot real parts of electric fields
                    ax2.plot(z_target + np.real(b.profile), rho, color = colors[i], linestyle=":")
                    ax2.plot(z_target + np.real(b.profile), -rho, color = colors[i], linestyle=":")
                else:
                    ax2.plot(z_target + np.real(b.profile), rho, linestyle=":")
                    
            max_amp = np.round(max([max(np.real(b.profile)) for b in self.beams]), decimals=3)
            xticks = [-7*max_amp+z_target, z_target, z_target + max_amp, 7*max_amp+z_target]
            ax2.set_xlim(xticks[0], xticks[-1])
            ax2.set_xlabel(r"$E(r)$ [a.u.]")
            ax2.set_ylabel(r"$r$ [m]")
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(["", 0, max_amp, ""])

        ### Axis line
        zero_axis = np.zeros(len(z))
        ax.plot(z, zero_axis, color = "grey", linestyle = "-.", linewidth = 1)
        ### Target position
        plt.axvline(x = z_target, color = "black", linestyle = "--")


        ### Axis labels
        ax.set_xlabel(r"$z$ [m]")
        ax.set_ylabel(r"$w(z)$ [m]")
        plt.ticklabel_format(axis="y", style="sci", scilimits=[-3,3])

        ax.legend(loc = 1)

        ### Set plot constraints
        max_w = max([max(b.w) for b in self.beams])
        if len(self.beams) <= len(colors):
            ax.set_ylim((-1.2*max_w, 1.2*max_w))
        else: 
            ax.set_ylim((0., 1.2*max_w))
        
        plt.show()

    def get_beam_profiles(self, z_target = 0., prop_phase = False, Gouy_phase = False):
        ### Rayleigh length
        z_R = lambda w_0, lamb: np.power(w_0, 2) * np.pi / lamb

        ### Waist radius in z for lambda (lamb) and focus (z_0)
        w = lambda w_0, lamb, z_0, z: w_0*np.sqrt(1 + np.power((z-z_0)/z_R(w_0, lamb), 2))

        ### 1/R term – without the singularity
        R_reciprocal = lambda z, w_0, lamb: z / (np.power(z_R(w_0, lamb), 2) +\
            np.power(z, 2))

        ### Complex gaussian beam in (r, z), focus (z_0) corresponding to the 
        ### beam with lambda (lamb), beam waist (w_0) and amplitude (amp)
        ### Note: without the Gouy phase term
        GaussianBeam = lambda r, z, z_0, w_0, lamb, amp: \
            amp*w_0/w(w_0, lamb, z_0, z)*np.exp(-np.power(r/w(w_0, lamb, z_0, z), 2))*\
            np.exp(-1j*2*np.pi/lamb*r*r/2*R_reciprocal(z-z_0, w_0, lamb) ) 

        ### Gouy phase term
        phi_G = lambda z, z_0, w_0, lamb: np.exp(1j*np.arctan((z-z_0)/z_R(w_0, lamb)))
        
        w_max = max([w(b.w_0, b.lamb, b.z_0, z_target) for b in self.beams])
        N_pts = max([b.N_pts for b in self.beams])
        z_R_max = max([z_R(b.w_0, b.lamb) for b in self.beams])
    
        z = np.linspace(-z_R_max + z_target, z_R_max + z_target, N_pts)
        ### Reach FWHM for all fields
        rho = np.linspace(0, np.sqrt(2)*w_max, N_pts) 
        
        for b in self.beams:
            b.w = w(b.w_0, b.lamb, b.z_0, z)
            b.profile = np.array([GaussianBeam(
                                        r, z_target, b.z_0, b.w_0, 
                                        b.lamb, b.amplitude) for r in rho])
            ### Add propagation phase (kz)
            if (prop_phase):
                b.profile = b.profile*np.exp(1j*2*np.pi/b.lamb*z_target)

            ### Add Gouy phase (phi_G)
            if (Gouy_phase):
                b.profile = b.profile*phi_G(z_target, b.z_0, b.w_0, b.lamb)

        #beam_profile = self.beams[1].profile
        #E_im, E_re = np.imag(beam_profile), np.real(beam_profile)
        ### Obtain the phase and normalize to pi radians
        #R_phase = np.arctan2(E_im, E_re)/np.pi
        #plt.plot(rho, R_phase)
        #plt.show()

        return [rho, z]
                
                


                

