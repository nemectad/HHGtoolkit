'''
Module plotting
===============
Author: Tadeáš Němec 2022, CTU FJFI

Module containing methods for data visualisation from the SFA code. 

List of plotting methods:
-------------------------

* plot_spectrum
* plot_scan
* plot_dipole
* plot_dipole_polar
* plot_stokes_params
* plot_integrand
* plot_Gabor_transform
* plot_spectrum_multiple
* plot_integrals
* plot_spectral_distribution


Note: 
----
Module plots labels in default LaTeX visual style. This option can be
disabled by changing the following line of the code to False:

    matplotlib.rcParams.update({
        "text.usetex": False
    })

'''



import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize
import HHGtoolkit.utilities as util
from HHGtoolkit.utilities import Data     # import data class
import HHGtoolkit.conversions as conv
import HHGtoolkit.process as process
from scipy.signal.windows import *
import HHGtoolkit.Hfn2 as hfn
import mynumerics.mynumerics as mn


matplotlib.rcParams.update({
    "text.usetex": False
})

### Plot dipole fourier transform
def plot_spectrum(
        data,
        plot_component = "zx",
        omega_min = 0,
        omega_max = 50,
        plot_scale = 'log',
        apply_window = False,
        win = blackman,
        legend = False,
        y_min = 1e-8, 
        plot_energy = False,
        plot_3D = False, 
        figsize = (7.5, 3.5),
        plot_eV = False,
        filter = "bandpass",
        filter_harmonics = None
    ):
    '''
    Plot spectrum
    =============

    plot_spectrum(
        data,
        plot_component = "zx",
        omega_min = 0,
        omega_max = 50,
        plot_scale = 'log',
        apply_window = False,
        win = blackman,
        legend = False,
        y_min = 1e-8, 
        plot_energy = False,
        plot_3D = False,
        figsize = (7.5, 3.5)
        plot_eV = False,
        filter = "bandpass",
        filter_harmonics = None
    ):

    Plots dipole harmonic spectrum.

    Parameters:
    -----------
    data: Data or str
        Data to be plotted.
    plot_component: {'zx', 'z', 'x'}, optional, default: 'zx'
        The plotted component.
    omega_min: int, optional, default: 0
        Minimum harmonics plotted.
    omega_max: int, optional, default: 50
        Maximum harmonics plotted. 
    plot_scale: {'log', 'linear'}, optional, default: 'log'
        Plot with logarithmic or linear scale.
    apply_window: bool, optional, default: False
        Apply windowing function on the signal.
    win: function, optional, default: blackman
        Select window function. 
        Note: module 'scipy.signal.windows' must be imported.
    legend: bool, optional, default: False
        Show legend in the plot.
    y_min: float, optional, default: 1e-8
        Minimum plotting value on the y-axis.
    plot_energy: bool, optional, default: False
        Show secondary energy axis in the plot.
    plot_3D: bool, optional, default: False
        Plot spectrum in 3 dimensions.
    figsize: tuple, optional, default: (7.5, 3.5)
        Set figure size in inches - (width, height)
    plot_eV: bool, optional, default: False,
        Electronvolts (eV) on x-axis instead of harmonics.
    filter: {'bandpass', 'lowpass'}, optional, default: 'bandpass'
        Select spectral filter
    filter_harmonics: int, optional, default: None
        Filter chosen harmonics from the spectrum.
    '''

    ### Check the data
    data = util.load_data(data)

    ### Init variables

    ### Time grid
    t = data.t  
    ### Frequency of the fundamental field
    omega_0 = data.omega_1
    freq_0 = omega_0/ (2 * np.pi)
    ### Time step
    T_step = t[1]-t[0]
    ### Number of steps per frequency of the fundamental field
    N_steps_per_freq = int(freq_0*t[-1]) + 1
    ### Result dipole
    dip_z, dip_x = data.d_z, data.d_x

    ### Applying window on dipole
    if apply_window:
        dip_z, dip_x = process.window(dip_z, win), process.window(dip_x, win)

    ### Filter spectrum
    if filter_harmonics != None:
        dip_z, dip_x = process.harmonic_filter([dip_z,dip_x], t[-1], freq_0,  
                        filter_harmonics, filter)

    ### Compute spectrum and time domain for spectrum
    spectrum, tf = process.spectrum([dip_z, dip_x], T_step)
    spectrum_z, spectrum_x = spectrum

    ### Set omega values
    omegas = process.set_omegas(tf, freq_0)
    omegas = omegas[0:omega_max+1]

    ### 3D plot
    if plot_3D:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        if plot_scale == "log":
            spectrum_z = process.linear_to_log(spectrum_z)
            spectrum_x = process.linear_to_log(spectrum_x)

        ### Range for plotting
        range_ = slice(int(N_steps_per_freq*omega_min),int(N_steps_per_freq*omega_max)) 

        ax.plot3D(
            spectrum_z[range_],
            spectrum_x[range_],
            tf[range_],
            zdir="x",
            label=r"$|D(\omega)|$",
            color="black",
            linewidth="1"
        )
        
    ### 2D plot
    else:
        fig, ax = plt.subplots()

        if plot_component == "z" or plot_component == "zx":
            ax.plot(
                tf,
                spectrum_z,
                label=r"$|D_{z}(\omega)|$",
                color="black",
                linewidth=1,
            )
        if plot_component == "x" or plot_component == "zx":
            ax.plot(
                tf,
                spectrum_x,
                label=r"$|D_{x}(\omega)|$",
                color="crimson",
                linewidth=1,
            )


    ### Set ticks corresponding to harmonics:
    if plot_eV:
        set_eV_labels(ax, omega_min, omega_max, freq_0)
    else:
        set_omega_labels(ax, omegas)

    ### Setting legend, axis and size
    if legend:
        ax.legend(loc=1)

    ax.set_xlim(omegas[omega_min], omegas[omega_max])
    ax.set_ylim(y_min, max([spectrum_z[N_steps_per_freq*omega_min:-1].max(),
                spectrum_x[N_steps_per_freq*omega_min:-1].max()])*1.2)
    ax.set_ylabel(r"$|D(\omega)|$ [arb.u.]", fontsize=13)

    if plot_3D:
        ax.set_zlim(y_min, max([spectrum_z[N_steps_per_freq*omega_min:-1].max(),
                spectrum_x[N_steps_per_freq*omega_min:-1].max()])*1.2)
        ax.set_zlabel(r"$|D_x(\omega)|$ [arb.u.]", fontsize=13)
        ax.set_ylabel(r"$|D_z(\omega)|$ [arb.u.]", fontsize=13)
    else:
        plt.yscale(plot_scale)
        fig.set_size_inches(figsize)
        fig.tight_layout()
        ### Show odd harmonics with colored lines
        if not plot_eV:
            for i, oddOmeg in enumerate(omegas):
                if (i % 2) != 0:
                    plt.axvline(x=oddOmeg, color="black", linestyle=":", linewidth=0.5)
    
    if plot_energy and not plot_3D:
        I_p = data.I_p  
        plt.axvline(x=I_p / (2 * np.pi), color="black", linestyle=":", linewidth=1)

        set_energy_axis(ax, data, omega_max)

        fig.tight_layout()
    
    plt.show()


def plot_scan(
        dataset, 
        plot_component = "z",
        omega_min = 0, 
        omega_max = 50, 
        N_y_ticks = 5,
        z_min = 1e-8,
        plot_energy = False,
        plot_scale = 'log',
        scan_variable = 'phase',
        normalize = False, 
        figsize = (9,4),
        intensity = False
    ):
    '''
    Plot scan
    =========

    plot_scan(
        dataset, 
        plot_component = "z",
        omega_min = 0, 
        omega_max = 50, 
        N_y_ticks = 5,
        z_min = 1e-8,
        plot_energy = False,
        plot_scale = 'log',
        scan_variable = 'phase',
        normalize = False,
        figsize = (9, 4),
        intensity = False
    ):

    Plot phase scan of the harmonic spectrum.

    Parameters:
    -----------
    dataset: Dataset or str
        Loaded dataset or the name of the directory with slash '/'.
    plot_component: {'z', 'x'}, optional, default: 'z'
        The plotted component.
    omega_min: int, optional, default: 0
        Minimum harmonics plotted.
    omega_max: int, optional, default: 50
        Maximum harmonics plotted. 
    N_y_ticks: int, optional, default: 5
        Set number of ticks on the y-axis.
    z_min: float, optional, default: 1e-8
        Minimum plotting value on the colormesh.
    plot_energy: bool, optional, default: False
        Show secondary energy axis in the plot.
    plot_scale: {'log', 'linear'}, optional, default: 'log'
        Plot with logarithmic or linear scale.
    scan_variable: {'phase', 'intensity', 'amplitude', 'theta_waveplate', 'delay'}, optional, default: 'phase'
        Plot y-axis labels corresponding to scan in phase, intensity or amplitude.
    normalize: bool, optional, default: False
        Normalize each slice maximum to one. 
    figsize: tuple, optional, default: (9, 4)
        Set figure size in inches - (width, height)
    intensity: bool, optional, default: False
        Plot the intensity of the harmonic signal, i.e. coherent sum of |D_x|^2 and |D_z|^2
    '''
    ### Loading data
    ds = util.load_dataset(dataset)
    
    ### Init variables
    t = ds[0].t
    T_step = t[1]-t[0]
    omega_0 = ds[0].omega_1
    freq_0 = omega_0/ (2*np.pi)
    N_steps_per_freq = int(freq_0*t[-1]) + 1

    if intensity:
        N_max = np.argmax([d.t[-1] for d in ds])
        #t_max = ds[N_max].t
        N = len(ds[N_max].d_z)
        z = 1e-13*np.ones((len(ds),N))
        x = 1e-13*np.ones((len(ds),N))
        ### Compute spectrum and time domain for spectrum
        for i, d in enumerate(ds):
            x[i,0:len(d.d_x)] = d.d_x
            z[i,0:len(d.d_z)] = d.d_z

        z, tf = process.spectrum(z, T_step)
        x, tf = process.spectrum(x, T_step)
        lbl = ""

        z = np.power(z, 2) + np.power(x, 2)

    else:
        N_max = np.argmax([d.t[-1] for d in ds])
        #t_max = ds[N_max].t
        N = len(ds[N_max].d_z)
        z = 1e-13*np.ones((len(ds),N))
        ### Compute spectrum and time domain for spectrum
        if (plot_component == "x"):
            for i, d in enumerate(ds):
                z[i,0:len(d.d_x)] = d.d_x
            z, tf = process.spectrum(z, T_step)
            lbl = r"$|D_x(\omega)|$"
        else:
            for i, d in enumerate(ds):
                z[i,0:len(d.d_z)] = d.d_z
            z, tf = process.spectrum(z, T_step)
            lbl = r"$|D_z(\omega)|$"


    ### Setting ticks
    ### x-axis
    omegas = process.set_omegas(tf, freq_0)
    omegas = omegas[0:omega_max+1]

    if normalize:
        for i in range(0,len(z)):
            z[i,:] = z[i,:]/z[i,N_steps_per_freq*omega_min:-1].max()

    ### Colormesh plot
    z_max = z[:,N_steps_per_freq*omega_min:-1].max()

    fig, ax = plt.subplots()
    
    if plot_scale == 'log':
        col = LogNorm(vmin = z_min, vmax = z_max)
    elif plot_scale == 'linear':
        col = Normalize(vmin = z_min, vmax = z_max)
    else:
        raise ValueError("Unknown scale '" + plot_scale +"'. "
                         "Available options are 'log' and 'linear'")
        

    c = ax.pcolormesh(tf, range(0,len(z)), z,
        cmap = 'jet',
        shading = 'gouraud',
        norm = col
    )

    ax.set_title(lbl)

    if intensity:
        fig.colorbar(c, ax=ax, label=r"$|D(\omega)|^2$ [arb.u.]")
    else:
        fig.colorbar(c, ax=ax, label=r"$|D(\omega)|$ [arb.u.]")

    ### Setting ticks
    ### y-axis
    yticks = np.linspace(0, len(z)-1, N_y_ticks)

    ### Set ticks correspondingly:
    set_omega_labels(ax, omegas)
    ax.set_yticks(yticks)

    ### Define y labels
    if scan_variable == 'phase':    
        ### Set initial CEP_2 value
        omega_1 = ds[0].omega_1
        omega_2 = ds[0].omega_2

        rel_omega = int(np.round(omega_2/omega_1))
        if (ds[0].CEP_1 == 0.5):
            cos_pulse = True
        else:
            cos_pulse = False
        CEP_2 = conv.secondary_field(omega_1, rel_omega, 0, 0, 0, 0, cos_pulse)[3]

        ### y-axis
        vals_y = -CEP_2 + np.linspace(ds[0].CEP_2, ds[-1].CEP_2, N_y_ticks)
        labels_y = [r"${:.1f}$".format(val) for val in vals_y]

        axis_label = r"$\varphi \, [\pi \, $rad$]$"
    elif scan_variable == 'intensity':
        if plot_component == 'z':
            I_min = conv.E_to_intensity(ds[0].E0_1)
            I_max = conv.E_to_intensity(ds[-1].E0_1)
        else:
            I_min = conv.E_to_intensity(ds[0].E0_2)
            I_min = conv.E_to_intensity(ds[-1].E0_2)
        vals_y = np.linspace(I_min, I_max, N_y_ticks)
        labels_y = [r"${:.2e}$".format(val) for val in vals_y]

        axis_label = r"$I$ [W/cm$^2$]"
    elif scan_variable == 'amplitude':
        A_min = ds[0].E0_2
        A_max = ds[-1].E0_2

        vals_y = np.linspace(A_min, A_max, N_y_ticks)/A_max
        labels_y = [r"${:.2f}$".format(val) for val in vals_y]

        axis_label = r"$E/E_0$"
    elif scan_variable == 'theta_waveplate':
        try:
            #vals_y = [d.theta_1+90 for d in ds]
            vals_y = np.linspace(90 + ds[0].theta_1, 90 + ds[-1].theta_1, N_y_ticks)
            #vals_y = vals_y[0:-1:len(vals_y)//(N_y_ticks+1)]
            labels_y = [r"${:.2f}$°".format(val) for val in vals_y]
            #labels_y = labels_y[0:-1:len(vals_y)//(N_y_ticks+1)]
            axis_label = r"$\theta_{MO}$"
        except AttributeError:
            print("No theta information is provided in the data.")
    elif scan_variable == 'delay':
        try:
            t_min = ds[0].delay
            t_max = ds[-1].delay
        except AttributeError:
            print("No time delay information is provided in the data.")

        vals_y = np.linspace(t_min, t_max, N_y_ticks)
        labels_y = [r"${:.1f}$".format(val) for val in vals_y]
        axis_label = r"$\tau_{delay}$"
    else:
        raise ValueError("Unknown scan variable '{}'".format(scan_variable))

    ### Set labels
    ax.set_yticklabels(labels_y, fontsize = 10)
    ax.set_ylabel(axis_label, fontsize = 12)

    ### Set the limits of the plot
    ax.axis([omegas[omega_min], omegas[omega_max], 0, (len(z))-1])

    ### Set secondary energy axis
    if plot_energy:
        set_energy_axis(ax, ds[0], omega_max)

    fig.set_size_inches(figsize)

    plt.show()


### Plot individual dipole in time t
def plot_dipole(
        data, 
        plot_component = 'zx', 
        plot_fields = False, 
        plot_3D = False,
        norm_T = False,
        apply_window = False,
        win = blackman,
        filter = "bandpass",
        filter_harmonics = None,
        figsize = (7,5)
    ):
    '''
    Plot dipole
    ===========

    plot_dipole(
        data,
        plot_component = "zx",
        plot_fields = False,
        apply_window = False,
        norm_T = False,
        win = blackman,
        plot_3D = False,
        filter = "bandpass",
        filter_harmonics = None
    ):

    Plot the dipole in time.

    Parameters:
    -----------
    data: Data or str
        Data to be plotted.
    plot_component: {'zx', 'z', 'x'}, optional, default: 'zx'
        The plotted component.
    plot_fields: bool, optional, default: False
        Plot fields in the background for reference.
    apply_window: bool, optional, default: False
        Apply windowing function on the signal.
    norm_T: bool, optional, default: False
        Normalize time to optical cycles of the fundamental field.
    win: function, optional, default: blackman
        Select window function. 
        Note: module 'scipy.signal.windows' must be imported.
    plot_3D: bool, optional, default: False
        Plot dipole in 3 dimensions.
    filter: {'bandpass', 'lowpass'}, optional, default: 'bandpass'
        Select spectral filter
    filter_harmonics: int, optional, default: None
        Filter chosen harmonics from the spectrum.

    '''

    ### Load data
    data = util.load_data(data)

   ### Init variables

    ### Time grid
    t = data.t  
    ### Frequency of the fundamental field
    omega_0 = data.omega_1
    freq_0 = omega_0/ (2 * np.pi)
    ### Result dipole
    dipole_z, dipole_x = data.d_z, data.d_x

    if apply_window and filter_harmonics == None:
        dipole_z = process.window(dipole_z, win)
        dipole_x = process.window(dipole_x, win)

    ### Filter spectrum
    if filter_harmonics != None:
        dipole_z, dipole_x = process.harmonic_filter([dipole_z,dipole_x], t[-1], freq_0,  
                        filter_harmonics, filter)

    ### Maximum dipole value
    max_dip = np.array([np.abs(dipole_z).max(), np.abs(dipole_x).max()]).max()

    if plot_3D:
        fig = plt.figure()

        ax = plt.axes(projection="3d")

        ### Plot dipole trajectory in 3D
        ax.plot3D(
            dipole_z,
            dipole_x,
            t,
            zdir="x",
            label=r"$D$",
            color="crimson",
            linewidth="1",
        )

        ### Plot dipole components
        ax.plot(
            t,
            dipole_x,
            zdir="y",
            zs=max_dip,
            label=r"$D_{x}$",
            linestyle="-",
            linewidth="0.5",
            color="black",
        )
        ax.plot(
            t,
            dipole_z,
            zdir="z",
            zs=-max_dip,
            label=r"$D_{z}$",
            linestyle="-",
            linewidth="0.5",
            color="gray",
        )
        ax.plot(
            dipole_z,
            dipole_x,
            zdir="x",
            zs=0,
            label=r"$D_{xz}$",
            linestyle=":",
            linewidth="0.5",
            color="black",
        )

        ax.legend(loc=1)
        ax.set_xlim([t[0], t[-1]])
        ax.set_ylim(-max_dip, max_dip)
        ax.set_zlim(-max_dip, max_dip)
        ax.set_xlabel(r"$t$ [a.u.]", fontsize = 10)
        ax.set_ylabel(r"$D_z(t)$ [a.u.]", fontsize = 10)
        ax.set_zlabel(r"$D_x(t)$ [a.u.]", fontsize = 10)

        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.tick_params(axis='z', which='major', labelsize=8)

        plt.show()
    else: 
        ### 2D plot
        fig, ax = plt.subplots()

        ### Set y-range for dipole
        ax.set_ylim(-max_dip, max_dip)

        if plot_fields:
            ax2 = ax.twinx()
            ax2.set_ylabel(r"$E(t)$ [a.u.]", fontsize = 12)

            E_z = data.E_z
            E_x = data.E_x

            ### Set y-range for fields
            max_field = np.array([np.abs(E_z).max(), np.abs(E_x).max()]).max()
            ax2.set_ylim(-max_field, max_field)

        ### Select components and plot
        if plot_component == "z" or plot_component == "zx":
            ax.plot(
                t,
                dipole_z,
                label = r"$D_{z}$",
                color = "black",
                linewidth = 1,
            )
            if plot_fields:
                ax2.plot(
                    t,
                    E_z,
                    label = r"$E_z$",
                    color = "black",
                    linewidth = "0.5",
                    linestyle = ":"
                )
        if plot_component == "x" or plot_component == "zx":
            ax.plot(
                t,
                dipole_x,
                label = r"$D_{x}$",
                color = "crimson",
                linewidth = 1
            )
            if plot_fields:
                ax2.plot(
                    t,
                    E_x,
                    label=r"$E_x$",
                    color="crimson",
                    linewidth="0.5",
                    linestyle = ":"
                )

        ax.legend(loc=1)

        if plot_fields:
            ax2.legend(loc=1)
            ax.legend(loc=2)

        ax.set_ylabel(r"$D(t)$ [a.u.]", fontsize=12)

        ### Norm to the optical cycles of the fundamental field
        if (norm_T):
            norm_to_cycles(ax, 1/freq_0, data.N_cycl_1)
        else:
            ax.set_xlabel(r"$t$ [a.u.]", fontsize=12)

        fig.tight_layout()
        fig.set_size_inches(figsize)

        plt.show()


def plot_dipole_polar(
        data, 
        cycles = [0,-1], 
        plot_fields = False, 
        filter_harmonics = None
    ):
    '''
    Plot dipole polar
    =================

    plot_dipole_polar(
        data,
        cycles = [0,-1]
        plot_fields = False,
        filter_harmonics = None
    ):

    Plot dipole in time from the .h5 dataset in polar coordinates.

    Parameters:
    -----------
    data: Data or str
        Data to be plotted.
    cycles: array_like, optional, default: [0, -1]
        Select timeframe to be plotted with respect to the optical cycles.
        Full pulse: [0, -1] 
    plot_fields: bool, optional, default: False
        Plot field in polar coordinates next to the dipole in time. 
    filter_harmonics: int, optional, default: None
        Filter chosen harmonics from the spectrum.

    '''
    ### Load data
    data = util.load_data(data)
    
    ### Time grid
    t = data.t
    ### Frequency of the fundamental field
    omega_0 = data.omega_1
    freq_0 = omega_0/ (2 * np.pi)
    ### Dipole data
    dipole_z = data.d_z
    dipole_x = data.d_x

    ### Optical cycles
    N_cycl_1 = data.N_cycl_1
    
    ### Pts per optical cycle
    N = len(t)//int(N_cycl_1)

    
    ### Filter spectrum
    if filter_harmonics != None:
        dip_z, dip_x = process.harmonic_filter([dipole_z,dipole_x], t[-1], freq_0,  
                        filter_harmonics, filter)

    if cycles[1] != -1 and cycles[1] > N_cycl_1:
            raise ValueError("Optical cycles exceed maximum cycles in pulse.")        
    if cycles[1] == -1:
        cycles[1] = N_cycl_1

    ### Create slice for optical cycles of interest
    plt_range = slice(int(cycles[0]*N), int(cycles[1]*N))
    
    t = t[plt_range]
    dipole_z = dipole_z[plt_range]
    dipole_x = dipole_x[plt_range]

    ### Create figure
    fig = plt.figure(figsize=(9,5))

    if plot_fields:
        ax1 = fig.add_subplot(121, projection = 'polar')
    else:
        ax1 = fig.add_subplot(111, projection = 'polar')
    ax1.plot(
        np.arctan2(dipole_z, dipole_x),
        np.abs(dipole_z + 1j*dipole_x),
        color = "black",
        linewidth = 1,
        label = r"$D_{\rho,\varphi}(t)$"
    )
    ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    if plot_fields:
        E_z = data.E_z
        E_x = data.E_x
        ax2 = fig.add_subplot(122, projection = 'polar')
        ax2.plot(
            np.arctan2(E_z, E_x),
            np.abs(E_z + 1j*E_x),
            color = "crimson",
            linewidth = 1,
            label = r"$E_{\rho, \varphi}(t)$"
        )
        ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()

    plt.show()


def plot_stokes_params(data, filter_harmonics = None):
    '''
    Plot Stokes parameters
    ======================

    plot_stokes_params(data, filter_harmonics = None)

    Plot visualization of the Stokes parameters computed from the data.

    Parameters:
    -----------
    data: Data or str
        Data for the visualization.
    filter_hramonics: int, optional, default: None
        Filter arbitrary harmonic frequency.
    '''
    S_0, S_1, S_2, S_3 = process.stokes_params_from_file(data, norm = True, filter_harmonics= filter_harmonics)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ### Plot sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='grey', alpha = 0.05)

    ### Contour spherical grid
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="grey", linewidth = 0.5, alpha = 0.3)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ### Set S axes
    ax.quiver(0,0,0,1,0,0,length=1.0, color = 'gray', linewidth = 1)
    ax.text(0.8, 0.1, 0.1, r"$S_1$", (1,0,0))
    ax.set_xticks([-1, 0, 1])

    ax.quiver(0,0,0,0,1,0,length=1.0, color = 'gray', linewidth = 1)
    ax.text(0.1, 0.8, 0.1, r"$S_2$", (0,1,0))
    ax.set_yticks([-1, 0, 1])

    ax.quiver(0,0,0,0,0,1,length=1.0, color = 'gray', linewidth = 1)
    ax.text(0.1, 0.1, 0.8, r"$S_3$", (0,0,1))
    ax.set_zticks([-1, 0, 1])

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25., azim=45.)

    ### Plot Stokes parameters
    
    #print("Stokes parameters: S_0 = ", S_0, "S_1 = " , S_1, " , ", "S_2 = ", S_2, " , ", "S_3 = ", S_3)
    ax.quiver(0, 0, 0, S_1, S_2, S_3, length = S_0, color = 'red')
    ax.quiver(0, 0, 0, S_1, 0, 0, color = 'red', linewidth=0.5)
    ax.quiver(0, 0, 0, 0, S_2, 0, color = 'red', linewidth=0.5)
    ax.quiver(0, 0, 0, 0, 0, S_3, color = 'red', linewidth=0.5)
    ax.text(S_1, S_2, S_3, r"$\mathbf{S} = $"+r"$({:.2f}, {:.2f}, {:.2f})$".format(S_1, S_2, S_3), color = 'r')


    plt.show()


def plot_integrand(
        data, 
        mode = "abs", 
        plot_fields = True, 
        plot_component = "zx",
        figsize = (8, 4)
    ):
    '''
    Plot integrand
    ==============

    plot_integrand(
        data,
        mode = "abs",
        plot_fields = True,
        plot_component = "zx"
    ):

    Plot z and x component of the integrand from the SFA formula and the 
    corresponding electric field. 

    Parameters:
    -----------
    data: Data or str
        Data to be plotted.
    mode: {'abs', 're', 'im'}, optional, default: 'abs'
        Plot either modulus of the integrand or its real or imaginary part.
    plot_fields: bool, optional, default: True
        Plot the field along the integrand.
    plot_component: {'zx', 'x', 'z'}, optional, default: 'zx'
        Plotted component of the integrand.
    figsize: tuple, optional, default: (8, 4)
        Set figure size in inches - (width, height)
    '''

    ### Check and load the data
    data = util.load_data(data)
    try:
        ### Init data
        integrand_z = data.I_z
        integrand_x = data.I_x
        ### Time grid
        t = data.t
        ### Number of points per cycle for integration
        N_int = data.N_int
        ### Field frequencies
        omega_1 = data.omega_1
        omega_2 = data.omega_2
        ### Number of points for integration for full time
        N = max([10, int(np.floor((t[-1]-t[0])*omega_1/2/np.pi*N_int)), 
                int(np.floor((t[-1]-t[0])*omega_2/2/np.pi*N_int))])

        labels = []

        fig, ax = plt.subplots()

        if mode == "im":
            integrand_z = np.imag(integrand_z)
            integrand_x = np.imag(integrand_x)
            labels = [r"$\Im [$Integrand$_{z}]$", r"$\Im [$Integrand$_{x}]$"]
        elif mode == "re":
            integrand_z = np.real(integrand_z)
            integrand_x = np.real(integrand_x)
            labels = [r"$\Re [$Integrand$_{z}]$", r"$\Re [$Integrand$_{x}]$"]
        elif mode == "abs":
            integrand_z = np.abs(integrand_z)
            integrand_x = np.abs(integrand_x)
            labels = [r"$|$Integrand$_{z}|$", r"$|$Integrand$_{x}|$"]
        else:
            integrand_z = np.real(integrand_z)
            integrand_x = np.real(integrand_x)
            labels = [r"$\Re [$Integrand$_{z}]$", r"$\Re [$Integrand$_{x}]$"]

        max_int = np.array([integrand_z.max(), integrand_x.max()]).max()
        ax.set_ylim(-max_int, max_int)

        ### Create range for plotting
        integr_z = np.zeros(N)
        integr_x = np.zeros(N)

        ### Broadcasting values so it corresponds to time exactly
        integr_z[0:len(integrand_z)] = integrand_z
        integr_x[0:len(integrand_z)] = integrand_x

        ### Save ranges
        integrand_z = integr_z
        integrand_x = integr_x

        ### Final plotting range for integrand
        t_int = np.linspace(t[0], t[-1], N)

        if plot_fields:
            ax2 = ax.twinx()

        if plot_component == "zx" or plot_component == "z":
            ax.plot(
                t_int,
                integrand_z,
                label=labels[0],
                color = "black",
                linewidth = 1,
            )
        if plot_component == "zx" or plot_component == "x":
            ax.plot(
                t_int,
                integrand_x,
                label=labels[1],
                color = "crimson",
                linewidth = 1,
            )
        ax.legend(loc=1)

        if plot_fields:
            E_z = data.E_z
            E_x = data.E_x

            max_field = np.array([np.abs(E_z).max(), np.abs(E_x).max()]).max()
            ax2.set_ylim(-max_field, max_field)

            if plot_component == "zx" or plot_component == "z":
                ax2.plot(
                    t,
                    E_z,
                    label = r"$E_z$",
                    color = "black",
                    linewidth = "0.5",
                    linestyle = ":"
                )
            if plot_component == "zx" or plot_component == "x":
                ax2.plot(
                    t,
                    E_x,
                    label=r"$E_x$",
                    color="crimson",
                    linewidth="0.5",
                    linestyle = ":"
                )
            ax2.legend(loc=1)
            ax.legend(loc=2)
            ax2.set_ylabel(r"$E(t)$ [a.u.]", fontsize = 12)

        ax.set_ylabel(r"Integrand [arb.u.]", fontsize=12)
        ax.set_xlabel(r"$t$ [a.u.]", fontsize=12)

        fig.set_size_inches(figsize)
        fig.tight_layout()
        plt.show()
    except AttributeError:
        print("No integrand data available.")


### Plot fields of the laser in 3D or using 2D projection
def plot_fields(
        data, 
        plot_3D = True, 
        norm_T = False, 
        cycles = None, 
        legend = True,
        figsize = (7,5)
    ):
    '''
    Plot fields
    ===========

    plot_fields(
        data,
        plot_3D = True,
        norm_T = False,
        cycles = [0, -1],
        legend = True,
        figsize = (7,5)
    ):

    Plot electric fields of the laser in 3D or in 2D projection. 

    Parameters:
    -----------
    data: Data or str
        Data for plotting.
    plot_3D: bool, optional, default: True
        Plot fields in 3D.
    norm_T: bool, optional, default: False
        Norm time axis to the optical cycles of the fundamental field.
    cycles: array, default: None
        Limit number of cycles of the field for plotting, -1: end of the pulse.
        Example: plot from cycle 4 to the last one: cycles = [4, -1]
    legend: bool, optional, default: True
        Set legend visible.
    figsize: tuple, optional, default: (7, 5)
        Set figure size in inches - (width, height)
    '''

    ### Check and load the data
    data = util.load_data(data)

    ### Electric fields
    E0_1 = data.E0_1
    E0_2 = data.E0_2
    E_z = data.E_z
    E_x = data.E_x
    ### Optical cycles
    N_cycl_1 = data.N_cycl_1
    ### Time grid
    t = data.t
    ### Number of point per one cycle
    N = len(t)//int(N_cycl_1)
    ### Maximum of the electric field
    E_max = np.max([np.max(np.abs(E_z)), np.max(np.abs(E_x))])

    if cycles != None:
        if cycles[1] != -1 and cycles[1] > N_cycl_1:
            raise ValueError("Optical cycles exceed maximum cycles in pulse.")
        if cycles[1] == -1:
            cycles[1] = N_cycl_1
        ### Create slice for optical cycles of interest
        plt_range = slice(int(cycles[0]*N), int(cycles[1]*N))

        ### Time frame
        t = t[plt_range]

        ### Fields data
        E_z = E_z[plt_range]
        E_x = E_x[plt_range]
        
        E_max = np.max([np.max(np.abs(E_z)), np.max(np.abs(E_x))])

    if plot_3D:
        fig = plt.subplots()

        ax = plt.axes(projection="3d")

        ### Plot sum of fields
        ax.plot3D(
            E_z,
            E_x,
            t,
            zdir="x",
            label=r"$E$",
            color="crimson",
            linewidth="1",
        )

        ### Plot fields components
        ax.plot(
            t,
            E_x,
            zdir="y",
            #zs=(np.abs(E0_1) + np.abs(E0_2)),
            zs=(1.2*E_max),
            label=r"$E_x$",
            linestyle="-",
            linewidth="0.5",
            color="black",
        )
        ax.plot(
            t,
            E_z,
            zdir="z",
            #zs=-(np.abs(E0_1) + np.abs(E0_2)),
            zs=(-1.2*E_max),
            label=r"$E_z$",
            linestyle="-",
            linewidth="0.5",
            color="gray",
        )
        ax.plot(
            E_z,
            E_x,
            zdir="x",
            zs=t[0],
            label=r"$E_{xz}$",
            linestyle=":",
            linewidth="0.5",
            color="black",
        )

        if legend:
            ax.legend(loc=1)

        ax.set_ylim(-1.2*E_max, 1.2*E_max)
        ax.set_zlim(-1.2*E_max, 1.2*E_max)

        #E_ticks = [-np.max([np.max(E_z), np.max(E_x)]), 0, np.max([np.max(E_z), np.max(E_x)])]
        #E_ticks_lbl = [r"$-E_{0}$", "0", r"$E_{0}$"]

        ### Norm to the optical cycles of the fundamental field
        if (norm_T):
            norm_to_cycles(ax, 2*np.pi/data.omega_1, N_cycl_1)
        else:
            ax.set_xlabel(r"$t$ [a.u.]", fontsize=12)

        #ax.set_yticks(E_ticks)
        #ax.set_zticks(E_ticks)
        #ax.set_zticklabels(E_ticks_lbl)
        #ax.set_yticklabels(E_ticks_lbl)
        ax.grid(False)

        plt.show()
    else:
        ### 2-D fields
        fig, ax = plt.subplots()

        if (norm_T):
            norm_to_cycles(ax, 2*np.pi/data.omega_1, N_cycl_1, spacing=3)
        else:
            ax.set_xlabel(r"$t$ [a.u.]", fontsize=12)

        ax.plot(
            t,
            E_z,
            label=r"$E_z$",
            color="black",
            linewidth="1",
        )
        ax.plot(
            t,
            E_x,
            label=r"$E_x$",
            color="crimson",
            linewidth="1",
        )
        ax.legend(loc=1)
        ax.set_ylabel(r"$E(t)$ [a.u.]", fontsize = 12)
        fig.tight_layout()
        fig.set_size_inches(figsize)
        plt.show()


### This function computes and plots Gabor transform of the dipole in time t
def plot_Gabor_transform(
        data, 
        plot_component = "z",
        t_min = 0, 
        t_max = 1600, 
        N_steps = 400,
        omega_min = 0, 
        omega_max = 40, 
        a = 8,
        norm_T = False,
        z_min = 1e-7,
        plot_scale = 'log',
        plot_energy = False,
        figsize = (7,5),
        plot_eV = False
    ):
    '''
    Plot Gabor transform
    ====================

    plot_Gabor_transform(
        data, 
        plot_component = "z",
        t_min = 0, 
        t_max = 1600, 
        N_steps = 400,
        omega_min = 0, 
        omega_max = 40, 
        a = 8,
        norm_T = False,
        z_min = 1e-7,
        plot_scale = 'log',
        plot_energy = False,
        figsize = (7,5),
        plot_eV = False
    ):

    Plot Gabor transform, i.e. time-frequency analysis of the dipole. 

    Parameters:
    -----------
    data: Data or str
        Data to be plotted.
    plot_component: {'zx', 'z', 'x'}, optional, default: 'zx'
        The plotted component.
    t_min: float, optional, default: 0
        Minimum time for the transform.
    t_max: float, optional, default: 1600
        Maximum time for the transform.
    N_steps: int, optional, default: 400
        Number of points for discretization. 
    omega_min: int, optional, default: 0
        Minimum harmonics plotted.
    omega_max: int, optional, default: 40
        Maximum harmonics plotted. 
    a: float, optional, default: 8
        Width of the Gaussian kernel.
    norm_T: bool, optional, default: False
        Normalize time to optical cycles of the fundamental field.
    z_min: float, optional, default: 1e-7
        Minimum plotted value.
    plot_scale: {'log', 'linear'}, optional, default: 'log'
        Plot with logarithmic or linear scale.
    plot_energy: bool, optional, default: False
        Show secondary energy axis in the plot.
    figsize: tuple, optional, default: (7, 5)
        Set figure size in inches - (width, height)
    plot_eV: bool, optional, default: False,
        Electronvolts (eV) on x-axis instead of harmonics.
    '''

    ### Data loading
    data = util.load_data(data)

    ### Time grid
    t = data.t  
    ### Frequency of the fundamental field
    omega_0 = data.omega_1
    freq_0 = omega_0/ (2 * np.pi)
    ### Time step
    T_step = t[1]-t[0]
    ### Number of steps per frequency of the fundamental field
    N_steps_per_freq = int(freq_0*(t[-1]-t[0])) + 1
    N_cycl_1 = data.N_cycl_1

    ### Number of samplepoints
    N = len(t)

    ### Time domain for FFT
    T_step = t[1]-t[0]
    t_f = np.linspace(0, 1.0 / (2.0 * T_step), N//2)

    ### Time domain for Gabor transform
    t_0 = np.linspace(t_min, t_max, N_steps)

    if (plot_component == "x"):
        Gabor_transf = process.gabor_transf(data.d_x, t, t_min, t_max, N_steps, a)
    else:
        if (plot_component != "z"):
            print("Plotting z component.")
        Gabor_transf = process.gabor_transf(data.d_z, t, t_min, t_max, N_steps, a)

    ### Omegas
    omegas = process.set_omegas(t_f, freq_0)
    omegas = omegas[0:int(omega_max)+1]

    ### Set boundaries - account for minimum omega
    z_max = Gabor_transf[:, int(N_steps_per_freq*omega_min): (N//2)].max()

    ### Plotting colormesh - account for maximum omega (faster)
    fig, ax = plt.subplots()

    if plot_scale == 'log':
        col = LogNorm(vmin = z_min, vmax = z_max)
    elif plot_scale == 'linear':
        col = Normalize(vmin = z_min, vmax = z_max)
    else:
        raise ValueError("Unknown scale '" + plot_scale +"'. "
                         "Available options are 'log' and 'linear'")

    c = ax.pcolormesh(
        t_0, 
        t_f[0:int(N_steps_per_freq*omega_max)], 
        np.transpose(Gabor_transf[:, 0 : int(N_steps_per_freq*omega_max)]),
        cmap = 'jet',
        shading = 'gouraud',
        norm = col
    )

    ### Set ticks corresponding to harmonics:
    if not plot_eV:
        ax.set_yticks(omegas)

        ### Tick labels
        labels_y = [item.get_text() for item in ax.get_yticklabels()]
        for x, val in enumerate(labels_y):
            # Only 1, 5, 9 etc. harmonics are shown
            if (int(x) - 1) % 4 == 0:
                labels_y[int(x)] = r"${}$".format(x)
            else:
                labels_y[int(x)] = ""

        ylabel = "H [-]"

    else:
        E_eV = conv.eV_per_harmonic(freq_0)

        E_min = omega_min*E_eV
        E_max = omega_max*E_eV

        Energs = np.arange(0, E_max, 50)

        Es = []
        for E in Energs:
            if E < E_min:
                continue
            else:
                Es.append(E)

        Energs = np.array(Es)
    
        E_ticks = Energs/E_eV*omega_0

        ax.set_yticks(E_ticks)

        labels_y = Energs

        ylabel = r"$E$ [eV]"

    ### Set labels & ticks
    ax.set_yticklabels(labels_y, fontsize = 12)

    ax.set_ylabel(ylabel, fontsize = 12)

    ### Norm to the optical cycles of the fundamental field
    if (norm_T):
        norm_to_cycles(ax, 1/freq_0, N_cycl_1, spacing=2)
    else:
        ax.set_xlabel(r"$t$ [a.u.]", fontsize=12)


    ### Set limits for the plot
    ax.axis([t_min, t_max, omegas[int(omega_min)], omegas[int(omega_max)]])

    if plot_energy:
        ### Show Ip
        I_p = data.I_p
        
        ### Ponderomotive potential
        E0_1 = np.max([data.E0_2, data.E0_1])
        U_p = conv.compute_Up(E0_1, freq_0)

        ### Secondary energy axis
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
    
        energies = (I_p + np.arange(0,omega_0*(omega_max)/U_p, 0.5*U_p))/(2*np.pi)
        ax2.set_yticks(energies)

        energies_ticks = []
        for i in range(0, len(energies)):
            if i%2 == 0:
                energies_ticks.append(r"${}$".format(int(i*0.5)))
            else:
                energies_ticks.append("")

        ax2.set_yticklabels(energies_ticks, fontsize = 12)
        ax2.set_ylabel(r"$(E-I_p)/U_p$", fontsize = 12)

        fig.colorbar(c, ax=ax, label=r"$|D(\omega)|$ [arb.u.]", pad = 0.1, orientation="vertical")
    else:
        fig.colorbar(c, ax=ax, label=r"$|D(\omega)|$ [arb.u.]")

    fig.set_size_inches(figsize)

    plt.show()

def plot_spectrum_multiple(
        dataset,
        plot_component = "z",
        omega_min = 0,
        omega_max = 50,
        plot_scale = 'log',
        apply_window = False,
        win = blackman,
        legend = False,
        y_min = 1e-8, 
        plot_energy = False,
        labels = [],
        figsize = (9,4)
    ):
    """
    Plot spectrum multiple
    ======================

    plot_spectrum_multiple(
        dataset,
        plot_component = "zx",
        omega_min = 0,
        omega_max = 50,
        plot_scale = 'log',
        apply_window = False,
        win = blackman,
        legend = False,
        y_min = 1e-8, 
        plot_energy = False,
        figsize = (9, 4)
    ):

    Plot multiple dipole harmonic spectra along each other.

    Parameters:
    -----------
    data: Dataset or str
        Dataset or array of files to be plotted.
    plot_component: {'zx', 'z', 'x'}, optional, default: 'zx'
        The plotted component.
    omega_min: int, optional, default: 0
        Minimum harmonics plotted.
    omega_max: int, optional, default: 50
        Maximum harmonics plotted. 
    plot_scale: {'log', 'linear'}, optional, default: 'log'
        Plot with logarithmic or linear scale.
    apply_window: bool, optional, default: False
        Apply windowing function on the signal.
    win: function, optional, default: blackman
        Select window function. 
        Note: module 'scipy.signal.windows' must be imported.
    legend: bool, optional, default: False
        Show legend in the plot.
    y_min: float, optional, default: 1e-8
        Minimum plotting value on the y-axis.
    plot_energy: bool, optional, default: False
        Show secondary energy axis in the plot.
    figsize: tuple, optional, default: (7.5, 3.5)
        Set figure size in inches - (width, height)
    """
    ### Loading data
    ds = util.load_dataset(dataset)

    ### Init variables
    t = ds[0].t
    T_step = t[1]-t[0]
    omega_0 = ds[0].omega_1
    freq_0 = omega_0/ (2*np.pi)
    ### Time steps
    T_steps = [d.t[1]-d.t[0] for d in ds]
    T_step = min(T_steps)

    spectra = []
    tfs = []

    if (plot_component == "x"):
        dip = [d.d_x for d in ds]
    else:
        dip = [d.d_z for d in ds]
    
    for d in dip:
        ### Applying window on dipole
        if apply_window:
            d = process.window(d) 
        spectr, tf = process.spectrum(d, T_step)
        spectra.append(spectr)
        tfs.append(tf)

    len_tfs = [len(tf) for tf in tfs]
    N_tf_max, i_tf_max = [max(len_tfs), np.argmax(len_tfs)]
        
    ### Set equal spectrum length
    for spectrum, tf, i in zip(spectra, tfs, range(0, len(tfs))):
        if len(tf) < N_tf_max:
            dum = np.zeros(N_tf_max)+1e-10
            dum[0:len(spectrum)] = spectrum
            spectra[i] = dum
            
    ### Plot solution
    fig, ax = plt.subplots()

    if labels == []:
        labels = [r"$|D_{0}$".format(i)+r"$(\omega)|$" for i in range(0,len(ds))]

    
    for i, spectrum_z in enumerate(spectra):
        ax.plot(
            tfs[i_tf_max],
            spectrum_z,
            label = labels[i],
            linewidth=1,
        )

    ### Setting ticks for dipole radiation
    omegas = process.set_omegas(tfs[i_tf_max], freq_0)
    omegas = omegas[0:omega_max+1]

    N_steps_per_freq = int(np.floor(omegas[1]/tfs[i_tf_max][1]))+1

    ### Set ticks corresponding to harmonics:
    set_omega_labels(ax, omegas)

    ### Show odd harmonics with colored lines
    for i, oddOmeg in enumerate(omegas):
        if (i % 2) != 0:
            plt.axvline(x=oddOmeg, color="black", linestyle=":", linewidth=0.5)

    ### Setting legend, axis and size
    if legend:
        ax.legend(loc=1)

    ax.set_xlim(omegas[omega_min], omegas[omega_max])
    ax.set_ylim(y_min, spectrum_z[N_steps_per_freq*omega_min:-1].max()*1.2)
    ax.set_ylabel(r"$|D(\omega)|$ [arb.u.]", fontsize=12)
    ax.set_xlabel("H [-]", fontsize=12)

    plt.yscale(plot_scale)
    fig.set_size_inches(figsize)

    if plot_energy:
        ### Show Ip
        I_p = ds[0].I_p
        plt.axvline(x=I_p / (2 * np.pi), color="black", linestyle=":", linewidth=1)

        set_energy_axis(ax, ds[0], omega_max)
        
    fig.tight_layout()

    plt.show()


def plot_integrals(
        dataset, 
        plot_component = "z",
        omega_min = 0, 
        omega_max = 50, 
        N_x_ticks = 11,
        z_min = 1e-8,
        scan_variable = 'phase',
        normalize = False,
        figsize = (8,5)
    ):
    """
    Plot integrals
    ==============
    """
    
    ### Check and load the data
    ds = util.load_dataset(dataset)

    z = np.zeros((len(ds), omega_max - omega_min))

    ### Integrals computation
    for i, d in enumerate(ds):
        for j in range(omega_min, omega_max):
            z[i,j-omega_min] = process.integrate_spectrum(
                d, 
                harmonics = [j - 0.5, j + 0.5], 
                component = plot_component
                )
    
    ### Setting ticks
    if normalize:
        for i in range(0,len(z)):
            z[i,:] = z[i,:]/z[i,:].max()

    ### Colormesh plot
    z_max = z.max()

    fig, ax = plt.subplots()
    
    col = Normalize(vmin = z_min, vmax = z_max)    

    c = ax.pcolormesh(range(0, np.shape(z)[0]), range(omega_min, omega_max), np.transpose(z),
        cmap = 'jet',
        shading = 'auto',
        norm = col
    )
    
    fig.colorbar(c, ax=ax, label=r"$\int |D(\omega)| d \omega$ [arb.u.]")

    ### Setting ticks
    yticks = range(omega_min, omega_max)
    xticks = np.linspace(0, len(z)-1, N_x_ticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ### Define y labels
    labels_y = [item.get_text() for item in ax.get_yticklabels()]
    for x, val in enumerate(labels_y):
        ### Only 1, 5, 9 etc. harmonics are shown
        if (int(x) - 1) % 4 == 0:
            labels_y[int(x)] = x + omega_min
        else:
            labels_y[int(x)] = ""

    ### Define x labels
    if scan_variable == 'phase':    
        ### Set initial CEP_2 value
        omega_1 = ds[0].omega_1
        omega_2 = ds[0].omega_2

        rel_omega = int(np.round(omega_2/omega_1))
        CEP_2 = conv.secondary_field(omega_1, rel_omega, 0, 0, 0, 0)[3]

        ### y-axis
        vals_x = -CEP_2 + np.linspace(ds[0].CEP_2, ds[-1].CEP_2, N_x_ticks)
        labels_x = [r"${:.1f}$".format(val) for val in vals_x]

        axis_label = r"$\varphi \, [\pi \, $rad$]$"
    elif scan_variable == 'intensity':
        if plot_component == 'z':
            I_min = conv.E_to_intensity(ds[0].E0_1)
            I_max = conv.E_to_intensity(ds[-1].E0_1)
        else:
            I_min = conv.E_to_intensity(ds[0].E0_2)
            I_min = conv.E_to_intensity(ds[-1].E0_2)
        vals_x = np.linspace(I_min, I_max, N_x_ticks)
        labels_x = [r"${:.2e}$".format(val) for val in vals_x]

        axis_label = r"$I$ [W/cm$^2$]"
    elif scan_variable == 'amplitude':
        A_min = ds[0].E0_2
        A_max = ds[-1].E0_2

        vals_x = np.linspace(A_min, A_max, N_x_ticks)/A_max
        labels_x = [r"${:.2f}$".format(val) for val in vals_x]

        axis_label = r"$E/E_0$"
    else:
        raise ValueError("Unknown scan variable '{}'".format(scan_variable))

    ### Set labels
    ax.set_xticklabels(labels_x, fontsize = 12)
    ax.set_yticklabels(labels_y, fontsize = 12)


    ax.set_xlabel(axis_label, fontsize = 12)

    ax.set_ylabel("H [-]", fontsize=12)


    fig.set_size_inches(figsize)

    plt.show()


def plot_spectral_distribution(
        dataset, 
        plot_component = "z",
        omega_min = 0, 
        omega_max = 50, 
        N_y_ticks = 5,
        z_min = 1e-13,
        plot_energy = False,
        plot_scale = 'log',
        figsize = (14,5),
        rgridFF_w = 0.005,
        distance = 1.,
        gridpoints = 150,
        near_field_factor = True,
        mirror = True,
        plot_beam = False,
        z_max = None
    ):
    '''
    Plot spectral distribution
    ==========================

    plot_spectral_distribution(
        dataset, 
        plot_component = "z",
        omega_min = 0, 
        omega_max = 50, 
        N_y_ticks = 5,
        z_min = 1e-13,
        plot_energy = False,
        plot_scale = 'log',
        figsize = (14,5),
        rgridFF_w = 0.005,
        distance = 1.,
        gridpoints = 150,
        near_field_factor = True,
        mirror = True,
        plot_beam = False
    ):

    Plot spectral distribution of the generated field.

    Parameters:
    -----------
    dataset: Dataset or str
        Loaded dataset or the name of the directory with slash '/'.
    plot_component: {'z', 'x'}, optional, default: 'z'
        The plotted component.
    omega_min: int, optional, default: 0
        Minimum harmonics plotted.
    omega_max: int, optional, default: 50
        Maximum harmonics plotted. 
    N_y_ticks: int, optional, default: 5
        Set number of ticks on the y-axis.
    z_min: float, optional, default: 1e-13
        Minimum plotting value on the colormesh.
    plot_energy: bool, optional, default: False
        Show secondary energy axis in the plot.
    plot_scale: {'log', 'linear'}, optional, default: 'log'
        Plot with logarithmic or linear scale.
    figsize: tuple, optional, default: (9, 4)
        Set figure size in inches - (width, height)
    rgridFF_w: float, optional, default: 0.005
        Radius of the spatio-spectral window for the resolved field.
    distance: float, optional, default: 1.
        Distance of the projection plane from the target.
    gridpoints: int, optional, default: 150
        Number of points for evaluation for the radius grid in the observation
        plane.
    near_field_factor: bool, optional, default: True
        Account the near field factor in the Hankel transform
    mirror: bool, optional, default: True
        Plot the spatially resolved spectrum symetrically along the axis.
    plot_beam: bool, optional, default: False
        Plot the driver beam.
    '''
    ### Loading data
    ds = util.load_dataset(dataset)

    try:
        w0_1 = ds[0].w0_1
        w0_2 = ds[0].w0_2
        z0_1 = ds[0].z0_1
        z0_2 = ds[0].z0_2
        z_trg = ds[0].z_trg
        N_distr = ds[0].N_distr
    except AttributeError:
        print("The selected data does not contain the beams characteristics. "
              "You might want to use plot_spectral_distribution_legacy instead.")
        return

    ### Init variables
    t = ds[0].t
    T_step = t[1]-t[0]
    omega_0 = ds[0].omega_1
    freq_0 = omega_0/ (2*np.pi)
    N_steps_per_freq = int(freq_0*t[-1])
    lambda_1 = mn.ConvertPhoton(omega_0, 'omegaau', 'lambdaSI')
    lambda_2 = mn.ConvertPhoton(ds[0].omega_2, 'omegaau', 'lambdaSI')
    
    ### get spectrum
    if (plot_component == "x"):
        lbl = r"$|E_x(\omega, r)|$"
        FField = np.array([d.d_x for d in ds])
    else:
        lbl = r"$|E_z(\omega, r)|$"
        FField = np.array([d.d_z for d in ds])
    
    ### Compute spectrum
    N = np.shape(FField)[-1]
    FField = scipy.fftpack.fft(FField)[0:N//2]
    ogrid = scipy.fftpack.fftfreq(N, T_step)[0:N//2]
    
    ### x-axis
    omegas = process.set_omegas(ogrid, freq_0)
    omegas = omegas[0:omega_max+1]
    ### Shorten the array 
    rng = slice((N_steps_per_freq)*omega_min,(N_steps_per_freq)*omega_max)
    FField = FField[:,rng]
    ogrid = ogrid[rng]

    FField = np.transpose(FField)

    ### Reconstruction of the beams

    ### find values of E0_1 and E0_2 in focus from the amplitudes in the file
    z_R = lambda w_0, lamb: np.power(w_0, 2) * np.pi / lamb
    z1 = z_trg - z0_1
    z2 = z_trg - z0_2
    E0_1 = ds[0].E0_1*np.sqrt(1 + z1*z1/np.power(z_R(w0_1, lambda_1), 2))
    E0_2 = ds[0].E0_2*np.sqrt(1 + z2*z2/np.power(z_R(w0_2, lambda_2), 2))

    #b1 = util.Beam(lambda_1, ds[0].E0_1, z_0=z0_1, w_0=w0_1, N_pts = N_distr)
    b1 = util.Beam(lambda_1, E0_1, z_0=z0_1, w_0=w0_1, N_pts = N_distr)
    #b1 = util.Beam(lambda_1, ds[0].E0_2, z_0=z0_2, w_0=w0_2, N_pts = N_distr)
    b2 = util.Beam(lambda_2, E0_2, z_0=z0_2, w_0=w0_2, N_pts = N_distr)
    beams = util.Beams()
    beams.add_beam(b1)
    beams.add_beam(b2)
    rgrid, _ = beams.get_beam_profiles(z_target=z_trg)

    if plot_beam:
        beams.plot_beams(z_target=z_trg)

    rgrid_FF = rgridFF_w * np.linspace(0,1,gridpoints) 
    
    ### Compute Hankel transform
    omeg_0 = (ogrid[N_steps_per_freq] - ogrid[0])
    omegaSI = mn.ConvertPhoton(omega_0, 'omegaau', 'omegaSI')
    z = np.abs(hfn.HankelTransform(omegaSI*ogrid/\
                                   omeg_0, rgrid, FField, distance, rgrid_FF, 
                                   near_field_factor = near_field_factor))
    z = np.transpose(z)

    ### Colormesh plot
    if z_max == None:
        z_max = z[:,rng].max()

    fig, ax = plt.subplots()
    
    if plot_scale == 'log':
        col = LogNorm(vmin = z_min, vmax = z_max)
    elif plot_scale == 'linear':
        col = Normalize(vmin = z_min, vmax = z_max)
    else:
        raise ValueError("Unknown scale '" + plot_scale +"'. "
                         "Available options are 'log' and 'linear'")
        
    ### Plot symetrically along the axis
    if mirror:
        range_ = range(0, 2*len(z))
        dum = np.zeros((2*z.shape[0], z.shape[1]))
        dum[len(z)-1:-1, :] = z
        for i in range(len(z)):
            dum[i, :] = z[-(i+1), :]
        z = dum
    else:
        range_ = range(0, len(z))    
    
    c = ax.pcolormesh(ogrid, range_, z,
            cmap = 'jet',
            shading = 'gouraud',
            norm = col
        )

    ax.set_title(lbl)

    cbar = fig.colorbar(c, ax=ax, label=r"$|E(\omega,r)|$ [arb.u.]")
    cbar.ax.tick_params(labelsize=14)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(12)

    ### Setting ticks
    ### Set ticks corresponding to omegas
    set_omega_labels(ax, omegas, ticksize=13, labelsize=14)
    
    ### Define y labels
    axis_label = r"$r$ [m]"
    if mirror:
        vals_y = np.linspace(-rgridFF_w, rgridFF_w, 2*N_y_ticks-1)
    else:
        vals_y = np.linspace(0, rgridFF_w, N_y_ticks)
    
    yticks = np.linspace(0, len(z)-1, len(vals_y))

    ax.set_yticks(yticks)
    labels_y = [r"${:.2e}$".format(val) for val in vals_y]
    
    ### Set labels
    ax.set_yticklabels(labels_y, fontsize = 12)
    ax.set_ylabel(axis_label, fontsize = 14)

    ### Set the limits of the plot
    ax.axis([omegas[omega_min], omegas[omega_max], 0, (len(z))-1])

    ### Set secondary energy axis
    if plot_energy:
        set_energy_axis(ax, ds[0], omega_max)

    fig.set_size_inches(figsize)

    plt.show()


def plot_spectral_distribution_legacy(
        dataset, 
        plot_component = "z",
        omega_min = 0, 
        omega_max = 50, 
        N_y_ticks = 5,
        z_min = 1e-13,
        plot_energy = False,
        plot_scale = 'log',
        figsize = (14,5),
        rgrid_w = 0.0001,
        rgridFF_w = 0.005,
        distance = 1.,
        gridpoints = 150,
        near_field_factor = True,
        mirror = True
    ):
    '''
    Plot spectral distribution - legacy
    ===================================

    plot_spectral_distribution_legacy(
        dataset, 
        plot_component = "z",
        omega_min = 0, 
        omega_max = 50, 
        N_y_ticks = 5,
        z_min = 1e-13,
        plot_energy = False,
        plot_scale = 'log',
        figsize = (14,5),
        rgrid_w = 0.0001,
        rgridFF_w = 0.005,
        distance = 1.,
        gridpoints = 150,
        near_field_factor = True,
        mirror = True
    ):

    Plot spectral distribution of the generated field.

    Parameters:
    -----------
    dataset: Dataset or str
        Loaded dataset or the name of the directory with slash '/'.
    plot_component: {'z', 'x'}, optional, default: 'z'
        The plotted component.
    omega_min: int, optional, default: 0
        Minimum harmonics plotted.
    omega_max: int, optional, default: 50
        Maximum harmonics plotted. 
    N_y_ticks: int, optional, default: 5
        Set number of ticks on the y-axis.
    z_min: float, optional, default: 1e-8
        Minimum plotting value on the colormesh.
    plot_energy: bool, optional, default: False
        Show secondary energy axis in the plot.
    plot_scale: {'log', 'linear'}, optional, default: 'log'
        Plot with logarithmic or linear scale.
    figsize: tuple, optional, default: (9, 4)
        Set figure size in inches - (width, height)
    rgrid_w: float, optional, default: 0.0001
        Beam waist radius in the focus.
    rgridFF_w: float, optional, default: 0.005
        Radius of the spatio-spectral window for the resolved field.
    distance: float, optional, default: 1.
        Distance of the projection plane from the target.
    gridpoints: int, optional, default: 150
        Number of points for evaluation for the radius grid in the observation
        plane.
    near_field_factor: bool, optional, default: True
        Account the near field factor in the Hankel transform
    mirror: bool, optional, default: True
        Plot the spatially resolved spectrum symetrically along the axis.
    '''
    ### Loading data
    ds = util.load_dataset(dataset)

    ### Init variables
    t = ds[0].t
    T_step = t[1]-t[0]
    omega_0 = ds[0].omega_1
    freq_0 = omega_0/ (2*np.pi)
    N_steps_per_freq = int(freq_0*t[-1]) + 1

    ### get spectrum
    if (plot_component == "x"):
        lbl = r"$|E_x(\omega, r)|$"
        FField = np.array([d.d_x for d in ds])
    else:
        lbl = r"$|E_z(\omega, r)|$"
        FField = np.array([d.d_z for d in ds])
    
    ### Compute spectrum
    N = np.shape(FField)[-1]
    FField = scipy.fftpack.fft(FField)[0:N//2]
    ogrid = scipy.fftpack.fftfreq(N, T_step)[0:N//2]
    
    ### x-axis
    omegas = process.set_omegas(ogrid, freq_0)
    omegas = omegas[0:omega_max+1]
    ### Shorten the array 
    rng = slice((N_steps_per_freq-1)*omega_min,(N_steps_per_freq-1)*omega_max)
    FField = FField[:,rng]
    ogrid = ogrid[rng]

    FField = np.transpose(FField)

    ### Get values for rgrid and rgrid_FF
    rgrid, EField = process.distribution(ds[0].E0_1, N_pts=len(ds))
    rgrid = rgrid_w * rgrid
    rgrid_FF = rgridFF_w * np.linspace(0,1,gridpoints)
    
    ### Compute Hankel transform
    omeg_0 = (ogrid[N_steps_per_freq-1] - ogrid[0])
    z = np.abs(hfn.HankelTransform(conv.lambda_to_omega(800, mode="SI")*ogrid/\
                                   omeg_0, rgrid, FField, distance, rgrid_FF, 
                                   near_field_factor = near_field_factor))
    z = np.transpose(z)

    ### Colormesh plot
    z_max = z[:,rng].max()

    fig, ax = plt.subplots()
    
    if plot_scale == 'log':
        col = LogNorm(vmin = z_min, vmax = z_max)
    elif plot_scale == 'linear':
        col = Normalize(vmin = z_min, vmax = z_max)
    else:
        raise ValueError("Unknown scale '" + plot_scale +"'. "
                         "Available options are 'log' and 'linear'")
        
    ### Plot symetrically along the axis
    if mirror:
        range_ = range(0, 2*len(z))
        dum = np.zeros((2*z.shape[0], z.shape[1]))
        dum[len(z)-1:-1, :] = z
        for i in range(len(z)):
            dum[i, :] = z[-(i+1), :]
        z = dum
    else:
        range_ = range(0, len(z))    
    
    c = ax.pcolormesh(ogrid, range_, z,
            cmap = 'jet',
            shading = 'gouraud',
            norm = col
        )

    ax.set_title(lbl, fontsize = 14)

    cbar = fig.colorbar(c, ax=ax, label=r"$|E(\omega,r)|$ [arb.u.]")
    cbar.ax.tick_params(labelsize=14)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(12)

    ### Setting ticks
    ### Set ticks corresponding to omegas
    set_omega_labels(ax, omegas, ticksize=13, labelsize=14)
    
    ### Define y labels
    axis_label = r"$r$ [m]"
    if mirror:
        vals_y = np.linspace(-rgridFF_w, rgridFF_w, 2*N_y_ticks-1)
    else:
        vals_y = np.linspace(0, rgridFF_w, N_y_ticks)
    
    yticks = np.linspace(0, len(z)-1, len(vals_y))

    ax.set_yticks(yticks)
    labels_y = [r"${:.2e}$".format(val) for val in vals_y]
    
    ### Set labels
    ax.set_yticklabels(labels_y, fontsize = 12)
    ax.set_ylabel(axis_label, fontsize = 14)

    ### Set the limits of the plot
    ax.axis([omegas[omega_min], omegas[omega_max], 0, (len(z))-1])

    ### Set secondary energy axis
    if plot_energy:
        set_energy_axis(ax, ds[0], omega_max)

    fig.set_size_inches(figsize)

    plt.show()

def set_omega_labels(ax, omegas, ticksize = 11, labelsize = 12):
    ax.set_xticks(omegas)

    ### Tick labels
    labels = [item.get_text() for item in ax.get_xticklabels()]

    for x in range(0, len(labels)):
        ### Only 1, 5, 9 etc. harmonics are shown
        if (int(x) - 1) % 4 == 0:
            labels[int(x)] = r"${}$".format(x)
        else:
            labels[int(x)] = ""

    xlabel = "H [-]"
    ax.set_xticklabels(labels, fontsize = ticksize)
    ax.set_xlabel(xlabel, fontsize=labelsize)

def set_eV_labels(ax, omega_min, omega_max, freq_0, tick_interval = 50):
    E_eV = conv.eV_per_harmonic(freq_0)

    E_min = omega_min*E_eV
    E_max = omega_max*E_eV

    Energs = np.arange(0, E_max, tick_interval)

    Es = []
    for E in Energs:
        if E < E_min:
            continue
        else:
            Es.append(E)

    Energs = np.array(Es)

    E_ticks = Energs/E_eV*freq_0

    ax.set_xticks(E_ticks)

    labels = Energs

    xlabel = r"$E$ [eV]"
    ax.set_xticklabels(labels, fontsize = 11)
    ax.set_xlabel(xlabel, fontsize=12)

def set_energy_axis(ax, data, omega_max):
    ### Show Ip
    I_p = data.I_p  
    freq_0 = data.omega_1 / (2*np.pi)

    ### Ponderomotive energy (computed from the field with max intensity)
    E0 = np.max([data.E0_2, data.E0_1])
    U_p = conv.compute_Up(E0, freq_0)

    ### Secondary energy axis
    ax2 = ax.twiny()

    ax2.set_xlim(ax.get_xlim())
    
    energies = [(0.5*U_p*x + I_p)/(2*np.pi) for x in range(0, 2*int((freq_0*omega_max*(2*np.pi)-I_p)/U_p))]
    
    ax2.set_xticks(energies)

    energies_ticks = []
    for i in range(0, len(energies)):
    #for i in range(1, 1):
        if i%2 == 0:
            energies_ticks.append(r"${}$".format(int(i*0.5)))
        else:
            energies_ticks.append("")

    ax2.set_xticklabels(energies_ticks, fontsize = 11)
    ax2.set_xlabel(r"$(E-I_p)/U_p$", fontsize = 12)

def norm_to_cycles(ax, T_0, N_cycl_1, spacing = 5):
    T_ticks = [x*T_0 for x in range(0,int(N_cycl_1) + 1)]
    ax.set_xticks(T_ticks)

    T_labels = ax.get_xticklabels()
    for i in range(0,len(T_labels)):
        if i%spacing == 0:
            T_labels[i] = "{}".format(i)
        else:
            T_labels[i] = ""

    ax.set_xticklabels(T_labels)
    ax.set_xlabel(r"o.c. $[T/T_0]$")

def plot_far_field(
        dataset, 
        distance = 1, 
        rgridFF_w = 5e-3, 
        omega_max = -1,
        stokes_max = 20, 
        filter_harmonics = None, 
        stokes_parameters = False,
        coherent_sum = True,
        figsize = (10,6)
    ):
    ### Loading data
    ds = util.load_dataset(dataset)

    ### Init variables
    t = ds[0].t
    omega_0 = ds[0].omega_1
    freq_0 = omega_0/(2*np.pi)
    w0_1 = ds[0].w0_1
    w0_2 = ds[0].w0_2
    z0_1 = ds[0].z0_1
    z0_2 = ds[0].z0_2
    z_trg = ds[0].z_trg
    N_distr = ds[0].N_distr
    lambda_1 = mn.ConvertPhoton(omega_0, 'omegaau', 'lambdaSI')
    lambda_2 = mn.ConvertPhoton(ds[0].omega_2, 'omegaau', 'lambdaSI')
    rgrid_FF = rgridFF_w * np.linspace(0,1,len(ds)) 
    D_z = np.array([d.d_z for d in ds])
    if stokes_parameters:
        D_x = np.array([d.d_x for d in ds])

    ### Reconstruction of the beams
    b1 = util.Beam(lambda_1, ds[0].E0_1, z_0=z0_1, w_0=w0_1, N_pts = N_distr)
    b2 = util.Beam(lambda_2, ds[0].E0_2, z_0=z0_2, w_0=w0_2, N_pts = N_distr)
    beams = util.Beams()
    beams.add_beam(b1)
    beams.add_beam(b2)
    rgrid, _ = beams.get_beam_profiles(z_target=z_trg)

    E_z = process.get_Hankel_transform(D_z, omega_0, rgrid, distance, rgrid_FF, True, t, omega_max)
    E_z = np.real(scipy.ifft(E_z))

    if stokes_parameters:
        E_x = process.get_Hankel_transform(D_x, omega_0, rgrid, distance, rgrid_FF, True, t, omega_max)
        E_x = np.real(scipy.ifft(E_x))

    field_z = np.zeros(E_z.shape[1])
    field_x = np.zeros(E_x.shape[1])

    if coherent_sum:
        for i in range(E_z.shape[0]):
            field_z = field_z + E_z[i,:]
        
        if stokes_parameters:
            for i in range(E_x.shape[0]):
                field_x = field_x + E_x[i,:]
    else:
        field_z = E_z[0,:]
        field_x = E_x[0,:]

    if filter_harmonics != None:
        if stokes_parameters:
            field_z, field_x = process.harmonic_filter([field_z, field_x], 
                                                       t[-1], freq_0,  
                                                       filter_harmonics, 
                                                       'bandpass')
        else: 
            field_z = process.harmonic_filter([field_z], t[-1], freq_0,  
                                          filter_harmonics, 'bandpass')

    max_ = np.array([np.abs(field_z).max(), np.abs(field_x).max()]).max()

    fig, ax = plt.subplots()

    ### Set y-range for fields
    ax.set_ylim(-max_, max_)

    ax.plot(np.linspace(0, t[-1], len(field_z)), field_z, label = r"$E_z$")

    if stokes_parameters != None:
        ax.plot(np.linspace(0, t[-1], len(field_x)), field_x, label = r"$E_x$")

    ax.legend(loc=1)
    ax.set_xlabel(r"$t$ [a.u.]", fontsize=12)
    fig.set_size_inches(figsize)
    plt.show()

    if stokes_parameters:
        ### Degree of coherence
        p = lambda S1, S2, S3: np.sqrt(S1*S1 + S2*S2 + S3*S3)

        if coherent_sum:
            print("Coherent sum:")
            for H in range(9, stokes_max+1):
                ### Spectrally filter harmonic H
                field_z, field_x = process.harmonic_filter([field_z, field_x], 
                                                        t[-1], freq_0,  
                                                        H, 'bandpass')

                ### Stokes parameters for the harmonic H
                S0, S1, S2, S3 = np.round(process.stokes_params(field_z, field_x, t, norm = True), decimals=3)

                print("Stokes parameters for harmonic {} are p = {}, S1 = {}, S2 = {}, S3 = {}".format(H, np.round(p(S1, S2, S3), decimals=2), S1, S2, S3))
                
        print("\n")
        print("Field at r = 0:")
        for H in range(9, stokes_max+1):
            ### Spectrally filter harmonic H
            field_z, field_x = process.harmonic_filter([E_z[0,:], E_x[0,:]], 
                                                       t[-1], freq_0,  
                                                       H, 'bandpass')

            ### Stokes parameters for the harmonic H
            S0, S1, S2, S3 = np.round(process.stokes_params(field_z, field_x, t, norm = True), decimals=3)

            print("Stokes parameters for harmonic {} are p = {}, S1 = {}, S2 = {}, S3 = {}".format(H, np.round(p(S1, S2, S3), decimals=2), S1, S2, S3))


    