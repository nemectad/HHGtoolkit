from scipy import special
from scipy import integrate
from scipy import interpolate
import numpy as np
import struct
import array
import os
import time
import warnings
# import ray
# import matlab.engine
# import string
import multiprocessing as mp
import math
# import joblib
# from mpi4py import MPI
# import oct2py
import shutil
import h5py
import sys
import mynumerics as mn
from mynumerics import units

def HankelTransform(ogrid, rgrid, FField, distance, rgrid_FF, integrator = integrate.trapz, near_field_factor = True):
    """
    It computes Hanikel transform with an optional near-field factor.
    

    Parameters
    ----------
    ogrid : array_like
        grid of FField in frequencies [SI]
    rgrid : array_like
        grid of FField in the radial coordinate [SI]
    FField : 2D array
        The source terms on ogrid and rgrid.        
    distance : scalar
       The distance of the generating plane from the observational screen
    rgrid_FF : array_like
        The grid used to investigate the transformed field
    integrator : function handle, optional
        The function used for the integration. It's called by
        'integrator(integrand,rgrid)'. It shall be extended by a list of
        [args, kwargs].
        The default is integrate.trapz (from scipy).
    near_field_factor : logical, optional
        Include near field factor. The default is True.

    Returns
    -------
    FField_FF : 2D array
         The far-field spectra on ogrid and rgrid_FF

    """
    No = len(ogrid); Nr = len(rgrid); Nr_FF = len(rgrid_FF)
    FField_FF = np.empty((No,Nr_FF), dtype=np.cdouble)
    integrand = np.empty((Nr), dtype=np.cdouble)
    for k1 in range(No):
        k_omega = ogrid[k1] / units.c_light # ogrid[k3] / units.c_light; ogrid[k1] * units.alpha_fine  # ??? units
        for k2 in range(Nr_FF):
            for k3 in range(Nr):
                if near_field_factor:
                    integrand[k3] = np.exp(-1j * k_omega * (rgrid[k3] ** 2) / (2.0 * distance)) * rgrid[k3] *\
                                    FField[k1,k3] * special.jn(0, k_omega * rgrid[k3] * rgrid_FF[k2] / distance)
                else:
                    integrand[k3] = rgrid[k3] * FField[k1,k3] * special.jn(0, k_omega * rgrid[k3] * rgrid_FF[k2] / distance)
            FField_FF[k1,k2] = integrator(integrand,rgrid)

    return FField_FF


def HankelTransform_long(ogrid, rgrid, zgrid, FSourceTerm, # FSourceTerm(r,z,omega)
                         distance, rgrid_FF,
                         dispersion_function = None, absorption_function = None,
                         integrator_Hankel = integrate.trapz, integrator_longitudinal = 'trapezoidal',
                         near_field_factor = True,
                         store_cummulative_result = False,
                         frequencies_to_trace_maxima = None
                         ):
    """
    It computes XUV propagation using a sum of Hankel transforms along the medium.

    Parameters
    ----------
    ogrid : array_like
        grid of FSourceTerm in frequencies [SI]
    rgrid : array_like
        grid of FSourceTerm in the radial coordinate [SI]
    zgrid : array_like
        grid of FSourceTerm in the longitudinal coordinate [SI]
    FSourceTerm : The source on the grid in the medium
        order: (r,z,omega); there is no inter check of dimensions, it has to match
    distance : scalar
        The distance from the first point of the medium [SI]
    rgrid_FF : array_like
        the required radial grid in far-field [SI]
    dispersion_function : function, optional
        The dispersion is factored as exp(i*z*omega*dispersion_function(omega)). The default is None.
    absorption_function : function, optional
        Analogical to dispersion. The default is None.
    integrator_Hankel : function, optional
        used method to integrate in the radial direction. The default is integrate.trapz.
    integrator_longitudinal : function, optional
        used method to integrate in the longitudinal direction. The default is 'trapezoidal'.
    near_field_factor : logical, optional
        False for far field without the Fresnel term. The default is True.    
    frequencies_to_trace_maxima : list of 2D-array_like, optional
        If present, these windows given by the 2-D-array-likes are used to trace maxima of respective planes of integration.
        The default is None.

    Raises
    ------
    NotImplementedError
        In the case a non-implemented integration rule is inputed.

    Returns
    -------
    result : the field in the radial grid of the observation plane
    
    result , planes_maxima: if frequencies_to_trace_maxima are present
        .

    """

    
    No = len(ogrid); Nz = len(zgrid); Nr_FF = len(rgrid_FF)
    include_dispersion = not(dispersion_function is None)
    include_absorption = not(absorption_function is None)
    trace_maxima_log = not(frequencies_to_trace_maxima is None)
    
    if include_dispersion:
        dispersion_factor = np.empty(No)
        for k1 in range(No):
            dispersion_factor[k1] = ogrid[k1]*dispersion_function(ogrid[k1])        
        
    if include_absorption:
        absorption_factor = np.empty(No)
        for k1 in range(No):
            absorption_factor[k1] = ogrid[k1]*absorption_function(ogrid[k1])
      
    # compute z-evolution of the factors        
    if (include_dispersion and include_absorption):
        factor_e = np.exp(
                          1j*np.outer(zgrid,dispersion_factor) +
                          np.outer(zgrid-zgrid[-1] ,absorption_factor)
                          )
    elif include_dispersion:
        factor_e = np.exp(1j*np.outer(zgrid,dispersion_factor))

    elif include_absorption:
        factor_e = np.exp(np.outer(zgrid-zgrid[-1] ,absorption_factor))     

            
    # we keep the data for now, consider on-the-fly change
    print('Computing Hankel from planes')
    t_start = time.perf_counter()
    for k1 in range(Nz):
        print('plane', k1, 'time:', time.perf_counter()-t_start)
        FSourceTerm_select = np.squeeze(FSourceTerm[:,k1,:]).T
        FField_FF = HankelTransform(ogrid,
                                  rgrid,
                                  FSourceTerm_select,
                                  distance-zgrid[k1],
                                  rgrid_FF,
                                  integrator = integrator_Hankel,
                                  near_field_factor = near_field_factor)
        
        if (k1 == 0): # allocate space
            FField_FF_z = np.zeros( (Nz,) + FField_FF.shape,dtype=np.cdouble) 
        
        if (include_dispersion or include_absorption):  
             FField_FF_z[k1,:,:] = np.outer(factor_e[k1,:],np.ones(FField_FF.shape[1]))*FField_FF
        else:
            FField_FF_z[k1,:,:] = FField_FF # (z,omega,r)
    
    if store_cummulative_result:
        cummulative_field = np.empty((Nz-1,) + FField_FF.shape, dtype=np.cdouble)
        
    if (integrator_longitudinal == 'trapezoidal'):        
        for k1 in range(Nz-1):    
            k_step = 1
            if (k1 == 0):
                dum = 0.5*(zgrid[(k1+1)*k_step]-zgrid[k1*k_step]) * \
                      (FField_FF_z[k1*k_step,:,:] + FField_FF_z[(k1+1)*k_step,:,:])
            else:
                dum = dum + \
                      0.5*(zgrid[(k1+1)*k_step]-zgrid[k1*k_step]) * \
                      (FField_FF_z[k1*k_step,:,:] + FField_FF_z[(k1+1)*k_step,:,:])
                      
            if store_cummulative_result:
                if include_absorption:
                    # we need renormalise the end of the medium
                    exp_renorm = np.exp( (zgrid[-1]-zgrid[k1]) * absorption_factor)
                    for k2 in range(No):
                        for k3 in range(Nr_FF):
                            cummulative_field[k1,k2,k3] = exp_renorm[k2]*dum[k2,k3]
                else:
                    cummulative_field[k1,:,:] = dum
                
    else:
        raise NotImplementedError('Only trapezoidal rule implemented now')
        
    if trace_maxima_log:
        
        frequency_indices = []
        planes_maxima = []
        for frequency_list in frequencies_to_trace_maxima:
            try:
                frequency_indices.append(mn.FindInterval(ogrid, frequency_list))
                planes_maxima.append([])
            except:
                warnings.warn("A frequency from frequencies_to_trace_maxima doesn't match ogrid.")
        
        if (len(frequency_indices)>0):
            for k1 in range(Nz):
                for k2 in range(len(frequency_indices)):
                    planes_maxima[k2].append(np.max(abs(
                                      FField_FF_z[k1,frequency_indices[k2][0]:frequency_indices[k2][1],:]
                                            )))
                    
            for k1 in range(len(frequency_indices)):
                planes_maxima[k1] = np.asarray(planes_maxima[k1])
        
        if store_cummulative_result:
            return dum , planes_maxima, cummulative_field
        else:
            return dum , planes_maxima
            
    else: 
        if store_cummulative_result:
            return dum, cummulative_field
        else:
            return dum
                
                
    
    

# def FieldOnScreen(k_start, k_num, NP, LP):
# # this function computes the Hankel transform of a given source term in omega-domain stored in FField_r
# # all the grids are specified in the inputs, except the analysis in omega_anal, this is specified by 'k_start' and 'k_num', it is used in the multiprocessing scheme
# # I tried tests while implementing OOP, sme runs were longer with the .-notation
#     print('Hankel transform running')
#     Nz_anal = np.asarray(NP.zgrid_anal.shape); Nz_anal = Nz_anal[1];
#     Nr_anal = len(NP.rgrid_anal); Nr = len(NP.rgrid); Nz_medium=len(NP.z_medium);
#     FHHGOnScreen = np.empty([Nz_medium, Nz_anal, k_num, Nr_anal], dtype=np.cdouble)
#     SourceTerm = np.empty([Nr], dtype=np.cdouble)
#     integrand = np.empty([Nr], dtype=np.cdouble)
#     for k1 in range(Nz_medium): # loop over different medium positions
#         print('process starting at omega', k_start, ' started computation of zgrid', NP.z_medium[k1])
#         for k2 in range(k_num):  # omega
#             k3 = k_start + k2 * NP.omega_step  # accesing the grid ## omegagrid, Igrid, FSourceterm, LaserParams):
#             if (NP.storing_source_terms == 'on-the-fly'):
#                 for k4 in range(Nr): SourceTerm[k4] = ComputeOneFieldFromIntensityList(NP.z_medium[k1], NP.rgrid[k4], k3, NP.omegagrid, NP.Igrid, NP.FSourceterm, LP) # precompute field in r
#             k_omega = NP.omegagrid[k3] / (units.TIMEau * units.c_light);  # omega divided by time: a.u. -> SI
#             for k4 in range(Nz_anal):
#                 for k5 in range(Nr_anal):
#                     if (NP.storing_source_terms == 'table'):
#                         for k6 in range(Nr): integrand[k6] = np.exp(-1j * k_omega * (NP.rgrid[k6] ** 2) / (2.0 * (NP.zgrid_anal[k1, k4] - NP.z_medium[k1]))) * NP.rgrid[k6] * NP.FField_r[k1, k3, k6] * special.jn(0, k_omega * NP.rgrid[k6] * NP.rgrid_anal[k5] / (NP.zgrid_anal[k1, k4] - NP.z_medium[k1]));
#                     elif (NP.storing_source_terms == 'on-the-fly'):
#                         for k6 in range(Nr): integrand[k6] = np.exp(-1j * k_omega * (NP.rgrid[k6] ** 2) / (2.0 * (NP.zgrid_anal[k1, k4] - NP.z_medium[k1]))) * NP.rgrid[k6] * SourceTerm[k6] * special.jn(0, k_omega * NP.rgrid[k6] * NP.rgrid_anal[k5] / (NP.zgrid_anal[k1, k4] - NP.z_medium[k1]));
#                     else: sys.exit('Wrong field storing method')

#                     if (NP.integrator['method'] == 'Romberg'):
#                         nint, value, err = mn.romberg(NP.rgrid[-1]-NP.rgrid[0],integrand,NP.integrator['tol'],NP.integrator['n0'])
#                         FHHGOnScreen[k1, k4, k2, k5] = nint; # (1.0/(NP.zgrid_anal[k1, k4] - NP.z_medium[k1])) *
#                     elif (NP.integrator['method'] == 'Trapezoidal'): FHHGOnScreen[k1, k4, k2, k5] = (1.0/(NP.zgrid_anal[k1, k4] - NP.z_medium[k1])) * integrate.trapz(integrand, NP.rgrid);
#                     elif (NP.integrator['method'] == 'Simpson'): FHHGOnScreen[k1, k4, k2, k5] = (1.0/(NP.zgrid_anal[k1, k4] - NP.z_medium[k1])) * integrate.simps(integrand, NP.rgrid);
#                     elif (NP.integrator['method'] == 'copysource'): FHHGOnScreen[k1, k4, k2, k5] = SourceTerm[0];
#                     else: sys.exit('Wrong integrator')

#     return (k_start, k_num, FHHGOnScreen)



# # Optimal workload is obtained by the same amount of load for each woker if possible, eventually one extra task for the last worker. Otherwise (NOT OPTIMAL!!!), every worker takes more load and some workers may be eventually not used. An optimal routine would be either balance the load among the workers or employ some sophisticated  parallel scheduler.
# def ObtainWorkload(Nomega_points,W):
#   if ( ( (Nomega_points % W)==0 ) or ( (Nomega_points % W)==1 ) ):
#     Nint = Nomega_points//W; # the number of points in an interval (beside the last interval...); mn.NumOfPointsInRange(0,Nomega_points,W);
#     N_PointsGrid=[]; N_PointsForProcess=[];
#     for k1 in range(W): N_PointsGrid.append(k1*Nint);
#     N_PointsGrid.append(Nomega_points);
#     for k1 in range(W): N_PointsForProcess.append(N_PointsGrid[k1+1]-N_PointsGrid[k1])
#   else:
#     Nint = (Nomega_points//W) + 1;
#     N_PointsGrid=[]; N_PointsForProcess=[];
#     for k1 in range(W+1):
#       dum = k1*Nint
#       if dum >= Nomega_points:
#         print('dum', dum)
#         N_PointsGrid.append(Nomega_points);
#         W = k1;
#         break;
#       else:
#         N_PointsGrid.append(dum);
#     print(N_PointsGrid)
#     for k1 in range(W): N_PointsForProcess.append(N_PointsGrid[k1+1]-N_PointsGrid[k1])

#   return W, N_PointsGrid, N_PointsForProcess
# # optimal workload is now given by the number of processes


# def CoalesceResults(results,Nz_medium,Nz_anal,Nomega_anal_start,Nomega_points,Nr_anal,W):
#     FHHGOnScreen = np.empty([Nz_medium,Nz_anal, Nomega_points, Nr_anal, 2], dtype=np.double)
#     for k1 in range(W):  # loop over unsorted results
#         for k5 in range(Nz_medium):
#             for k2 in range(results[k1][1]):  # # of omegas computed by this worker
#                 for k3 in range(Nr_anal):  # results in the radial grid
#                     for k4 in range(Nz_anal):  # results in the z grid
#                         FHHGOnScreen[k5, k4, results[k1][0] - Nomega_anal_start + k2, k3, 0] = results[k1][2][k5][k4][k2][k3].real  # we adjust the field index properly to the field it just sorts matrices the following way [A[1], A[2], ...], the indices are retrieved by the append mapping
#                         FHHGOnScreen[k5, k4, results[k1][0] - Nomega_anal_start + k2, k3, 1] = results[k1][2][k5][k4][k2][k3].imag
#         #      FHHGOnScreen[ results[k1][0]-Nomega_anal_start+k2 , k3 ] = results[k1][2][k2][k3]/(r_Bohr**2) # eventually fully in atomic units for the integral, but there is still a prefactor of the integral!!!
#     return FHHGOnScreen




# # # define function to integrate, there are some global variables! ## THE OUTPUT IS IN THE MIX OF ATOMIC UNITS (field) and SI UNITS (radial coordinate + dr in the integral)
# # def FieldOnScreen_singleplane(z_medium, omegagrid, omega_step, rgrid, FField_r, rgrid_anal, zgrid_anal, k_start, k_num, integrator):
# # # this function computes the Hankel transform of a given source term in omega-domain stored in FField_r
# # # all the grids are specified in the inputs, except the analysis in omega_anal, this is specified by 'k_start' and 'k_num', it is used in the multiprocessing scheme
# #     Nz_anal = len(zgrid_anal); Nr_anal = len(rgrid_anal); Nr = len(rgrid);
# #     FHHGOnScreen = np.empty([Nz_anal, k_num, Nr_anal], dtype=np.cdouble)
# #     k4 = 0  # # of loops in omega
# #     for k1 in range(k_num):  # Nomega
# #         k5 = k_start + k1 * omega_step  # accesing the grid
# #         tic = time.process_time()
# #         for k6 in range(Nz_anal):
# #             for k2 in range(Nr_anal):  # Nr
# #                 k_omega = omegagrid[k5] / (units.TIMEau * units.c_light);  # omega divided by time: a.u. -> SI # use this as a prefactor to obtain curvature
# #                 integrand = np.empty([Nr], dtype=np.cdouble)
# #                 for k3 in range(Nr): integrand[k3] = np.exp(-(rgrid[k3] ** 2) / (2.0 * (zgrid_anal[k6] - z_medium))) * rgrid[k3] * FField_r[k5, k3] * special.jn(0, k_omega * rgrid[k3] * rgrid_anal[k2] / (zgrid_anal[k6] - z_medium));  # rescale r to atomic units for spectrum in atomic units! (only scaling)
# #                 if (integrator == 'Trapezoidal'): FHHGOnScreen[k6, k4, k2] = (1.0 / (zgrid_anal[k6] - z_medium)) * integrate.trapz(integrand, rgrid);
# #                 elif (integrator == 'Simpson'): FHHGOnScreen[k6, k4, k2] = (1.0 / (zgrid_anal[k6] - z_medium)) * integrate.simps(integrand, rgrid);
# #                 else: sys.exit('Wrong integrator')
# #             # k2 loop end
# #         # k6 loop end
# #         toc = time.process_time()
# #         #    print('cycle',k1,'duration',toc-tic)
# #         k4 = k4 + 1
# #     return (k_start, k_num, FHHGOnScreen)



# def CoalesceResults_serial(results,Nz_anal,Nomega_anal_start,Nomega_points,Nr_anal,W):
#     FHHGOnScreen = np.empty([Nz_anal, Nomega_points, Nr_anal, 2], dtype=np.double)
#     for k1 in range(W):  # loop over unsorted results
#         for k2 in range(results[k1][1]):  # # of omegas computed by this worker
#             for k3 in range(Nr_anal):  # results in the radial grid
#                 for k4 in range(Nz_anal):  # results in the z grid
#                     FHHGOnScreen[k4, results[k1][0] - Nomega_anal_start + k2, k3, 0] = results[k1][2][k4][k2][k3].real  # we adjust the field index properly to the field it just sorts matrices the following way [A[1], A[2], ...], the indices are retrieved by the append mapping
#                     FHHGOnScreen[k4, results[k1][0] - Nomega_anal_start + k2, k3, 1] = results[k1][2][k4][k2][k3].imag
#     #      FHHGOnScreen[ results[k1][0]-Nomega_anal_start+k2 , k3 ] = results[k1][2][k2][k3]/(r_Bohr**2) # eventually fully in atomic units for the integral, but there is still a prefactor of the integral!!!
#     return FHHGOnScreen






# ###########################################################
# #  there is the part for phenomenological dipoles:        #
# ###########################################################
# #

# # this should be loaded from somewhere or computed, or whatever... NOT directly in the code!
# omegawidth = 4.0/np.sqrt(4000.0**2); # roughly corresponds to 100 fs
# PhenomParams = np.array([
# [1, 2, 35, 39], # harmonics
# [0, 0, 1775., 3600.], # alphas
# # [1, 29, 35, 39], # harmonics
# # [0, 500., 1775., 3600.], # alphas
# [omegawidth, omegawidth, omegawidth, omegawidth]
# ])
# NumHarm = 2; # number of harmonics

# ## define dipole function
# def dipoleTimeDomainApp(tgrid,z,r,I0,PhenomParams,tcoeff,rcoeff,omega0,zR):
# # !!!! THERE ARE SEVERAL POINTS NEEDED TO MENTION HERE
# #  - We don't follow the notation of the complex field exp(-i*(omega*t-...)). The reason is that we can then use fft instead of ifft... Be careful with conversions.
# #  - One should then check the sign of alpha as well...

# #  tcoeff = 4.0*np.log(2.0)*TIMEau**2 / ( TFWHMSI**2 )
# #  rcoeff = 2.0/(w0r**2)
#   # we are in atomic units, we go to electric field instead of intensity
#   E0 = np.sqrt(I0) / np.sqrt(1.0+(z/zR)**2) # reduced in z
#   lambd = mn.ConvertPhoton(omega0,'omegaau','lambdaSI')
#   res = []
#   for k1 in range(len(tgrid)):
#     res1 = 0.0*1j;
#     # intens = E0*np.exp(-tcoeff*(tgrid[k1])**2 - rcoeff*r**2) # qeff here?
#     for k2 in range(NumHarm):
#         # if (k2 == 2): intens = E0 * np.exp(-tcoeff * (tgrid[k1]-900.0) ** 2 - rcoeff * r ** 2)  # qeff here?
#         # else: intens = E0 * np.exp(-tcoeff * (tgrid[k1]) ** 2 - rcoeff * r ** 2)  # qeff here?
#         intens = E0 * np.exp(-tcoeff * (tgrid[k1]) ** 2 - rcoeff * r ** 2)  # qeff here?
#         alpha = PhenomParams[1, k2]
#         order = PhenomParams[0, k2]
#         k_omega_wave = 2.0*np.pi*order/lambd
#         IR_induced_phase = mn.GaussianBeamCurvaturePhase(r,z,k_omega_wave,zR) # order included
#         res1 = res1 + intens*np.exp(1j*(tgrid[k1]*omega0*order-IR_induced_phase-alpha*intens))
#     res.append(res1); ## various points in time
#   return np.asarray(res)



# def ComputeFieldsPhenomenologicalDipoles(I0SI,omega0,TFWHM,w0,tgrid,omegagrid,rgrid,z_medium):
#   print('Computing phenomenological dipoles: FFTs')
#   Nomega = len(omegagrid); Nr = len(rgrid); Nz_medium = len(z_medium);
#   FField_r=np.empty([Nz_medium,Nomega,Nr], dtype=np.cdouble)
# #   FField_r=np.empty([Nomega,Nr], dtype=np.cdouble)

#   # coeffs applied in intensity
#   # tcoeff = 4.0*np.log(2.0)*units.TIMEau**2 / ( TFWHM**2 )
#   # rcoeff = 2.0/(w0**2)

#   # but we apply them in the field
#   zR = mn.GaussianBeamRayleighRange(w0, mn.ConvertPhoton(omega0, 'omegaau', 'lambdaSI'))
#   tcoeff = 2.0*np.log(2.0)*units.TIMEau**2 / ( TFWHM**2 )



#   k1 = mn.FindInterval(tgrid,0.0)
#   dt = tgrid[k1+1]-tgrid[k1]

#   for k1 in range(Nz_medium):
#     w = w0 * np.sqrt(1.0 + (z_medium[k1] / zR) ** 2)  # reduced in z
#     rcoeff = 1.0 / (w ** 2)
#     for k2 in range(Nr): # multiprocessing?
#       print('kr',k2)
#       # the expression of the dipole is tricky, see appendix C in Jan Vabek's diploma thesis (using "two-times" Fourier in omega)
#       dum = dipoleTimeDomainApp(tgrid,z_medium[k1],rgrid[k2],I0SI/units.INTENSITYau,PhenomParams,tcoeff,rcoeff,omega0,zR) # compute "complex" dipole in t-domain
#       dum = (dt/np.sqrt(2.0*np.pi))*np.fft.fft(dum) # fft with normalisation
#       # FField_r[k1,:,k2] = dum # !! CANNOT BE DONE THIS WAY, WE HAVE EXTRA ASSUMPTION THAT OUR SIGNAL IS REAL fft: C -> C
#       for k3 in range(Nomega): FField_r[k1,k3,k2] = dum[k3]
#   print('FFT computed');
#   return  FField_r






# # # define function to integrate, there are some global variables! ## THE OUTPUT IS IN THE MIX OF ATOMIC UNITS (field) and SI UNITS (radial coordinate + dr in the integral)
# # def FieldOnScreenLambda1(k_start, k_num, NP, LP):
# #     Nz_anal = np.asarray(NP.zgrid_anal.shape); Nz_anal = Nz_anal[1]; Nr_anal = len(NP.rgrid_anal); Nr = len(NP.rgrid); Nz_medium=len(NP.z_medium);
# #     FHHGOnScreen = np.empty([Nz_medium, Nz_anal, k_num, Nr_anal], dtype=np.cdouble); SourceTerm = np.empty([Nr], dtype=np.cdouble)
# #
# #     for k1 in range(Nz_medium): # loop over different medium positions
# #         print('process starting at omega', k_start, ' started computation of zgrid', NP.z_medium[k1])
# #         for k2 in range(k_num):  # omega
# #             k3 = k_start + k2 * NP.omega_step  # accesing the grid ## omegagrid, Igrid, FSourceterm, LaserParams):
# #             if (NP.storing_source_terms == 'on-the-fly'):
# #                 for k4 in range(Nr): SourceTerm[k4] = ComputeOneFieldFromIntensityList(NP.z_medium[k1], NP.rgrid[k4], k3, NP.omegagrid, NP.Igrid, NP.FSourceterm, LP) # precompute field in r
# #             SourceTerm = lambda r: ComputeOneFieldFromIntensityList2(NP.z_medium[k1], r, k3, NP.omegagrid, NP.Igrid, NP.FSourceterm, LP) # precompute field in r
# #             k_omega = NP.omegagrid[k3] / (units.TIMEau * units.c_light);  # omega divided by time: a.u. -> SI
# #             for k4 in range(Nz_anal):
# #                 for k5 in range(Nr_anal):
# #                     integrand = lambda r: np.real( np.exp(-1j * k_omega * (r ** 2) / (2.0 * (NP.zgrid_anal[k1, k4] - NP.z_medium[k1]))) * r * SourceTerm(r) * special.jn(0, k_omega * r * NP.rgrid_anal[k5] / (NP.zgrid_anal[k1, k4] - NP.z_medium[k1])) )
# #                     # integrand_imag = lambda r: np.imag(np.exp(-1j * k_omega * (r ** 2) / (2.0 * (NP.zgrid_anal[k1, k4] - NP.z_medium[k1]))) * r * SourceTerm(r) * special.jn(0, k_omega * r * NP.rgrid_anal[k5] / (NP.zgrid_anal[k1, k4] - NP.z_medium[k1])))
# #                     FHHGOnScreen[k1, k4, k2, k5] = (1.0/(NP.zgrid_anal[k1, k4] - NP.z_medium[k1])) * integrate.fixed_quad(integrand, 0, NP.rmax,n=1000)
# #
# #     return (k_start, k_num, FHHGOnScreen)




# # # define function to integrate, there are some global variables! ## THE OUTPUT IS IN THE MIX OF ATOMIC UNITS (field) and SI UNITS (radial coordinate + dr in the integral)
# # def FieldOnScreenApertured1(k_start, k_num, NP, LP):
# #     Nz_anal = np.asarray(NP.zgrid_anal.shape); Nz_anal = Nz_anal[1]; Nr_anal = len(NP.rgrid_anal); Nr = len(NP.rgrid); Nz_medium=len(NP.z_medium);
# #     FHHGOnScreen = np.empty([Nz_medium, Nz_anal, k_num, Nr_anal], dtype=np.cdouble); SourceTerm = np.empty([Nr], dtype=np.cdouble)
# #
# #     def Green(r, r1, k_omega, D1, D2):
# #         return r * D1 * special.jn(1, k_omega * r * LP.r_pinhole / D2 ) * special.jn(0, k_omega * r1 * LP.r_pinhole / D1 ) - r1 * D2 * special.jn(0, k_omega * r * LP.r_pinhole / D2 ) * special.jn(1, k_omega * r1 * LP.r_pinhole / D1 )
# #
# #     for k1 in range(Nz_medium): # loop over different medium positions
# #         print('process starting at omega', k_start, ' started computation of zgrid', NP.z_medium[k1])
# #         for k2 in range(k_num):  # omega
# #             k3 = k_start + k2 * NP.omega_step  # accesing the grid ## omegagrid, Igrid, FSourceterm, LaserParams):
# #             SourceTerm = lambda r: ComputeOneFieldFromIntensityList(NP.z_medium[k1], r, k3, NP.omegagrid, NP.Igrid, NP.FSourceterm, LP) # precompute field in r
# #             k_omega = NP.omegagrid[k3] / (units.TIMEau * units.c_light)  # omega divided by time: a.u. -> SI
# #             for k4 in range(Nz_anal):
# #                 for k5 in range(Nr_anal):
# #                     D1 = (LP.z_pinhole - NP.z_medium[k1]); D2 = (NP.zgrid_anal[k1, k4] - LP.z_pinhole);
# #                     integrand = lambda r: np.exp(-1j * k_omega * (r ** 2) / (2.0 * D1)) * Green(NP.rgrid_anal[k5],r,k_omega,D1,D2) * SourceTerm(r) * r / ((D1*NP.rgrid_anal[k5])**2 - (D2*r)**2)
# #                     FHHGOnScreen[k1, k4, k2, k5] = (LP.r_pinhole/k_omega) * integrate.quad(integrand, 0, NP.rmax)
# #
# #     return (k_start, k_num, FHHGOnScreen)


# # define function to integrate, there are some global variables! ## THE OUTPUT IS IN THE MIX OF ATOMIC UNITS (field) and SI UNITS (radial coordinate + dr in the integral)
# def FieldOnScreenApertured1(k_start, k_num, NP, LP):
# # The problem is the integration, the divergence is unfortunatelly mixture of r,r1,D1,D2, so it's not easy to avoid, I tried to use adaprive quadrature rules and others, one of problems is that some work miss to full vectorisation,...

#     print('Composed optical system, analytic; running')

#     def Green_pref(r, r1, k_omega, D1, D2): # Green function with the prefactor
#         if ( (abs((D1*r)**2 - (D2*r1)**2) < 4*np.finfo(np.double).eps) or (k_omega < 4*np.finfo(np.double).eps)  ): #2*np.finfo(np.double).eps ): #((D1*r)**2 - (D2*r1)**2) == 0.0 : # eventually use some not sharp comparision
#             return ((0.5*LP.r_pinhole**2)/(D1*D2)) * \
#                    ( (special.jn(0, k_omega * r1 * LP.r_pinhole / D1 ))**2 + (special.jn(1, k_omega * r1 * LP.r_pinhole / D1 ))**2 )
#         else:
#             return (LP.r_pinhole/k_omega) * \
#                    (\
#                            (r * D1 * special.jn(1, k_omega * r * LP.r_pinhole / D2 ) * special.jn(0, k_omega * r1 * LP.r_pinhole / D1 ) - \
#                             r1 * D2 * special.jn(0, k_omega * r * LP.r_pinhole / D2 ) * special.jn(1, k_omega * r1 * LP.r_pinhole / D1 ) \
#                     )/ ((D1*r)**2 - (D2*r1)**2) )

#     Nz_anal = np.asarray(NP.zgrid_anal.shape); Nz_anal = Nz_anal[1];
#     Nr_anal = len(NP.rgrid_anal); Nr = len(NP.rgrid); Nz_medium=len(NP.z_medium);
#     FHHGOnScreen = np.empty([Nz_medium, Nz_anal, k_num, Nr_anal], dtype=np.cdouble)
#     SourceTerm = np.empty([Nr], dtype=np.cdouble)
#     integrand = np.empty([Nr], dtype=np.cdouble)
#     for k1 in range(Nz_medium): # loop over different medium positions
#         print('process starting at omega', k_start, ' started computation of zgrid', NP.z_medium[k1])
#         for k2 in range(k_num):  # omega
#             k3 = k_start + k2 * NP.omega_step  # accesing the grid ## omegagrid, Igrid, FSourceterm, LaserParams):
#             if (NP.storing_source_terms == 'on-the-fly'):
#                 for k4 in range(Nr): SourceTerm[k4] = ComputeOneFieldFromIntensityList(NP.z_medium[k1], NP.rgrid[k4], k3, NP.omegagrid, NP.Igrid, NP.FSourceterm, LP) # precompute field in r
#             k_omega = NP.omegagrid[k3] / (units.TIMEau * units.c_light);  # omega divided by time: a.u. -> SI
#             for k4 in range(Nz_anal):
#                 D1 = (LP.z_pinhole - NP.z_medium[k1]); D2 = (NP.zgrid_anal[k1, k4] - LP.z_pinhole);
#                 for k5 in range(Nr_anal):
#                     if (NP.storing_source_terms == 'on-the-fly'):
#                         # np.exp(-1j * k_omega * (NP.rgrid[k6] ** 2) / (2.0 * (NP.zgrid_anal[k1, k4] - NP.z_medium[k1]))) * NP.rgrid[k6] * SourceTerm[k6] * special.jn(0, k_omega * NP.rgrid[k6] * NP.rgrid_anal[k5] / (NP.zgrid_anal[k1, k4] - NP.z_medium[k1]));
#                         for k6 in range(Nr): integrand[k6] = np.exp(-1j * k_omega * (NP.rgrid[k6] ** 2) / (2.0 * D1)) * Green_pref(NP.rgrid_anal[k5],NP.rgrid[k6],k_omega,D1,D2) * SourceTerm[k6] * NP.rgrid[k6] #/ ((D1*NP.rgrid_anal[k5])**2 - (D2*NP.rgrid[k6])**2)
#                     else: sys.exit('Wrong field storing method')

#                     # if (NP.integrator['method'] == 'Romberg'):
#                     #     nint, value, err = mn.romberg(NP.rgrid[-1]-NP.rgrid[0],integrand,NP.integrator['tol'],NP.integrator['n0'])
#                     #     FHHGOnScreen[k1, k4, k2, k5] = nint; # (1.0/(NP.zgrid_anal[k1, k4] - NP.z_medium[k1])) *
#                     if (NP.integrator['method'] == 'Trapezoidal'): FHHGOnScreen[k1, k4, k2, k5] = integrate.trapz(integrand, NP.rgrid);
#                     # elif (NP.integrator['method'] == 'Simpson'): FHHGOnScreen[k1, k4, k2, k5] = (1.0/(NP.zgrid_anal[k1, k4] - NP.z_medium[k1])) * integrate.simps(integrand, NP.rgrid);
#                     else: sys.exit('Wrong integrator')

#     return (k_start, k_num, FHHGOnScreen)







# # define function to integrate, there are some global variables! ## THE OUTPUT IS IN THE MIX OF ATOMIC UNITS (field) and SI UNITS (radial coordinate + dr in the integral)
# def FieldOnScreenApertured2D1(k_start, k_num, NP, LP):
# # Uses 2D integration

#     print('Composed optical system, 2D integration; running')

#     Nz_anal = np.asarray(NP.zgrid_anal.shape); Nz_anal = Nz_anal[1];
#     Nr_anal = len(NP.rgrid_anal); Nr = len(NP.rgrid); Nz_medium=len(NP.z_medium); Nr2 = len(NP.rgrid2)
#     FHHGOnScreen = np.empty([Nz_medium, Nz_anal, k_num, Nr_anal], dtype=np.cdouble)
#     SourceTerm = np.empty([Nr], dtype=np.cdouble)
#     integrand = np.empty([Nr,Nr2], dtype=np.cdouble)
#     integral_r1 = np.empty([Nr], dtype=np.cdouble)
#     for k1 in range(Nz_medium): # loop over different medium positions
#         print('process starting at omega', k_start, ' started computation of zgrid', NP.z_medium[k1])
#         for k2 in range(k_num):  # omega
#             k3 = k_start + k2 * NP.omega_step  # accesing the grid ## omegagrid, Igrid, FSourceterm, LaserParams):
#             if (NP.storing_source_terms == 'on-the-fly'):
#                 for k4 in range(Nr): SourceTerm[k4] = ComputeOneFieldFromIntensityList(NP.z_medium[k1], NP.rgrid[k4], k3, NP.omegagrid, NP.Igrid, NP.FSourceterm, LP) # precompute field in r
#             k_omega = NP.omegagrid[k3] / (units.TIMEau * units.c_light);  # omega divided by time: a.u. -> SI
#             for k4 in range(Nz_anal):
#                 D1 = (LP.z_pinhole - NP.z_medium[k1]); D2 = (NP.zgrid_anal[k1, k4] - LP.z_pinhole);
#                 for k5 in range(Nr_anal):
#                     if (NP.storing_source_terms == 'on-the-fly'):
#                         for k6 in range(Nr):
#                             for k7 in range(Nr2):
#                                 integrand[k6,k7] = np.exp(-1j * k_omega * ( (NP.rgrid[k6] ** 2) / (2.0 * D1) +  (NP.rgrid2[k7] ** 2) / (2.0 * D2))) * \
#                                                    special.jn(0, k_omega * NP.rgrid[k6] * NP.rgrid2[k7] / D1) * special.jn(0, k_omega * NP.rgrid_anal[k5] * NP.rgrid2[k7] / D2) * \
#                                                    NP.rgrid[k6] * NP.rgrid2[k7] * SourceTerm[k6]
#                     else: sys.exit('Wrong field storing method')

#                     if (NP.integrator['method'] == 'Trapezoidal'): # it should be optimised by a single built-in sum
#                         for k6 in range(Nr): integral_r1[k6] = integrate.trapz(integrand[k6,:], NP.rgrid2)
#                         FHHGOnScreen[k1, k4, k2, k5] = (1.0/(D1*D2)) * integrate.trapz(integral_r1, NP.rgrid)
#                     else: sys.exit('Wrong integrator')

#     return (k_start, k_num, FHHGOnScreen)


# # def ComputeFieldsPhenomenologicalDipoles_mp(I0SI,omega0,TFWHM,w0,tgrid,omegagrid,rgrid,z_medium):
# #   print('Computing phenomenological dipoles: FFTs')
# #   Nomega = len(omegagrid); Nr = len(rgrid); Nz_medium = len(z_medium);
# #   FField_r=np.empty([Nz_medium,Nomega,Nr], dtype=np.cdouble)
# # #   FField_r=np.empty([Nomega,Nr], dtype=np.cdouble)
# #
# #   tcoeff = 4.0*np.log(2.0)*units.TIMEau**2 / ( TFWHM**2 )
# #   rcoeff = 2.0/(w0**2)
# #   k1 = mn.FindInterval(tgrid,0.0)
# #   dt = tgrid[k1+1]-tgrid[k1]
# #
# #   for k1 in range(Nz_medium):
# #     # # define output queue
# #     # output = mp.Queue()
# #     for k2 in range(Nr): # multiprocessing?
# #       print('kr',k2)
# #       # the expression of the dipole is tricky, see appendix C in Jan Vabek's diploma thesis (using "two-times" Fourier in omega)
# #       dum = dipoleTimeDomainApp(tgrid,rgrid[k2],I0SI/units.INTENSITYau,PhenomParams,tcoeff,rcoeff,omega0) # compute "complex" dipole in t-domain
# #       # add the extra phase
# #       dum = (dt/np.sqrt(2.0*np.pi))*np.fft.fft(dum) # fft with normalisation
# #       # FField_r[k1,:,k2] = dum # !! CANNOT BE DONE THIS WAY, WE HAVE EXTRA ASSUMPTION THAT OUR SIGNAL IS REAL fft: C -> C
# #       for k3 in range(Nomega): FField_r[k1,k3,k2] = dum[k3]
# #   print('FFT computed');
# #   return  FField_r