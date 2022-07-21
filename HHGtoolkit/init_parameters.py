import numpy as np
import HHGtoolkit.conversions as conv
import HHGtoolkit.process as proc
import subprocess as s
import os
import HHGtoolkit.utilities as util

### Write a line of variables into a file
def write_ln(file, var, val, type_, unit, comment=""):
    #v = val
    #print(type_(val))
    if (type(val) == type(np.array(()))):
        write_multiparametric(file, var, val, type_, unit, comment)
    else:
        file.write(
            "{} \t {} \t {} \t {} \t {} \n".format(var, val, type_, unit, comment)
        )

### Write multiparametric variables into a file
def write_multiparametric(file, var, arr, type_, unit, comment=""):
    file.write("$multiparametric {0} \t {1} \n".format(np.shape(arr)[0], comment))
    file.write("{0} \n".format(var))
    file.write("{0} \n".format(type_))
    file.write("{0} \n".format(unit))
    for val in arr:
        file.write("{} \n".format(val))

### Writing simple line into the file.
### Takes string argument, comments must start with '#'
def write_comment(file, comment):
    file.write("{}\n".format(comment))

### Universal input prompt for numerical values
def u_input(
    prompt, type_=None, min_=None, max_=None, values_=None, range_=None
):
    if min_ is not None and max_ is not None and max_ < min_:
        raise ValueError("min_ must be less than or equal to max_.")
    while True:
        ui = input(prompt)
        if type_ is not None:
            try:
                ui = type_(ui)
            except ValueError:
                print("Input type must be {0}.".format(type_.__name__))
                continue
        if max_ is not None and ui > max_:
            print("Input must be less than or equal to {0}.".format(max_))
        elif min_ is not None and ui < min_:
            print("Input must be greater than or equal to {0}.".format(min_))
        elif range_ is not None and ui not in range_:
            if isinstance(range_, range):
                template = "Input must be between {0.start} and {0.stop}."
                print(template.format(range_))
            else:
                template = "Input must be {0}."
                if len(range_) == 1:
                    print(template.format(*range_))
                else:
                    expected = " or ".join(
                        (
                            ", ".join(str(x) for x in range_[:-1]),
                            str(range_[-1]),
                        )
                    )
                    print(template.format(expected))
        elif values_ is not None and ui not in values_:
            template = "Input must be {0}."
            if len(values_) == 1:
                print(template.format(*values_))
            else:
                expected = " or ".join(
                    (", ".join(str(x) for x in values_[:-1]), str(values_[-1]))
                )
                print(template.format(expected))
        else:
            return ui


### Simple function to evaluate yes or no question
def ask(question):
    print(question + " (y/n)")
    a = input()
    if (a == "y"):
        return True
    elif (a == "n"):
        return False
    else:
        print("Type 'y' == yes or 'n' == no.")
        return ask(question)


### Function which prompts input from user and
### converts input data for the simulation and saves it
### as an .inp file
def prompt_input(filename):
    if filename[-4:-1] != ".inp":
        filename = filename + ".inp"

    ### Default values:
    Ip = 0.5
    I = 1e14

    ### Pulse length
    t = 40
    tau = conv.SI_prefix("f")*t

    ### Atomic number
    atomic_number = 1

    ### Quantum numbers
    n = 1
    l = 0
    m = 0

    ### First field parameters
    E0_1 = conv.intensity_to_E(I)
    ell_1 = 0.
    p_1 = 1.
    CEP_1 = 0.
    theta_1 = 0.
    lambda_1 = 800.
    N_cycl_1 = conv.pulse_length_to_cycles(
        conv.lambda_to_omega(lambda_1, "SI", "n"), tau
    )
    omega_0 = conv.lambda_to_omega(lambda_1, "au", "n")

    ### Delay between pulses
    delay = 0

    ### Second field
    E0_2 = 0.
    ell_2 = 0.
    p_2 = 1.
    CEP_2 = 0.
    theta_2 = 0.
    rel_omega = 2
    N_cycl_2 = rel_omega*N_cycl_1
    omega_2 = rel_omega*omega_0
    phi = -0.5
    cosine_pulse = False

    ### Prompt inputs ###

    print("*** Target properties *** \n")
    atomic_number = u_input("Type atomic number: ", int, 1)
    n = u_input("Type principal quantum number: ", int, 1)
    l = u_input("Type radial quantum number: ", int, 0, n - 1)
    m = u_input("Type magnetic quantum number: ", int, -l, l)

    print("\n")
    print("*** Laser pulse properties ***\n")      

    ### First field ###

    print("*** First field properties *** \n")

    if (not
        ask("? Use default value for intensity ({:.2e} W/cm^2)?".format(I))
    ):
        I = u_input("Type field intensity (for example 1E14): ", float, 0)
        E0_1 = conv.intensity_to_E(I)

    if (ask("? Do you want cosine (y) or sine (n) pulse?")):
        CEP_1 = 0.5
        cosine_pulse = True

    ell_1 = u_input(
        "Type ellipticity of the first field (0.0 - 1.0): ", float, 0, 1
    )

    p_1 = u_input(
        "Type sense of polarization of the first field "
        "(-1 anticlockwise, 1 clockwise): ",
        int,
        -1,
        1,
        None,
        [-1, 1],
    )


    if (not
        ask(
            "? Use default value for first field wavelength ({} nm)?"
            " \n".format(lambda_1)
        )
    ):
        lambda_1 = u_input("Type wavelength (in nm): ", float, 0)
        omega_0 = conv.lambda_to_omega(lambda_1)


    if (not ask("? Use default value for pulse length (40 fs)?")):
        t = u_input("Type pulse length (in fs): ", float, 0)
        tau = conv.SI_prefix("f")*t
        N_cycl_1 = conv.pulse_length_to_cycles(
            conv.lambda_to_omega(lambda_1, "SI", "n"), tau
        )

    ### Second field ###

    print("*** Second field properties *** \n")

    A_rel = u_input(
        "Type relative amplitude of the second field: ", float, 0, 10
    )

    ### If relative amplitude == 0 then default values for the
    ### second field are used
    if A_rel > 0.0:
        rel_omega = u_input(
        "Type relative frequency of the second field: ", int, 1, 3
        )
        
        if(ask(
            "? Do you want multiparametric study for relative phases "
            "between fields?"
            )
        ):
            phi_min = u_input(
                "Type minimum phase ((0-2) π rad): ",
                float,
                0,
                2
            )
            phi_max = u_input(
                "Type maximum phase (({}-2) π rad): ".format(phi_min),
                float,
                phi_min,
                2
            )
            N_val = u_input(
                "Type number of values for multiparametric study: ",
                int,
                2
            )

            phi = np.linspace(phi_min, phi_max, N_val)
        else:
            phi = u_input(
                "Type phase shift between fields ((0-2) π rad): ", float, 0, 2
            )

        second_field = conv.secondary_field(
            omega_0,
            rel_omega,
            E0_1,
            A_rel,
            N_cycl_1,
            phi,
            cosine_pulse
        )

        omega_2 = second_field[0]
        E0_2 = second_field[1]
        N_cycl_2 = second_field[2]
        CEP_2 = second_field[3]

        ell_2 = u_input(
            "Type ellipticity of the second field (0.0 - 1.0): ",
            float,
            0,
            1,
        )

        p_2 = u_input(
            "Type sense of polarization of the first field "
            "(-1 anticlockwise, 1 clockwise): ",
            int,
            -1,
            1,
            None,
            [-1, 1],
        )

        theta_2 = u_input(
            "Type deviation between pulses ((0-1) π rad),"
            " 0.5 for orthogonal fields): ",
            float,
            0,
            1,
        )



    N_int = u_input(
        "Type number of points per cycle for integration: ",
        int,
        10
    )

    N_dip = u_input(
        "Type number of points per cycle for dipole evaluation: ",
        int,
        10
    )

    N = 1

    gaussian = ask("? Do you want Gaussian intensity profile for the fields? ")
    if(gaussian):
        N = u_input(
            "Type number of points for distribution discretization: ",
            int, 
            41
        )

        beams = util.Beams()

        w0_1 = u_input(
            "Type waist radius for the first (fundamental) beam: ",
            float,
            1e-8
        )
        z0_1 = u_input(
            "Type beam focus for the first (fundamental) beam: ",
            float
        )

        beams.create_beam(lambda_1*1e-9, E0_1, z_0 = z0_1, w_0 = w0_1, N_pts = N)
        
        if (E0_2 != 0.):
            w0_2 = u_input(
                "Type waist radius for the secondary (harmonic) beam: ",
                float,
                1e-8
            )
            z0_2 = u_input(
                "Type beam focus for the secondary (harmonic) beam: ",
                float
            )
            beams.create_beam(lambda_1*1e-9/rel_omega, E0_2, z_0 = z0_2, w_0 = w0_2, N_pts = N)
        else:
            beams.create_beam(lambda_1*1e-9/rel_omega, E0_2, N_pts = N)

        z_target = u_input(
            "Type target position on the z-axis: ",
            float
        )

        Gouy_phase = ask("? Do you want to implement the Gouy phase?")

        beams.get_beam_profiles(z_target, Gouy_phase=Gouy_phase)

        E_1s, E_2s = np.real([b.profile for b in beams.beams])
        CEP_1s = np.ones(N)*CEP_1
        CEP_2s = np.ones(N)*CEP_2
    else:
        E_1s = [E0_1]
        E_2s = [E0_2]
        CEP_1s = [CEP_1]
        CEP_2s = [CEP_2]




    filenames = [filename[0:-4]+"_{}.inp".format(i+1) for i in range(N)]
    ### Write parameters to file ###
    for i, file in enumerate(filenames):
        with open(file, "w") as f:
            write_ln(f, "ground_state_energy", Ip, "R", "[a.u.]")

            write_ln(f, "principal_quantum_number", n, "I", "-")
            write_ln(f, "radial_quantum_number", l, "I", "-")
            write_ln(f, "magnetic_quantum_number", m, "I", "-")
            write_ln(f, "atomic_number", atomic_number, "I", "-")

            write_comment(f, "")
            write_ln(f, "E0_1", E_1s[i], "R", "a.u.")
            write_ln(f, "ellipticity_1", ell_1, "R", "-")
            write_ln(f, "polarisation_direction_1", p_1, "I", "-")
            ### Include phase shift originating from the beam curvature
            if (z0_1 != z_target):
                beam_profile = beams.beams[0].profile[i]
                E_im, E_re = np.imag(beam_profile), np.real(beam_profile)
                ### Obtain the phase and normalize to pi radians
                CEP_1s[i] = CEP_1s[i] + np.arctan2(E_im, E_re)/np.pi
            write_ln(f, "CEP_1", CEP_1s[i], "R", "pi_rad")
            write_ln(f, "theta1", theta_1, "R", "pi_rad")
            write_ln(
                f,
                "number_of_cycles_1",
                N_cycl_1,
                "R",
                "-",
            )
            write_ln(f, "omega_1", omega_0, "R", "a.u.")
            if gaussian:
                write_ln(f, "waist_1", w0_1, "R", "m")
                write_ln(f, "focus_1", z0_1, "R", "m")
                write_ln(f, "z_target", z_target, "R", "m")

            write_comment(f, "")
            write_ln(f, "delay_between_pulses", 0.0, "R", "a.u.")
            write_comment(f, "")

            write_ln(f, "E0_2", E_2s[i], "R", "a.u.")
            write_ln(f, "ellipticity_2", ell_2, "R", "-")
            write_ln(f, "polarisation_direction_2", p_2, "I", "-")
            ### Include phase shift originating from the beam curvature
            if (z0_2 != z_target):
                beam_profile = beams.beams[1].profile[i]
                E_im, E_re = np.imag(beam_profile), np.real(beam_profile)
                ### Obtain the phase and normalize to pi radians
                CEP_2s[i] = CEP_2s[i] + np.arctan2(E_im, E_re)/np.pi
            write_ln(f, "CEP_2", CEP_2s[i], "R", "pi_rad")
            write_ln(f, "theta_2", theta_2, "R", "pi_rad")
            write_ln(f, "number_of_cycles_2", N_cycl_2, "R", "-")
            write_ln(f, "omega_2", omega_2, "R", "a.u.")
            if gaussian:
                write_ln(f, "waist_2", w0_2, "R", "m")
                write_ln(f, "focus_2", z0_2, "R", "m")

            write_comment(f, "")
            write_ln(f, "points_per_cycle_for_integration", N_int, "I", "-")
            write_ln(f, "points_per_cycle_for_evaluation", N_dip, "I", "-")
            if gaussian:
                write_ln(f, "points_per_field_distribution", N, "I", "-")


def is_in_dir(filename):
    list_files = os.listdir()
    return any(elem in filename for elem in list_files)


def create_params(filename):
    if (not is_in_dir(filename)):
        print("File does not exist!")
        return
    else:
        if (is_in_dir(filename[0:-4]+".h5")):
            if(ask("HDF5 archive already exists, do you wish to overwrite it?")):
                 s.run(["rm", filename[0:-4]+".h5"])
            else:
                return

        args = ["python3", "universal_input/create_universal_HDF5.py", 
                "-i", filename, "-ohdf5", filename[0:-4]+".h5", "-g", "inputs"]

        s.run(args)
        print("HDF5 archive '"+ filename[0:-4]+".h5' created.")
    
        if (is_in_dir("multiparameters")):
            if (ask("Do you want to rename the 'multiparameters' archive?")):
                print("Type the archive name: ")
                arch = input()
                s.run(["mv", "multiparameters", arch])


if __name__ == '__main__':
    print("Type file name: ")
    h5_filename = input()
    prompt_input(h5_filename)

    create_params(h5_filename)
