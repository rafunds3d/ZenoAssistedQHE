####################################################################
#################-----IMPORTANT REMARKS-----########################
####################################################################

# Remark 1) In our convention, the ground state of the system is  
# provided by the normalized vector state |1>. This is because we
# write, for example, H = 1/2(|0><0| - |1><1|) and take |1> to have
# eigenvalue -1/2 and |0> to have eigenvalue 1/2. 

# Remark 2) There are various 'TESTS' in the code. These are  
# consistency check. In order to implement them it suffices to 
# uncomment them.

# Remark 3) The code is specific for 2-level systems, so it assumes a 
# 2-level system as a working fluid and a specific 2-level system 
# lubricant

# Remark 4) In our code, we have that the input of the non-interacting
# system Hamiltonian to the Zeno drive needs to be 

# H_sys0 = omega/2 Z 

# which is distinct from the other one which is described by H_sys1. In
# that case we have that the input should be 

# H_sys1 =  X

# in other words, NOT including the Omega(t). Be careful with this choice

####################################################################

import numpy as np
from qutip import *
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def H_total_at_t(t, H_joint_base, H_sys1_joint, Z_sys_X_lub, X_sys_X_lub,
                 Omega_cb, coeff_Z_cb, coeff_X_cb):
    return (
        H_joint_base
        + Omega_cb(t) * H_sys1_joint
        + coeff_Z_cb(t) * Z_sys_X_lub
        + coeff_X_cb(t) * X_sys_X_lub
    )



def h_expectation(H, rho):
    """
    Compute the expectation value of the Hamiltonian `H` with respect to the
    quantum state `rho`.

    The expectation value is given by

        ⟨H⟩ = Tr(H ρ)

    where `ρ` is the density matrix describing the quantum state. In general
    the trace may be complex due to numerical precision, so the real part is
    returned.

    Parameters
    ----------
    H : qutip.Qobj
        Hamiltonian operator.
    rho : qutip.Qobj
        Density matrix defined on the same Hilbert space as `H`.

    Returns
    -------
    float
        The real-valued expectation value Tr(H ρ).
    """
    val = (H * rho).tr()

    #######################################
    ##############--TEST--#################
    #######################################

    # Use this to note that the output is a complex number
    # where there is a very small non-zero imaginary term
    # that is due to numerical instability

    # print("Expectation value of the Hamiltonian:", val) 

    # Because of that we take the real part only
    #######################################

    return float(np.real(val))

def l1_coherence(H, rho):
    """
    Compute the ℓ1-coherence of the density matrix `rho` in the eigenbasis of
    the Hamiltonian `H`.

    The ℓ1-coherence is defined as the sum of the absolute values of all
    off-diagonal elements of `rho` when expressed in the eigenbasis of `H`:

        C = sum_{i ≠ j} |ρ_ij|

    where ρ_ij = ⟨E_i| ρ |E_j⟩ and |E_i⟩ are eigenstates of H.

    Parameters
    ----------
    H : qutip.Qobj
        Hamiltonian whose eigenbasis defines the reference basis.
    rho : qutip.Qobj
        Density matrix defined on the same Hilbert space as `H`.

    Returns
    -------
    float
        The ℓ1-coherence of `rho` in the eigenbasis of `H`.
    """
    
    # Note that this code works for any N-level system
    
    eigvals, eigkets = H.eigenstates()

    C = 0.0 # Start with no coherence
    N = len(eigkets) # Number of eigenvectors 
    
    for i in range(N):
        for j in range(N):
            if i != j:
                val = (eigkets[i].dag() * rho * eigkets[j]) # Careful with chatGPT for this part here
                C += abs(val)
    return float(C)

def gibbs_state(H, T=None, beta=None, eps=1e-14):
    """
    Return the Gibbs state rho = exp(-beta H)/Tr[exp(-beta H)] as a Qobj.

    It only works for the 2-level system case! 
    
    Parameters
    ----------
    H : Qobj
        System Hamiltonian (Hermitian).
    T : float or None
        Temperature (k_B = 1 units). If given, beta = 1/T.
    beta : float or None
        Inverse temperature beta = 1/T. If provided, takes precedence over T.
    eps : float
        Threshold for treating beta as "very large" (zero-temperature limit).
    
    Returns
    -------
    rho : Qobj
        Density matrix (same dimension / basis as H).
    """

    #--------------------------------------------------------------------------
    # Handle the case where neither T nor beta was provided, as well as T < 0
    #--------------------------------------------------------------------------

    if beta is None:
        if T is None:
            raise ValueError("Either T or beta must be provided!!")
        if T <= 0:
            # Treat T == 0 as zero-temperature limit. Should be used to output the ground state.
            if T == 0:
                beta = np.inf
            else:
                raise ValueError("Temperature T must be >= 0 (T=0 allowed for ground state).")
        else:
            beta = 1.0 / T

    #--------------------------------------------------------------------------
    # Handle the case where T = 0 (exact zero-temperature) and thus beta = +inf
    #--------------------------------------------------------------------------

    if np.isinf(beta) or beta > 1.0/eps: # If beta is larger than the threshold we simply assume T=0
        
        eigvals, eigkets = H.eigenstates() # extracts eigenvalues and eigenvectors
        
        # Ordering so that the ground state is given by the eigenvector with smallest eigenvalue
        idx_min = int(np.argmin(np.real(eigvals))) # Explicitly making eigenvalues real
        ket_g = eigkets[idx_min]
        rho_gs = ket_g * ket_g.dag() # Make the ket into a density matrix

        #######################################
        ##############--TEST--#################
        #######################################

        # print('Output state if T = 0:', rho_gs) # This should output the state \rho = |1><1| if T = 0 since this is our ground state
        
        #######################################

        return rho_gs

    # Stable eigen-decomposition construction:
    eigvals, eigkets = H.eigenstates()
    
    # real parts of eigenvalues (should be real for Hermitian H but we force them to be real for safety)
    E = np.real(eigvals)

    # Constructing the terms to generate a Gibbs state
    boltz = np.exp(-beta * E)
    Z = np.sum(boltz)
    probs = boltz / Z

    # build the classical Gibbs state rho = sum_i p_i |e_i><e_i|
    rho = probs[0] * (eigkets[0] * eigkets[0].dag()) + probs[1] * (eigkets[1] * eigkets[1].dag())  # zero operator of correct shape

    #######################################
    ##############--TEST--#################
    #######################################

    # print('Output Gibbs state:', rho) 
        
    #######################################

    return rho

def project_to_inst_eigenbasis(H, rho_sys, prefer_basis_index: int = 0):
    """
    Project `rho_sys` into the instantaneous eigenbasis of `H`, ordering the
    eigenvectors so that the one with the largest overlap with the computational
    basis state |prefer_basis_index> comes first.

    Parameters
    ----------
    H : qutip.Qobj
        Hamiltonian whose eigenbasis is used.
    rho_sys : qutip.Qobj
        System density matrix (Qobj).
    prefer_basis_index : int, optional
        Index of the computational basis state to prefer when ordering (default 0 -> |0>).

    Returns
    -------
    rho_sys_eig : qutip.Qobj
        The density matrix expressed in the reordered instantaneous eigenbasis.
    U : qutip.Qobj
        The unitary whose columns are the ordered eigenkets (so rho_sys_eig = U.dag() * rho_sys * U).
    """
    # Eigen-decomposition (returns eigenvalues, list of ket Qobjs)
    eigvals, eigkets = H.eigenstates()

    # Infer system dimension (supports >2 dims)
    dim = int(rho_sys.shape[0])

    # Build computational basis vector to compare overlaps with
    prefer_ket = basis(dim, prefer_basis_index)

    # Compute overlaps (use Qobj.overlap for clarity)
    overlaps = np.array([abs(ket.overlap(prefer_ket)) for ket in eigkets])

    # Order eigenvectors by descending overlap with prefer_ket
    order = np.argsort(-overlaps)

    # Stack ordered eigenkets as columns into a single numpy array and convert to Qobj
    U_mat = np.hstack([eigkets[i].full() for i in order])
    U = Qobj(U_mat, dims=rho_sys.dims)

    # Transform rho into the instantaneous eigenbasis
    rho_sys_eig = U.dag() * rho_sys * U

    return rho_sys_eig, U # Return the state in the instantaneous eigenbasis of H

def logarithmic_negativity(rho, subsystem=0):
    """
    Compute the logarithmic negativity of a two-qubit state.
    Works with any QuTiP version (no need for qutip.norm).
    """
    # If input is a ket, convert to density matrix
    if rho.isket:
        rho = ket2dm(rho)

    # Ensure the state is on two qubits
    if rho.dims != [[2, 2], [2, 2]]:
        raise ValueError("State must be a two-qubit state (dims = [[2,2],[2,2]])")

    # Partial transpose with respect to the chosen subsystem
    rho_pt = partial_transpose(rho, [subsystem, 1])

    # Trace norm = sum of singular values (computed with NumPy)
    singular_vals = np.linalg.svd(rho_pt.full(), compute_uv=False)
    trace_norm = np.sum(singular_vals)

    # Logarithmic negativity (base 2)
    return np.log2(trace_norm)

def trace_distance(rho1, rho2):
    """
    Compute the trace distance D_tr(rho1, rho2) = 1/2 * ||rho1 - rho2||_1.

    Parameters
    ----------
    rho1, rho2 : qutip.Qobj
        Input states; may be kets or density matrices. They must act on the
        same Hilbert space (same shape / dims).

    Returns
    -------
    float
        The trace distance (in [0, 1]).
    """
    # Convert kets to density matrices if needed
    if rho1.isket:
        rho1 = ket2dm(rho1)
    if rho2.isket:
        rho2 = ket2dm(rho2)

    # Basic shape/dimension check
    if rho1.shape != rho2.shape:
        raise ValueError("rho1 and rho2 must have the same shape / act on the same Hilbert space.")

    # Delta is Hermitian (difference of two Hermitians). Use eigvalsh for stability.
    delta = (rho1 - rho2).full()  # dense numpy array
    eigs = np.linalg.eigvalsh(delta)  # real eigenvalues for Hermitian matrix
    trace_norm = np.sum(np.abs(eigs))

    return float(0.5 * trace_norm)

# ------------------------------------------------------------------
# Zeno-drive simulator
# ------------------------------------------------------------------

def simulate_zeno_drive_joint(
    H_sys0, H_sys1, H_L,        # Hamiltonians of system and lubricant
    Omega_func,                 # Function for changing the spectral gap
    rho_initial_joint,          # Initial state of SL
    t_initial, t_final,         # Duration of the work stroke
    Gamma=1000.0,               # Strenght of the coupling, if not specified set to be 1000
    measurements=1000,          # Total number of measurements equally spaced, if not specified set to be 1000
    n_steps_per_interval=50,    # per interval (t_final - t_initial)/measurements we drive the system unitarily
    selective=True,             # Choice of implementing a selective or a non-selective Zeno
    args=None,                  # Args used to pass on specifications of the interaction such as omega or Omega0
    atol=1e-15, rtol=1e-15      # High precision required for the drive in the short intervals
    ):
    """
    Simulate a driven system–lubricant system with frequent equally sparsed
    measurements implementing a quantum Zeno–drive as in the paper.

    The two-level system (S) is coupled to a two-level lubricant (L). During each
    interval the joint state evolves unitarily under a time-dependent
    Hamiltonian. After each interval a projective measurement is performed
    on the lubricant in the computational basis {|0>, |1>}.

    The protocol therefore alternates between

        1. Unitary evolution of the joint system (S ⊗ L)
        2. Measurement of the lubricant

    If `selective=True`, the post-measurement state is conditioned on the
    measurement outcome (single trajectory). If `selective=False`, the
    measurement is non-selective and produces a dephased state.

    The system Hamiltonian is

        H_S(t) = H_sys0 + Ω(t) H_sys1

    where Ω(t) is provided by `Omega_func`. The joint Hamiltonian includes
    an additional strong coupling term controlled by Γ that implements the
    Zeno interaction between system and lubricant.

    Parameters
    ----------
    H_sys0 : qutip.Qobj
        Static part of the system Hamiltonian.

    H_sys1 : qutip.Qobj
        Operator multiplying the time-dependent drive Ω(t).

    H_L : qutip.Qobj
        Hamiltonian of the lubricant qubit.
    
    Omega_func : callable
        Function Ω(t, args) defining the time-dependent drive.
    
    rho_initial_joint : qutip.Qobj
        Initial joint density matrix of system and lubricant.
    
    t_initial : float
        Initial time of the protocol.
    
    t_final : float
        Final time of the protocol.
    
    Gamma : float, optional
        Strength of the system–lubricant coupling implementing the Zeno
        interaction. Default is 1000.
    
    measurements : int, optional
        Number of equally spaced measurements performed during the drive.
        Default is 1000.
    
    n_steps_per_interval : int, optional
        Number of solver time steps used inside each measurement interval.
    
    selective : bool, optional
        If True, simulate a selective measurement trajectory. If False,
        perform non-selective measurements (dephasing channel).
    
    args : dict, optional
        Dictionary of parameters passed to `Omega_func`.
    
    atol, rtol : float, optional
        Absolute and relative tolerances used by the ODE solver.

    Returns
    -------
    
    Dictionary containing simulation data:

        tlist : ndarray
            Times at which measurements occur.

        states_before : list
            Reduced system states immediately before each measurement
            (expressed in the instantaneous eigenbasis).

        states_after : list
            Reduced system states immediately after the measurement.

        H_sys_inst : list
            Instantaneous system Hamiltonians H_S(t).

        expect_energy : ndarray
            Expectation values ⟨H_S(t)⟩ evaluated after measurement.

        coherences_before : ndarray
            ℓ1-coherence of the system state before measurement.

        coherences_after : ndarray
            ℓ1-coherence of the system state after measurement.

        rho_final : qutip.Qobj
            Final reduced system state in the instantaneous eigenbasis.

        coherences : ndarray
            Diagnostic array used for testing/debugging.
    """


    if args is None:
        raise ValueError("No argument passed to the simulation; missing information on omega, Omega0, etc!!")

    # Total time and interval length for the duraction of the work stroke
    tau = t_final - t_initial

    if measurements < 1: # If there are no measurements, we simply drive the system during an interval tau
        measurements = 1
    
    # Each unitary pulse has duration dt =  tau/measurements 
    dt = tau / float(measurements)

    # Define projectors on lubricant (computational basis |0>,|1>) 
    P0 = basis(2, 0) * basis(2, 0).dag() # |0><0|
    P1 = basis(2, 1) * basis(2, 1).dag() # |1><1|

    ket_plus  = (basis(2, 0) + basis(2, 1)).unit()
    ket_minus = (basis(2, 0) - basis(2, 1)).unit()

    # Cefine projectors on lubricant (coherent basis |#>,|->) 
    Pp = ket_plus * ket_plus.dag()
    Pm = ket_minus * ket_minus.dag()

    # Identities for both systems
    I_sys = qeye(2)
    I_lub = qeye(2)

    # Preparing the Hamiltonians
    H_joint_base = tensor(H_sys0, I_lub) + tensor(I_sys, H_L) # H_0 x I + I x H_L
    H_sys1_joint = tensor(H_sys1, I_lub) # H_1(t) x I

    # print("H_free:\n", H_sys0)

    # Here we are computing the relevant operators to use in R(t) \otimes X
    Z_sys_X_lub = tensor(sigmaz(), sigmax())   # Z (system) \otimes X (lubricant)
    X_sys_X_lub = tensor(sigmax(), sigmax())   # X (system) \otimes X (lubricant)

    omega_sys = args.get('omega', 1.0) # This picks omega from 'args' (if args={} this generates a  ValueError; see above)

    # Helper for the function Omega.
    # It picks up the function Omega_func, 
    # so it could also be used in different strokes

    def Omega_cb(t, args):
        return float(Omega_func(t, args))

    # Callbacks for the two scalar coefficients Gamma * cos(theta(t)) and Gamma * sin(theta(t))
    def coeff_Z_cb(t, _args=None):
        Omega_t = float(Omega_func(t, args))
        theta = np.arctan2(2.0 * Omega_t, omega_sys)   # theta(t) = atan2( 2*Omega / omega )

        return float(Gamma * np.cos(theta))

    def coeff_X_cb(t, _args=None):
        Omega_t = float(Omega_func(t, args))
        theta = np.arctan2(2.0 * Omega_t, omega_sys)

        return float(Gamma * np.sin(theta))

    # containers for results
    tlist_all = []
    states_all_before = []
    states_all_after = []
    H_sys_inst_list = []
    energies = []
    coherences_before = []
    coherences_after = []
    coherences = []


    # Initial state of the system
    rho = rho_initial_joint
    #print('Rounded state of the system:\n', np.round(rho.ptrace(0).full(),3))  # Check the state of the working system

    #######################################
    ##############--TEST--#################
    #######################################

    # print('Initial joint state of SL:', rho) # This tests what is the initial state of the composite system+lubricant

    # If the code is correct, it should be a diagonal state with 
    # the diagonal given by Gibbs distributions relative to the 
    # temperature of the cold bath T_c
        
    #######################################
    

    # Iterate over measurement intervals
    for k in range(measurements):
        # print('Rounded state of the lubricant \n BEFORE MEASUREMENTS:\n', np.round(rho.ptrace(1).full(),4))  # Check the state of the lubricant
        
        t0 = t_initial + k * dt # If measurement = 1 then k = 0 and  dt = tau (since range(1) = [0])
        t1 = t0 + dt            # Take a dt step towards t1
        # print("Time:", t0)
        # Internal time grid for this interval
        tlist_interval = np.linspace(t0, t1, max(2, n_steps_per_interval))

        # Build H_list for mesolve: time-independent base + H_sys1_joint * Omega(t) + R_joint * Gamma
        H_list = [
            H_joint_base,                        # H_0 x I + I x H_L
            [H_sys1_joint, Omega_cb],            # H1 Omega(t) x I
            [Z_sys_X_lub, coeff_Z_cb],           # Gamma * cos(theta(t)) * Z \otimes X
            [X_sys_X_lub, coeff_X_cb]            # Gamma * sin(theta(t)) * X \otimes X
            ]

        opts = Options(atol=atol, rtol=rtol, nsteps=100000, store_states=True)

        # solve for this interval
        res = mesolve(H_list, rho, list(tlist_interval), c_ops=[], e_ops=[], args=args, options=opts)

        states_interval = res.states   # Here, states_interval has all the states in the interval dt. However, we are only interested in the last one        
        rho = states_interval[-1]      # update rho to the final state after the evolution takes place


        #######################################
        ##############--TEST--#################
        #######################################

        # print('Time:', t0+t1) 
        # print('Rounded state of the composite:\n', np.round(rho.full(),3))  # Check the state of the composite system
        # H_sys_last = H_sys0 + Omega_cb(t1,args) * H_sys1
        # print("H sys as time goes: \n ", H_sys_last)
        # test_sys_state = project_to_inst_eigenbasis(H_sys_last,rho.ptrace(0))
        # print('Rounded state of the SYSTEM:\n', test_sys_state)  # Check the state of the working system
        # print('Rounded state of the lubricant:\n', np.round(rho.ptrace(1).full(),4))  # Check the state of the lubricant
        # print("Number of measurements:", measurements)
        #rho_prod = tensor(rho.ptrace(0),rho.ptrace(1))
        #print('Rounded state of the tensor product:\n', np.round(rho_prod.full(),3))  # Check the state of the composite system

        # distance = trace_distance(rho,rho_prod)
        # print('Distance between rho and its prod:', distance)
        
        #######################################


        # Calculate the instantaneous coherences and energies of rho at t0 = t_initial + k * dt
        H_sys_inst = H_sys0 + Omega_func(t1, args) * H_sys1 # H_S(t)
        H_sys_inst_list.append(H_sys_inst) # Probably totally stupid to be doing that

        # Reduced state on system before measurements take place
        rho_sys = rho.ptrace(0)

        #######################################
        ##############--TEST--#################
        #######################################

        # Instantaneous_coherence_test = l1_coherence(H_sys_inst, rho_sys)
        # print('Instantaneous coherence before the measurements:\n', Instantaneous_coherence_test) 
        
        #######################################

        coherences_before.append(l1_coherence(H_sys_inst, rho_sys)) # Colects the instantaneous coherences before the measurement takes place

        rho_sys_eig, U = project_to_inst_eigenbasis(H_sys_inst, rho_sys, prefer_basis_index=0) # finds out the state in the instantaneous eigenbasis

        #######################################
        ##############--TEST--#################
        #######################################

        # print('State of the working system in the instantaneous eigenbasis \n (should be diagonal if successfull lubrication):\n', np.round(rho_sys_eig.full(),7)) 
        
        #######################################
        

        tlist_all.append(t0)
        states_all_before.append(rho_sys_eig)

        # Projectors for performing measurement on the lubricant
        P0_joint = tensor(I_sys, P0) # I \otimes |0><0|
        P1_joint = tensor(I_sys, P1) # I \otimes |1><1|
        Pp_joint = tensor(I_sys, Pp) # I \otimes |+><+|
        Pm_joint = tensor(I_sys, Pm) # I \otimes |-><-|

        if measurements!= 1:
            if selective:
                # Compute probabilities
                # p0 = float(np.real((P0_joint * rho).tr())) # |0><0| case
                p0 = float(np.real((Pp_joint * rho).tr()))   # |+><+| case
                
                #######################################
                ##############--TEST--#################
                #######################################

                # print('Probability of obtaining |0><0|:', p0) 
                
                #######################################

                p1 = 1 - p0 # equal to float(np.real((P1_joint * rho).tr()))

                # Flip a biased coin simulating the experiment
                r = '0' if random.random() < p0 else '1'

                if r == '0':
                    # rho = (P0_joint * rho * P0_joint) / p0 # |0><0| case

                    rho = (Pp_joint * rho * Pp_joint) / p0 # |+><+| case

                    #######################################
                    ##############--TEST--#################
                    #######################################

                    # print('State of the lubricant if |0><0| is observed:\,', rho.ptrace(1)) 
                    
                    #######################################
                else:
                    # rho = (P1_joint * rho * P1_joint) / p1 # |1><1| case

                    rho = (Pm_joint * rho * Pm_joint) / p1 # |-><-| case
                    
                    #######################################
                    ##############--TEST--#################
                    #######################################

                    # print('State of the lubricant if |1><1| is observed:\,', rho.ptrace(1)) 
                    
                    #######################################


                # Reduced state on system after the measurement takes place
                rho_sys = rho.ptrace(0)

                # Calculates the coherence after the measurement takes place
                coherences_after.append(l1_coherence(H_sys_inst, rho_sys))

                # Calculate the energy after the measurement takes place
                energies.append(h_expectation(H_sys_inst, rho_sys))
                # print("ENERGY:", h_expectation(H_sys_inst, rho_sys))

                states_all_after.append(rho_sys)

            else:
                # Non-selective case has not been tested up until now 
                # non-selective measurement (dephasing in Z basis of lubricant)
                # rho = P0_joint * rho * P0_joint + P1_joint * rho * P1_joint # computational basis case

                rho = P1_joint * rho * P1_joint + Pm_joint * rho * Pm_joint # X basis case

        
        # right after the selective measurement update  
        # print("Lubricant immediately after measurement:")
        # print(np.round(rho.ptrace(1).full(), 4))
        
        # Calculate the energy
        energies.append(h_expectation(H_sys_inst, rho_sys))
        #print("ENERGY:", h_expectation(H_sys_inst, rho_sys))
    
    #######################################
    ##############--TEST--#################
    #######################################

    # print('Joint state after lubrication:', np.round(rho.full(),4)) # This tests what is the final state of the composite system+lubricant
    # log_neg = logarithmic_negativity(rho)
    # print('Value of Gamma:', Gamma)
    # print('Number of measurments:', measurements)
    # print('Amount of entanglement:', log_neg)
    # rho_prod = tensor(rho.ptrace(0),rho.ptrace(1))
    # print('Rounded state of the product:\n', np.round(np.real(rho_prod.full()),3))  # Check the state of the composite system
    # distance = tracedist(rho,rho_prod)
    # print('Distance between rho and its prod:', distance)
        
    #######################################

    
    rho_sys = rho.ptrace(0) # Reduced state of the system after the entire lubrication

    rho_lubricant = rho.ptrace(1) # Reduced state of the lubricant after the entire lubrication

    rho_sys_eig, U = project_to_inst_eigenbasis(H_sys_inst, rho_sys, prefer_basis_index=0)  # Reduced state of the system after the entire lubrication in the instantaneous eigenbasis


    #######################################
    ##############--TEST--#################
    #######################################
    # print("-------------------------------------------------")
    # print("-------------------------------------------------")
    # print("State of the system after lubrication:\n", rho_sys)
    
    # print("-------------------------------------------------")
    # print("-------------------------------------------------")
    # print("State of the lubricant after lubrication:\n", np.round(rho_lubricant.full(), 4)) # If there are sufficiently many measurements it should be always |0><0|
    # Average_Z = np.real((sigmaz() * rho_lubricant).tr())
    #Average_X = np.real((sigmax() * rho_lubricant).tr())
    #Useful_test_state = gibbs_state(0.5* sigmaz(),0.5)
    #p_0 = float(Useful_test_state.full()[0,0])
    # print(p_0)
    # print("Average of X on sigma:", float(Gamma * p_0 * Average_X))
    
    # print('State of system after lubrication\n (in its instantaneous eigenbasis):\n', np.round(rho_sys_eig.full(),4)) # Reduced state of the system after the entire lubrication in the instantaneous eigenbasis

        
    #######################################

    # convert lists to numpy arrays where appropriate
    return {
        'tlist': np.array(tlist_all),               # [t0, t0+dt, t0+2dt, ..., t1]
        'states_before': states_all_before,         # [rho(t0), rho(t0+dt), ..., rho(t1)]
        'states_after': states_all_after,           # [rho^M(t0), rho^M(t0+dt), ..., rho^M(t1)]
        'H_sys_inst': H_sys_inst_list,              # [H_S(t0), H_S(t0+dt), ..., H_S(t1)]
        'expect_energy': np.array(energies),        # [<H_S(t0)>_{rho^M(t0)}, <H_S(t0+dt)>_{rho^M(t0+dt)}, ..., <H_S(t1)>_{rho^M(t1)}]
        'coherences_after': np.array(coherences_after),     # [C(rho^M(t0)), C(rho^M(t0+dt)), ..., C(rho^M(t1))]
        'coherences_before': np.array(coherences_before),   # [C(rho(t0)), C(rho(t0+dt)), ..., C(rho(t1))]
        'rho_final': rho_sys_eig,
        'coherences': np.array(coherences) # test
    }

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    # Input parameters
    precision=1
    start=5
    omega = 1.0
    Omega0 = 3.01105 
    T_cold = 0.5
    T_hot = 3.0
    tau_compression_maximum = 50
    Gamma = 0
    measurements = 0
    
    n_steps_drive = 10000
    omega_lubricant = 2
    selective=True

    # Top-level control Booleans
    RUN_NON_LUB = True      # TRUE if you want the final plot to have also the non-lubricated plot for comparison; otherwise use FALSE
    SAVE_DATA   = False      # TRUE if you want to save the outputs in some txt file for plotting later
    PLOT        = True      # TRUE if you want to see a plot at the end

    for varying in np.arange(Gamma,100*Gamma+1,1): # This 'varying' variable is used for changing either Gamma or the number of measurements
        if RUN_NON_LUB:
            # ------------------------------------------------------------------
            # Non-lubricated simulation
            # ------------------------------------------------------------------

            extracted_state = []
            
            for step in np.arange(start,tau_compression_maximum+precision,precision):
                t_final_comp = step
                # Input parameters
                t_initial_comp = 0.0
                #t_final_comp = 10.0

                # Without lubrication these must be zero
                Gamma_non_lub = 0.0 
                measurements_non_lub = 0 

                # Stroke durations
                tau_comp = t_final_comp - t_initial_comp
                
                # Define system and lubricant Hamiltonians
                H_sys0 = (omega / 2.0) * sigmaz()
                H_sys1 = sigmax()
                H_L = (omega_lubricant / 2.0) * sigmaz()

                # Preparation 
                rho_sys_cold = gibbs_state(H_sys0, T=T_cold)        # system in cold thermal state
                
                # Lubricant states
                # -----------------------------------------------------------------------------
                # rho_lub_init = basis(2, 0) * basis(2, 0).dag()      # lubricant in |0><0|
                # rho_lub_init = basis(2, 1) * basis(2, 1).dag()      # lubricant in |1><1|

                
                ket_plus  = (basis(2, 0) + basis(2, 1)).unit()
                ket_minus = (basis(2, 0) - basis(2, 1)).unit()

                Pp = ket_plus * ket_plus.dag()
                Pm = ket_minus * ket_minus.dag()

                rho_lub_init = Pp  # lubricant in |+><+|
                # -----------------------------------------------------------------------------


                rho_joint_init = tensor(rho_sys_cold, rho_lub_init) 

                # Compression Omega function: Omega(t) goes as 0 -> Omega0/2
                def Omega_comp(t, args):
                    return args['Omega0'] * (t / (2.0 * args['tau']))

                comp_args = {'Omega0': Omega0, 'tau': tau_comp, 'omega': omega}        

                sim_comp = simulate_zeno_drive_joint(
                    H_sys0, H_sys1, H_L,
                    Omega_comp,
                    rho_joint_init,
                    t_initial_comp, t_final_comp,
                    Gamma=Gamma_non_lub,
                    measurements=measurements_non_lub,
                    n_steps_per_interval=max(2, int(n_steps_drive / max(1, measurements))),
                    selective=selective,
                    args=comp_args
                )

                extracted_state_element = sim_comp['rho_final'] # Extracting the final state
                Coherence_of_extracted_state = 2*abs(extracted_state_element.full()[0,1]) # The final state is already in the instantaneous eigenbasis

                extracted_state.append(Coherence_of_extracted_state) # Append the coherences 

        extracted_lubricated_state = []

        # ------------------------------------------------------------------
        # Lubricated simulation
        # ------------------------------------------------------------------

        for step in np.arange(start,tau_compression_maximum+precision,precision):
            print(step)
            t_final_comp = step
            
            t_initial_comp = 0.0
            Gamma = varying             # uncomment if you want to this to vary
            # measurements = varying    # uncomment if you want to this to vary


            # Stroke duration
            tau_comp = t_final_comp - t_initial_comp 

            # Define system and lubricant Hamiltonians
            H_sys0 = (omega / 2.0) * sigmaz()
            H_sys1 = sigmax()
            H_L = (omega_lubricant / 2.0) * sigmaz() 

            # Preparation 
            rho_sys_cold = gibbs_state(H_sys0, T=T_cold)  
            
            # Lubricant states
            # -----------------------------------------------------------------------------
            # rho_lub_init = basis(2, 0) * basis(2, 0).dag()      # lubricant in |0><0|
            # rho_lub_init = basis(2, 1) * basis(2, 1).dag()      # lubricant in |1><1|

                
            ket_plus  = (basis(2, 0) + basis(2, 1)).unit()
            ket_minus = (basis(2, 0) - basis(2, 1)).unit()
            Pp = ket_plus * ket_plus.dag()
            Pm = ket_minus * ket_minus.dag()

            rho_lub_init = Pp  # lubricant in |+><+|
            # -----------------------------------------------------------------------------

            rho_joint_init = tensor(rho_sys_cold, rho_lub_init)

            # Compression function
            def Omega_comp(t, args):
                return args['Omega0'] * (t / (2.0 * args['tau']))

            comp_args = {'Omega0': Omega0, 'tau': tau_comp, 'omega': omega}

            # Main simulation
            sim_comp = simulate_zeno_drive_joint(
                H_sys0, H_sys1, H_L,
                Omega_comp,
                rho_joint_init,
                t_initial_comp, t_final_comp,
                Gamma=Gamma,
                measurements=measurements,
                n_steps_per_interval=max(2, int(n_steps_drive / max(1, measurements))),
                selective=selective,
                args=comp_args
            )

            extracted_state_lubricated_element = sim_comp['rho_final']

            # Calculate the coherences
            Coherence_of_extracted_lubricated_state = 2*abs(extracted_state_lubricated_element.full()[0,1])
            extracted_lubricated_state.append(Coherence_of_extracted_lubricated_state)


        #######################
        # Plotting the results
        #######################

        if RUN_NON_LUB:
            extracted_state = np.array(extracted_state)
        
        extracted_lubricated_state = np.array(extracted_lubricated_state)
        
        x = np.arange(start,tau_compression_maximum+precision,precision)   
        
        #######################
        # SAVING THE RESULTS 
        #######################
        if SAVE_DATA:
            data = np.column_stack((x, extracted_lubricated_state))
            np.savetxt(f'SUPERDENSE_lubricated_results_Measurements_{int(measurements)}_Gamma_{int(Gamma)}.txt', # Change depending on what is varying
                data,
                header='x extracted_lubricated',
                comments='',
                fmt='%.18e')
        
        # Arbitrary color scheme

        color_lubricated = (250/255, 117/255, 0/255)

        if PLOT:
            plt.figure(figsize=(8,4))
            if RUN_NON_LUB:
                # plt.plot(x, extracted_state, marker='o', label='Non-lubricated')
                plt.plot(x, extracted_state, linewidth=2)

            #plt.plot(x, extracted_lubricated_state, marker='^', color=color_lubricated, label='Lubricated')
            plt.xlabel(r'$\tau_{\rm comp}$', fontsize=20)
            plt.ylabel(r'$C_{\ell_1}$', fontsize=20)
            plt.tick_params(axis='both', labelsize=12)
            plt.grid(True, alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.show()


