import numpy as np
import random
from qutip import *
import matplotlib.pyplot as plt
import time # for the random seed

# --- thermalize_system ---
def thermalize_system(rho_sys, H_sys, T,
                      gamma_th=1.0, t_th=5.0, nsteps=300,
                      return_result=False, atol=1e-9, rtol=1e-9):
    dim = H_sys.shape[0]
    if dim < 2:
        raise ValueError("H_sys must have dimension >= 2.")

    eigvals, eigkets = H_sys.eigenstates()
    order = np.argsort(np.real(eigvals))
    idx_g, idx_e = order[0], order[1]
    E_g = np.real(eigvals[idx_g])
    E_e = np.real(eigvals[idx_e])
    ket_g = eigkets[idx_g]
    ket_e = eigkets[idx_e]

    omega = E_e - E_g
    if omega <= 0:
        omega = 0.0

    if T is None or T <= 0 or omega == 0.0:
        n_th = 0.0
    else:
        expo = np.exp(omega / T)
        if np.isfinite(expo) and expo > 1.0:
            n_th = 1.0 / (expo - 1.0)
            if n_th < 0:
                n_th = 0.0
        else:
            n_th = 0.0

    gamma_down = gamma_th * (1.0 + n_th)
    gamma_up = gamma_th * n_th

    sigma_plus = ket_e * ket_g.dag()
    sigma_minus = ket_g * ket_e.dag()

    c_ops = []
    if gamma_down > 0.0:
        c_ops.append(np.sqrt(gamma_down) * sigma_minus)
    if gamma_up > 0.0:
        c_ops.append(np.sqrt(gamma_up) * sigma_plus)

    tlist = np.linspace(0.0, t_th, nsteps)
    opts = Options(atol=atol, rtol=rtol)
    result = mesolve(H_sys, rho_sys, tlist, c_ops=c_ops, e_ops=[], options=opts)

    rho_final = result.states[-1]
    Q = float(np.real((H_sys * (rho_final - rho_sys)).tr()))

    if return_result:
        return rho_final, Q, result
    else:
        return rho_final, Q


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

    return rho_sys_eig # Return the state in the instantaneous eigenbasis of H

# --- utility functions ---
def rho_eig_to_comp(rho_eig, H_sys_inst):
    eigvals, eigkets = H_sys_inst.eigenstates()
    ov0 = np.array([abs(np.vdot(basis(2, 0).full().ravel(), ket.full().ravel())) for ket in eigkets])
    order = np.argsort(-ov0)
    U_mat = np.hstack([eigkets[i].full() for i in order])
    U = Qobj(U_mat)
    rho_comp = U * rho_eig * U.dag()
    return rho_comp

def save_incremental_contributions_file(filename, details, n_trajectories, measurements, Gamma=None, tau_comp=None, tau_exp=None):
    """
    Write a human-readable log of the incremental work/heat contributions
    for both compression and expansion trajectories.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Incremental trajectory contributions\n")
        f.write(f"n_trajectories = {n_trajectories}\n")
        f.write(f"measurements = {measurements}\n")
        if Gamma is not None:
            f.write(f"Gamma = {Gamma}\n")
        if tau_comp is not None:
            f.write(f"tau_comp = {tau_comp}\n")
        if tau_exp is not None:
            f.write(f"tau_exp = {tau_exp}\n")
        f.write("\n")

        for i, pair in enumerate(details, start=1):
            f.write("=" * 80 + "\n")
            f.write(f"TRAJECTORY {i}\n")
            f.write("=" * 80 + "\n\n")

            for stroke_name in ("comp", "exp"):
                tr = pair[stroke_name]
                f.write(f"{stroke_name.upper()} trajectory {i}\n")
                f.write(f"outcomes: {tr['outcomes']}\n")
                f.write(f"p_seq: {tr['p_seq']}\n")
                f.write(f"DeltaW_total: {tr['DeltaW_total']}\n")
                f.write(f"DeltaQ_total: {tr['DeltaQ_total']}\n")
                f.write("dW per increment:\n")
                f.write(np.array2string(tr['deltaW_list'], separator=", ", precision=12))
                f.write("\n")
                f.write("dQ per increment:\n")
                f.write(np.array2string(tr['deltaQ_meas_list_steps'], separator=", ", precision=12))
                f.write("\n\n")

def change_to_eigenbasis(rho, H, order="eigenvalue", return_U=False):
    Hq = Qobj(H)
    rhoq = Qobj(rho)
    eigvals, eigkets = Hq.eigenstates()
    if order == "eigenvalue":
        order_idx = np.argsort(np.real(eigvals))
    elif order == "overlap":
        N = Hq.shape[0]
        comp0 = basis(N, 0).full().ravel()
        ov0 = np.array([abs(np.vdot(comp0, ket.full().ravel())) for ket in eigkets])
        order_idx = np.argsort(-ov0)
    else:
        raise ValueError("order must be 'eigenvalue' or 'overlap'")

    U_mat = np.hstack([eigkets[i].full() for i in order_idx])
    U = Qobj(U_mat)
    rho_eig = U.dag() * rhoq * U
    if return_U:
        return rho_eig, U
    else:
        return rho_eig


def ideal_otto_work(omega, Omega0, T_c, T_h, use_half=False):
    Omega_eff = np.sqrt(omega ** 2 + (Omega0 / 2.0) ** 2) if use_half else np.sqrt(omega ** 2 + Omega0 ** 2)
    dW = (Omega_eff - omega) / 2.0 * (np.tanh(Omega_eff / (2.0 * T_h)) - np.tanh(omega / (2.0 * T_c)))
    return Omega_eff, dW


def Qhot_ideal(omega, Omega, Tc, Th):
    t1 = np.tanh((omega / (2.0 * Tc)))
    t2 = np.tanh((Omega / (2.0 * Th)))
    return 0.5 * Omega * (t1 - t2)


def h_expectation(H, rho):
    val = (H * rho).tr()
    return float(np.real(val))


def l1_coherence(H, rho):
    evals, evecs = H.eigenstates()
    C = 0.0
    N = len(evecs)
    for i in range(N):
        for j in range(N):
            if i != j:
                val = (evecs[i].dag() * rho * evecs[j])
                C += abs(val)
    return float(C)


def gibbs_state(H, T=None, beta=None, eps=1e-14):
    if beta is None:
        if T is None:
            raise ValueError("Either T or beta must be provided.")
        if T <= 0:
            if T == 0:
                beta = np.inf
            else:
                raise ValueError("Temperature T must be >= 0 (T=0 allowed for ground state).")
        else:
            beta = 1.0 / T

    if np.isinf(beta) or beta > 1.0 / eps:
        eigvals, eigkets = H.eigenstates()
        idx_min = int(np.argmin(np.real(eigvals)))
        ket_g = eigkets[idx_min]
        rho_gs = ket_g * ket_g.dag()
        return rho_gs

    eigvals, eigkets = H.eigenstates()
    E = np.real(eigvals)
    boltz = np.exp(-beta * E)
    Z = np.sum(boltz)
    probs = boltz / Z
    rho = probs[0] * (eigkets[0] * eigkets[0].dag()) + probs[1] * (eigkets[1] * eigkets[1].dag())
    return rho


def H_joint_at(t, H_sys0, H_sys1, H_L, Omega_func, args, Gamma):
    """Return the joint Hamiltonian H_tot(t) used for energy bookkeeping.

    Same convention as in the paper: H_joint_base + Omega(t)*H_sys1_joint +
    coeff_Z * (sz × sx_lub) + coeff_X * (sx × sx_lub).
    """
    I_sys = qeye(2)
    I_lub = qeye(2)
    sx_sys = sigmax()
    sz_sys = sigmaz()
    sx_lub = sigmax()

    H_joint_base = tensor(H_sys0, I_lub) + tensor(I_sys, H_L)
    H_sys1_joint = tensor(H_sys1, I_lub)
    Z_sys_X_lub = tensor(sz_sys, sx_lub)
    X_sys_X_lub = tensor(sx_sys, sx_lub)

    Omega_val = float(Omega_func(t, args))
    omega_sys = args.get('omega', 1.0)
    theta = np.arctan2(2.0 * Omega_val, omega_sys)
    coeff_Z = float(Gamma * np.cos(theta))
    coeff_X = float(Gamma * np.sin(theta))

    H_tot = H_joint_base + Omega_val * H_sys1_joint + coeff_Z * Z_sys_X_lub + coeff_X * X_sys_X_lub
    return H_tot


# ----------------------------
# Single-trajectory sampler (unchanged)
# ----------------------------

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


def simulate_single_trajectory(
    H_sys0, H_sys1, H_L,
    Omega_func,
    rho_joint_init,
    t_initial, t_final,
    measurements=100,
    n_steps_per_interval=10,
    Gamma=100.0,
    selective=True,
    args=None,
    atol=1e-15, rtol=1e-15,
    rng=None
):
    if args is None:
        args = {}
    if rng is None:
        rng = random.Random()

    tau = t_final - t_initial
    if measurements < 1:
        measurements = 1
    dt = tau / float(measurements)
    
    #print("This is the dt:", dt)

    #####################
    # Z basis measurements
    #####################

    # # lubricant projectors
    # P0 = basis(2, 0) * basis(2, 0).dag()
    # P1 = basis(2, 1) * basis(2, 1).dag()
    # P_list = [P0, P1]

    #####################
    # X basis measurements
    #####################

    ket_plus  = (basis(2, 0) + basis(2, 1)).unit()
    ket_minus = (basis(2, 0) - basis(2, 1)).unit()

    Pp = ket_plus * ket_plus.dag()
    Pm = ket_minus * ket_minus.dag()

    P_list = [Pp, Pm]

    

    # time-dependent H for evolution (robust callback style + solver settings)
    H_joint_base = tensor(H_sys0, qeye(2)) + tensor(qeye(2), H_L)
    H_sys1_joint = tensor(H_sys1, qeye(2))
    Z_sys_X_lub = tensor(sigmaz(), sigmax())
    X_sys_X_lub = tensor(sigmax(), sigmax())

    # capture omega_sys and Gamma locally for the callbacks
    omega_sys_local = args.get('omega', 1.0)
    Gamma_local = float(Gamma)

    # explicit callbacks (use the same pattern as the working code)
    def Omega_cb(t, _args=None):
        return float(Omega_func(t, args))

    def coeff_Z_cb(t, _args=None):
        Omega_t = float(Omega_func(t, args))
        theta = np.arctan2(2.0 * Omega_t, omega_sys_local)
        return float(Gamma_local * np.cos(theta))

    def coeff_X_cb(t, _args=None):
        Omega_t = float(Omega_func(t, args))
        theta = np.arctan2(2.0 * Omega_t, omega_sys_local)
        return float(Gamma_local * np.sin(theta))

    outcomes = []
    deltaW_list = []
    deltaQ_meas_list = []
    times = []

    rho = rho_joint_init
    #print("This is the INITIAL state of the system,\n BEFORE the work stroke:\n", np.round(rho.ptrace(0).full(),4))
    p_seq = 1.0

    # initial measurement at t0
    # p0 = float(np.real((tensor(qeye(2), P0) * rho).tr())) # Z basis

    p0 = float(np.real((tensor(qeye(2), Pp) * rho).tr())) # X basis
    p0 = max(p0, 0.0)
    p1 = 1.0 - p0
    r0 = 0 if rng.random() < p0 else 1
    p_r0 = p0 if r0 == 0 else p1
    p_seq *= p_r0
    

    # if selective:
    #     if p_r0 == 0:
    #         rho = tensor(rho.ptrace(0), P0)
    #     else:
    #         Pj = tensor(qeye(2), P_list[r0])
    #         rho = (Pj * rho * Pj) / p_r0
    # else:
    #     rho = tensor(rho.ptrace(0), P0)

    outcomes.append(r0)
    # print(rho)
    #print(rho.ptrace(1))

    jump_tester = 0
    jump_tester_new = 0
    

    # iterate intervals
    for k in range(measurements):
        
        t0 = t_initial + k * dt
        t1 = t0 + dt
        times.append((t0, t1))
        # print("Time:", t1)

        H_list = [
            H_joint_base,
            [H_sys1_joint, Omega_cb],
            [Z_sys_X_lub, coeff_Z_cb],
            [X_sys_X_lub, coeff_X_cb]
        ]

        # Make solver able to resolve very fast dynamics from large Gamma:
        # increase nsteps and keep desired tolerances.
        opts = Options(atol=atol, rtol=rtol, nsteps=100000, store_states=True)


        tlist_interval = np.linspace(t0, t1, max(2, n_steps_per_interval))
        
        res = mesolve(H_list, rho, tlist_interval, c_ops=[], e_ops=[], args=args, options=opts)

        rho_post = res.states[-1]
        # print("---------------------------")
        # print("This is the state of the composite after a pulse (before meas): \n", np.round(rho_post.full(), 4))
        # print("---------------------------")
        # print("---------------------------")
        # print("Gamma:", Gamma)
        # print("This is the state after a pulse (before meas):\n", np.round(rho_post.ptrace(0).full(),4))
        # print("---------------------------")
        # H_sys_last = H_sys0 + Omega_cb(t1,args) * H_sys1
        # test_sys_state = project_to_inst_eigenbasis(H_sys_last,rho_post.ptrace(0))
        # print("---------------------------")
        # print("This is the INSTANTANEOUS state after a pulse (before meas):\n", test_sys_state)
        # print("---------------------------")
        # print("This is the LUBRICANT state after a pulse (before meas):\n", np.round(rho_post.ptrace(1).full(),4))
        # print("---------------------------")
        # log_neg = logarithmic_negativity(rho_post)
        # print("This is the entanglement after the pulse:\n", log_neg)
        # print("---------------------------")
        # print("---------------------------")
        # print("---------------------------")

        H_t0 = H_joint_at(t0, H_sys0, H_sys1, H_L, Omega_func, args, Gamma)
        H_t1 = H_joint_at(t1, H_sys0, H_sys1, H_L, Omega_func, args, Gamma)
        #print(H_t0)
        #print(H_t1)

        rho_pre = rho
        E_pre = float(np.real((H_t0 * rho_pre).tr()))
        E_post = float(np.real((H_t1 * rho_post).tr()))

        ##########################
        ################SYSTEM####
        # H_sys_pre = H_sys0 + Omega_cb(t0,args) * H_sys1
        # H_sys_post = H_sys0 + Omega_cb(t1,args) * H_sys1
        # E_pre_sys = float(np.real((H_sys_pre * rho_pre.ptrace(0)).tr()))
        # E_post_sys = float(np.real((H_sys_post * rho_post.ptrace(0)).tr()))
        ##########################

        deltaW = E_post - E_pre
        # print("Time:",t1+t0)
        # print("This is the work generated:", deltaW)
        #print("This is the work system only:", float(E_post_sys-E_pre_sys))

        # measurement at t1 (selective Z measurement)
        # p0 = float(np.real((tensor(qeye(2), P0) * rho_post).tr()))

        # measurement at t1 (selective X measurement)
        p0 = float(np.real((tensor(qeye(2), Pp) * rho_post).tr()))


        #print("--------------------------------------------------------------")
        #print("This is the probability of measuring and observing 0:", p0)
        #print("State after the pulse:\n", rho_post.ptrace(1))
        #p0 = np.clip(p0, 0.0, 1.0)
        p1 = 1.0 - p0
        # print(rho_post.ptrace(1))
        if selective:
            r = 0 if rng.random() < p0 else 1
            p_r = p0 if r == 0 else p1
            p_seq *= p_r
            if r == 0:
                # print("We observed 0")
                # rho_meas = (tensor(qeye(2), P0)* rho_post * tensor(qeye(2), P0)) / p0 # Z basis projectorss
                rho_meas = (tensor(qeye(2), Pp)* rho_post * tensor(qeye(2), Pp)) / p0 # X basis projectorss
                jump_tester_new = 0
            else:
                #print("We observed 1")
                # rho_meas = (tensor(qeye(2), P1)* rho_post * tensor(qeye(2), P1)) / p1 # Z basis projects
                rho_meas = (tensor(qeye(2), Pm)* rho_post * tensor(qeye(2), Pm)) / p1   # X basis projects
                jump_tester_new = 1
            if jump_tester != jump_tester_new:
                print(f"THERE WAS A JUMP from {jump_tester} to {jump_tester_new}!")
                jump_tester = jump_tester_new
                #print(rho_meas.ptrace(1))
        else:
            r = None
            rho_meas = tensor(rho_post.ptrace(0), qeye(2))
        #print("--------------------------------------------------------------")
        

        deltaQ_meas = float(np.real((H_t1 * (rho_post - rho_meas)).tr()))
        # print("This is the heat dissipated:", np.sum(deltaQ_meas))
        

        deltaW_list.append(deltaW)
        deltaQ_meas_list.append(deltaQ_meas)
        outcomes.append(r)

        rho = rho_meas
        # print("--------------------------------------------------------------")
        # print("This is the state after a pulse (after meas):\n", np.round(rho.full(),4))
        # print("This is the system state: \n", np.round(rho.ptrace(0).full(),4))
        # print("This is the lubricant state: \n", np.round(rho.ptrace(0).full(),4))
        # distance_between_the_two = tracedist(rho,tensor(rho.ptrace(0),rho.ptrace(1)))
        # print("This is the trace distance to the product state: \n", distance_between_the_two)
        # print("This is the trace plus state: \n", Pp)
        # print("--------------------------------------------------------------")



    joint_rho_final = rho
    #print("This is the final state after the work stroke:\n", np.round(rho.full(),4))
    #print("This is the final state of the system,\n after the work stroke:\n", np.round(rho.ptrace(0).full(),4))
    #print("This is the final state of the lubricant,\n after the work stroke:\n", np.round(rho.ptrace(1).full(),4))
    H_sys_last = H_sys0 + Omega_cb(t1,args) * H_sys1
    test_sys_state = project_to_inst_eigenbasis(H_sys_last,rho_post.ptrace(0))
    #print("---------------------------")
    #print("This is the INSTANTANEOUS final state of the system:\n", test_sys_state)
    #print("---------------------------")
    sys_rho_final = joint_rho_final.ptrace(0)
    DeltaW_total = float(np.sum(np.array(deltaW_list)))
    DeltaQ_total = float(np.sum(np.array(deltaQ_meas_list)))
    # DeltaW_total = float(np.sum(np.array(deltaW_list)))
    # print("This is the list of dissipated heat:\n", deltaQ_meas_list)
    # print("This is the sum of dissipated heat:", float(np.sum(deltaQ_meas_list)))

    #print("That is the sum of all the generated work:", float(np.sum(np.array(deltaW_list))))
    #print("---------------------------------------------------")
    #print("That is the sum of all the dissipated heat:", float(np.sum(np.array(deltaQ_meas_list))))
    #print("That is the average of all the dissipated heat:", float(np.mean(np.array(deltaQ_meas_list))))
    #print(outcomes)
    #print("That is the probability for that specific trajectory:", p_seq)
    #print("That is the work contribution:", DeltaW_total)
    # print("FLUCTUATING WORK contribution:", float(p_seq * DeltaW_total))
    # print("FLUCTUATING Dissipation contribution:", float(p_seq * np.sum(np.array(deltaQ_meas_list))))
    #print("---------------------------------------------------")
    

    return {
        'outcomes': tuple(outcomes),
        'p_seq': float(p_seq),
        'DeltaW_total': DeltaW_total,
        'DeltaQ_total': DeltaQ_total,
        'deltaW_list': np.array(deltaW_list, dtype=float),
        'deltaQ_meas_list': np.sum(deltaQ_meas_list),
        'deltaQ_meas_list_steps': np.array(deltaQ_meas_list, dtype=float),
        'joint_rho_final': joint_rho_final,
        'sys_rho_final': sys_rho_final,
        'times': times,
        'fluctuating_work': float(p_seq * DeltaW_total),
        'fluctuating_dissipation': float(p_seq * np.sum(np.array(deltaQ_meas_list)))
    }


# ----------------------------
# NEW: simulate many sampled trajectories for a *pair* of drives
# ----------------------------
def simulate_sampled_pair_drives(
    n_trajectories,
    H_sys0, H_sys1, H_L,
    Omega_comp, Omega_exp,
    rho_joint_init_comp,
    rho_joint_init_exp,
    t_comp, t_exp,
    measurements_comp=100,
    measurements_exp=100,
    n_steps_per_interval_comp=10,
    n_steps_per_interval_exp=10,
    Gamma=100.0,
    selective=True,
    comp_args=None,
    exp_args=None,
    seed=None,
):
    """
    Run n_trajectories sampled trajectories where for each trajectory we:
      - run compression drive via simulate_single_trajectory with Omega_comp on [0, t_comp]
      - optionally (reset_lub_for_expansion) reinitialize lubricant before expansion
      - run expansion drive via simulate_single_trajectory with Omega_exp on [0, t_exp]
    Returns arrays and summary statistics for compression and expansion separately and their Zeno-weighted sum.
    Important: this function keeps your existing per-trajectory work computation intact.
    """
    if comp_args is None:
        comp_args = {}
    if exp_args is None:
        exp_args = {}
    rng = random.Random(seed)

    DeltaW_comp_list = []
    DeltaQ_comp_list = []
    p_comp_list = []
    DeltaW_exp_list = []
    DeltaQ_exp_list = []
    p_exp_list = []
    fluctuating_work_comp = []
    fluctuating_heat_comp = []
    fluctuating_work_exp = []
    fluctuating_heat_exp = []
    
    details = []

    for i in range(n_trajectories):
        # compression stroke
        print("COMPRESSION RUNNING")
        print(f"This is the trajectory {i}")
        traj_comp = simulate_single_trajectory(
            H_sys0, H_sys1, H_L,
            Omega_comp,
            rho_joint_init_comp,
            0.0, t_comp,
            measurements=measurements_comp,
            n_steps_per_interval=n_steps_per_interval_comp,
            Gamma=Gamma,
            selective=selective,
            args=comp_args,
            rng=rng
        )
        DeltaW_comp_list.append(traj_comp['DeltaW_total'])
        DeltaQ_comp_list.append(traj_comp['DeltaQ_total'])
        p_comp_list.append(traj_comp['p_seq'])
        fluctuating_work_comp.append(traj_comp['fluctuating_work'])
        fluctuating_heat_comp.append(traj_comp['fluctuating_dissipation'])

        # expansion stroke (note: different Omega and args)
        print("EXPANSION RUNNING")
        traj_exp = simulate_single_trajectory(
            H_sys0, H_sys1, H_L,
            Omega_exp,
            rho_joint_init_exp,
            0.0, t_exp,
            measurements=measurements_exp,
            n_steps_per_interval=n_steps_per_interval_exp,
            Gamma=Gamma,
            selective=selective,
            args=exp_args,
            rng=rng
        )
        DeltaW_exp_list.append(traj_exp['DeltaW_total'])
        DeltaQ_exp_list.append(traj_exp['DeltaQ_total'])
        fluctuating_work_exp.append(traj_exp['fluctuating_work'])
        fluctuating_heat_exp.append(traj_exp['fluctuating_dissipation'])
        
        p_exp_list.append(traj_exp['p_seq'])

        details.append({'comp': traj_comp, 'exp': traj_exp})

    
    # print("Expansion works:", DeltaW_exp_list)
    # print("Compression works:", DeltaW_comp_list)
    # print("Expansion Dissipation", DeltaQ_exp_list)
    # print("Compression Dissipation", DeltaQ_comp_list)
    DeltaW_comp_arr = np.array(DeltaW_comp_list, dtype=float)
    #print("Sum of work contributions for each trajectory in the compression:", DeltaW_comp_arr)
    p_comp_arr = np.array(p_comp_list, dtype=float)
    # print("Table of probabilities for the compression stroke:\n",p_comp_arr)

    DeltaW_exp_arr = np.array(DeltaW_exp_list, dtype=float)
    #print("Sum of work contributions for each trajectory in the expansion:", DeltaW_exp_arr)
    p_exp_arr = np.array(p_exp_list, dtype=float)
    # print("Table of probabilities for the expansion stroke:\n", p_exp_arr)

    # Zeno-averaged (weighted) mean for each stroke (match your earlier definition)
    # weighted_mean_comp = float(np.sum(p_comp_arr * DeltaW_comp_arr) / np.sum(p_comp_arr)) if np.sum(p_comp_arr) > 0 else np.nan
    weighted_mean_comp = float(np.sum(p_comp_arr * DeltaW_comp_arr))
    average_comp = np.mean(DeltaW_comp_arr)
    
    
    
    # weighted_mean_exp = float(np.sum(p_exp_arr * DeltaW_exp_arr) / np.sum(p_exp_arr)) if np.sum(p_exp_arr) > 0 else np.nan
    weighted_mean_exp = float(np.sum(p_exp_arr * DeltaW_exp_arr)) 
    average_exp = np.mean(DeltaW_exp_arr)

    DeltaQ_comp_arr = np.array(DeltaQ_comp_list, dtype=float)
    DeltaQ_exp_arr = np.array(DeltaQ_exp_list, dtype=float)

    sum_dissipation_comp = np.sum(DeltaQ_comp_arr)
    sum_dissipation_exp = np.sum(DeltaQ_exp_arr)
    average_dissipation_comp = np.mean(DeltaQ_comp_arr)
    # print(DeltaQ_comp_arr)
    average_dissipation_exp = np.mean(DeltaQ_exp_arr)
    # print(DeltaQ_exp_arr)

    average_dissipation_comp = float(np.mean(DeltaQ_comp_arr))
    average_dissipation_exp = float(np.mean(DeltaQ_exp_arr))
    Q_meas_total = average_dissipation_comp + average_dissipation_exp
    
    
    
    
    

    print("AVERAGED OPTIONS")
    print("This is the average compression sum:", average_comp)
    print("This is the average expansion sum:", average_exp)
    print("This is the average energy extracted:", average_comp + average_exp)
    print("This is the average dissipation compression:", average_dissipation_comp)
    print("This is the average dissipation expansion:", average_dissipation_exp)

    # print("ZENO OPTIONS")
    # print("Average Zeno work for the compression:", float(np.sum(fluctuating_work_comp)))
    # print("Average Zeno work for the expansion:", float(np.sum(fluctuating_work_exp)))
    # print("This is the average Zeno total work", float(np.sum(fluctuating_work_comp)+np.sum(fluctuating_work_exp)))
    # print("This is the dissipation contribution:", float(np.sum(fluctuating_heat_comp)+np.sum(fluctuating_heat_exp)))
    return {
        'n_trajectories': n_trajectories,
        'DeltaW_comp': DeltaW_comp_arr,
        'p_comp': p_comp_arr,
        'weighted_mean_comp': weighted_mean_comp,
        'DeltaW_exp': DeltaW_exp_arr,
        'p_exp': p_exp_arr,
        'weighted_mean_exp': weighted_mean_exp,
        'weighted_mean_total': (average_comp + average_exp),
        'average_dissipation_comp': average_dissipation_comp,
        'average_dissipation_exp': average_dissipation_exp,
        'Q_meas_total': Q_meas_total,
        'details': details
    }


# ----------------------------
# DEMO: sweep over tau and plot ΔW^(Zeno)_comp + ΔW^(Zeno)_exp vs tau
# ----------------------------
if __name__ == "__main__":
    # --- small demo configuration (tune these for your full runs) ---
    omega = 1.0
    Omega0 = 3.01105
    T_cold = 0.5
    T_hot = 3.0
    gamma_th = 0.5

    # Hamiltonians
    H_sys0 = (omega / 2.0) * sigmaz()
    H_sys1 = sigmax()
    H_L = (1.0 / 2.0) * sigmaz()  # lubricant Hamiltonian (omega_lub/2 * sz)

    # initial joint state for the compression
    rho_sys_init = gibbs_state(H_sys0, T=T_cold)
    # rho_lub_init = basis(2, 0) * basis(2, 0).dag() # Initial lubricant in |0><0|

    ket_plus  = (basis(2, 0) + basis(2, 1)).unit()
    ket_minus = (basis(2, 0) - basis(2, 1)).unit()

    rho_lub_init = ket_plus * ket_plus.dag()   # Initial lubricant in |+><+|
    rho_joint_init = tensor(rho_sys_init, rho_lub_init)

    # define two different drives (compression and expansion)
    # note: these receive (t, args) and you can pass different args dictionaries
    def Omega_comp(t, args):
        # linear ramp from 0 to Omega0 over args['tau']
        tau_local = args['tau']
        return (args['Omega0'] / 2 ) * (t / tau_local)

    def Omega_exp(t, args):
        # linear ramp from Omega0 to 0 over args['tau'] (reverse)
        tau_local = args['tau']
        return (args['Omega0'] / 2) * (1.0 - (t / tau_local))


    # sampling settings (keep small for quick demo)

    for generating_data in np.arange(0, 1000, 20):

        # tau values 
        taus = np.arange(5.0, 10.0, 0.05)  # try [5,10,...,30] fast demo
        zeno_total_vs_tau = []
        qmeas_vs_tau = []

        n_trajectories = 100
        measurements = 3200
        n_steps_per_interval = 20000  # fine time resolution inside each interval
        Gamma = 20 + generating_data
        selective = True
        
        count = int(time.time())

        for tau in taus:
            count+=1
            seed = count
            print("Running tau =", tau)
            comp_args = {'Omega0': Omega0, 'tau': tau, 'omega': omega}
            # choose expansion duration; in your Otto code you used tau/2
            tau_exp = tau / 2.0
            exp_args = {'Omega0': Omega0, 'tau': tau_exp, 'omega': omega}



            ###################################################
            # Preparing the expansion initial state
            ###################################################
            # instantaneous system Hamiltonian at end of compression (t = tau)
            H_sys_inst_end_comp = H_sys0 + Omega_comp(tau, {'Omega0': Omega0, 'tau': tau, 'omega': omega}) * H_sys1

            # Gibbs state of the system at the hot temperature but using the instantaneous Hamiltonian
            rho_sys_init_exp = gibbs_state(H_sys_inst_end_comp, T=T_hot)

            # lubricant initial (same as before)

            #####################
            # X basis version
            #####################

            ket_plus  = (basis(2, 0) + basis(2, 1)).unit()
            ket_minus = (basis(2, 0) - basis(2, 1)).unit()

            Pp = ket_plus * ket_plus.dag()
            Pm = ket_minus * ket_minus.dag()
            # rho_lub_init = basis(2, 0) * basis(2, 0).dag()
            rho_lub_init = Pp
            # joint initial for expansion: system Gibbs (w.r.t. H_sys_inst_end_comp) ⊗ lubricant
            rho_joint_init_exp = tensor(rho_sys_init_exp, rho_lub_init)

            #print("This is the state entering the expansion stroke:", rho_joint_init_exp)
            ####################################################################

            res_pair = simulate_sampled_pair_drives(
                n_trajectories=n_trajectories,
                H_sys0=H_sys0, H_sys1=H_sys1, H_L=H_L,
                Omega_comp=Omega_comp, Omega_exp=Omega_exp,
                rho_joint_init_comp=rho_joint_init,
                rho_joint_init_exp=rho_joint_init_exp,
                t_comp=tau, t_exp=tau_exp,
                measurements_comp=measurements, measurements_exp=measurements,
                n_steps_per_interval_comp=max(2, int(n_steps_per_interval / max(1, measurements))),
                n_steps_per_interval_exp=max(2, int(n_steps_per_interval / max(1, measurements))),
                Gamma=Gamma,
                selective=selective,
                comp_args=comp_args, exp_args=exp_args,
                seed=seed
            )

            tau_exp = tau / 2.0
            tau_total = tau + tau_exp

            qmeas_vs_tau.append(res_pair['Q_meas_total'])

            zeno_total_vs_tau.append(res_pair['weighted_mean_total'])
            #print("tau:", tau, "weighted_total (comp+exp):", res_pair['weighted_mean_total'])

            detailed_filename = (
                f"Incremental_work_contributions_Gamma_{Gamma:.1f}"
                f"_tau_{tau:.3f}_trajectories_{n_trajectories}"
                f"_measurements_{measurements}.txt"
            )
            save_incremental_contributions_file(
                detailed_filename,
                res_pair['details'],
                n_trajectories=n_trajectories,
                measurements=measurements,
                Gamma=Gamma,
                tau_comp=tau,
                tau_exp=tau_exp
            )



        zeno_total_vs_tau = np.array(zeno_total_vs_tau, dtype=float)

        qmeas_vs_tau = np.array(qmeas_vs_tau, dtype=float)
        tau_total_vs_tau = 1.5 * taus

        np.savetxt(
            f'Otto_lubricated_Fluctuating_Gamma_{Gamma:.1f}_measurements_{int(measurements)}_trajectories_{n_trajectories}.txt',
            np.column_stack((tau_total_vs_tau, zeno_total_vs_tau, qmeas_vs_tau)),
            header='tau_total W_out Q_meas'
        )
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(taus, zeno_total_vs_tau, marker='o', linestyle='-')
        plt.axhline(-0.30107, linestyle='--', color='black', linewidth=2,
                label=r'Otto work = -0.30107')
        plt.xlabel(r'$\tau$ (compression time)')
        plt.ylabel(r'$\Delta W^{\rm(Zeno)}_{\rm comp} + \Delta W^{\rm(Zeno)}_{\rm exp}$')
        plt.title('Zeno-averaged fluctuating work (comp + exp) vs tau')
        plt.grid(True)
        plt.tight_layout()
        plt.show()