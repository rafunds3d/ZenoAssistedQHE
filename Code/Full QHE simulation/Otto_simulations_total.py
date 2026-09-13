import numpy as np
from qutip import *
import random
import matplotlib.pyplot as plt


# ----------------------------
# Thermalize a 2-level system to a bath at temperature T
# ----------------------------
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


# ------------------------------------------------------------------
# utility functions
# ------------------------------------------------------------------

def rho_eig_to_comp(rho_eig, H_sys_inst):
    eigvals, eigkets = H_sys_inst.eigenstates()
    ov0 = np.array([abs(np.vdot(basis(2, 0).full().ravel(), ket.full().ravel())) for ket in eigkets])
    order = np.argsort(-ov0)
    U_mat = np.hstack([eigkets[i].full() for i in order])
    U = Qobj(U_mat)
    rho_comp = U * rho_eig * U.dag()
    return rho_comp


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


# ------------------------------------------------------------------
# Zeno-drive simulator (joint SL)
# ------------------------------------------------------------------
def simulate_zeno_drive_joint(
    H_sys0, H_sys1, H_L,
    Omega_func,
    rho_initial_joint,
    t_initial, t_final,
    Gamma=1000.0,
    measurements=1000,
    n_steps_per_interval=50,
    selective=True,
    args=None,
    atol=1e-15, rtol=1e-15
):
    if args is None:
        args = {}

    tau = t_final - t_initial
    if measurements < 1:
        measurements = 1
    dt = tau / float(measurements)

    P0 = basis(2, 0) * basis(2, 0).dag()
    P1 = basis(2, 1) * basis(2, 1).dag()

    I_sys = qeye(2)
    I_lub = qeye(2)

    H_joint_base = tensor(H_sys0, I_lub) + tensor(I_sys, H_L)
    H_sys1_joint = tensor(H_sys1, I_lub)

    Z_sys_X_lub = tensor(sigmaz(), sigmax())
    X_sys_X_lub = tensor(sigmax(), sigmax())

    omega_sys = args.get('omega', 1.0)

    def Omega_cb(t, _args=None):
        return float(Omega_func(t, args))

    def coeff_Z_cb(t, _args=None):
        Omega_t = float(Omega_func(t, args))
        theta = np.arctan2(2.0 * Omega_t, omega_sys)
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
    H_joint_inst_list = []
    joint_states_list = []
    joint_times = []

    rho = rho_initial_joint

    for k in range(measurements):
        t0 = t_initial + k * dt
        t1 = t0 + dt
        tlist_interval = np.linspace(t0, t1, max(2, n_steps_per_interval))

        H_list = [
            H_joint_base,
            [H_sys1_joint, Omega_cb],
            [Z_sys_X_lub, coeff_Z_cb],
            [X_sys_X_lub, coeff_X_cb]
        ]

        opts = Options(atol=atol, rtol=rtol, nsteps=100000, store_states=True)

        res = mesolve(H_list, rho, list(tlist_interval), c_ops=[], e_ops=[], args=args, options=opts)

        states_interval = res.states
        rho = states_interval[-1]

        # record joint state & time
        joint_states_list.append(rho)
        joint_times.append(t1)

        # instantaneous system Hamiltonian and record
        H_sys_inst = H_sys0 + Omega_func(t1, args) * H_sys1
        H_sys_inst_list.append(H_sys_inst)

        # compute scalars and build joint H exactly as used by solver
        Omega_val = float(Omega_func(t1, args))
        omega_sys_local = args.get('omega', 1.0)
        theta = np.arctan2(2.0 * Omega_val, omega_sys_local)
        coeff_Z_val = float(Gamma * np.cos(theta))
        coeff_X_val = float(Gamma * np.sin(theta))

        H_joint_inst = (H_joint_base
                        + Omega_val * H_sys1_joint
                        + coeff_Z_val * Z_sys_X_lub
                        + coeff_X_val * X_sys_X_lub)

        H_joint_inst_list.append(H_joint_inst)

        rho_sys = rho.ptrace(0)

        eigvals, eigkets = H_sys_inst.eigenstates()
        ov0 = np.array([abs(np.vdot(basis(2, 0).full().ravel(), ket.full().ravel())) for ket in eigkets])
        order = np.argsort(-ov0)
        U_mat = np.hstack([eigkets[i].full() for i in order])
        U = Qobj(U_mat)
        rho_sys_eig = U.dag() * rho_sys * U
        coherences_before.append(2 * abs(rho_sys_eig.full()[0, 1]))

        tlist_all.append(t0)
        states_all_before.append(rho_sys_eig)

        P0_joint = tensor(I_sys, P0)
        P1_joint = tensor(I_sys, P1)
        if measurements != 1:
            if selective:
                p0 = float(np.real((P0_joint * rho).tr()))
                if p0 < 0:
                    p0 = -p0
                p1 = 1 - p0
                r = '0' if random.random() < p0 else '1'
                if r == '0':
                    rho = (P0_joint * rho * P0_joint) / p0
                else:
                    rho = (P1_joint * rho * P1_joint) / p1

                rho_sys = rho.ptrace(0)
                coherences_after.append(l1_coherence(H_sys_inst, rho_sys))
                energies.append(h_expectation(H_sys_inst, rho_sys))
                states_all_after.append(rho_sys)
            else:
                rho = P0_joint * rho * P0_joint + P1_joint * rho * P1_joint

    joint_state_rho = rho

    rho_sys = rho.ptrace(0)
    eigvals, eigkets = H_sys_inst.eigenstates()
    ov0 = np.array([abs(np.vdot(basis(2, 0).full().ravel(), ket.full().ravel())) for ket in eigkets])
    order = np.argsort(-ov0)
    U_mat = np.hstack([eigkets[i].full() for i in order])
    U = Qobj(U_mat)
    rho_sys_eig = U.dag() * rho_sys * U

    return {
        'tlist': np.array(tlist_all),
        'states_before': states_all_before,
        'states_after': states_all_after,
        'H_sys_inst': H_sys_inst_list,
        'expect_energy': np.array(energies),
        'coherences_after': np.array(coherences_after),
        'coherences_before': np.array(coherences_before),
        'rho_final': rho_sys_eig,
        'joint_rho_final': joint_state_rho,
        'H_joint_inst': H_joint_inst_list,
        'joint_states': joint_states_list,
        'joint_times': np.array(joint_times),
        'coherences': np.array(coherences)
    }


# ----------------------------
# Build an Otto cycle that calls simulate_zeno_drive_joint for work strokes
# ----------------------------
def run_otto_cycle(
    tau_comp,
    tau_exp,
    tau_hot,
    tau_cold,
    Omega0, omega,
    T_cold, T_hot,
    Gamma, measurements,
    gamma_th=1.0,
    n_steps_drive=1000,
    selective=True
):
    # Hamiltonians
    H_sys0 = (omega / 2.0) * sigmaz()
    H_sys1 = sigmax()
    omega_lub = 1
    H_L = (omega_lub / 2.0) * sigmaz()

    # initial states
    rho_sys_init = gibbs_state(H_sys0, T=T_cold)
    
    ket_plus  = (basis(2, 0) + basis(2, 1)).unit()
    ket_minus = (basis(2, 0) - basis(2, 1)).unit()

    Pp = ket_plus * ket_plus.dag()      # |+><+|
    Pm = ket_minus * ket_minus.dag()    # |-><-|

    rho_lub_init = Pp # basis(2, 0) * basis(2, 0).dag() # maximally mixed 
    rho_joint_init = tensor(rho_sys_init, rho_lub_init)

    # Omega schedules
    def Omega_comp(t, args):
        return args['Omega0'] * (t / (2.0 * args['tau']))

    def Omega_exp(t, args):
        return args['Omega0'] * ((1.0 - t / args['tau']) / 2.0)

    comp_args = {'Omega0': Omega0, 'tau': tau_comp, 'omega': omega}
    exp_args = {'Omega0': Omega0, 'tau': tau_exp, 'omega': omega}

    # 1) Compression work stroke
    sim_comp = simulate_zeno_drive_joint(
        H_sys0, H_sys1, H_L,
        Omega_comp,
        rho_joint_init,
        0.0, tau_comp,
        Gamma=Gamma,
        measurements=measurements,
        n_steps_per_interval=max(2, int(n_steps_drive / max(1, measurements))),
        selective=selective,
        args=comp_args
    )

    H_sys_inst_comp = sim_comp['H_sys_inst'][-1]
    rho_sys_after_comp_eig = sim_comp['rho_final']

    # Joint operators (build here as well)
    I_sys = qeye(2)
    I_lub = qeye(2)
    H_joint_base = tensor(H_sys0, I_lub) + tensor(I_sys, H_L)
    H_sys1_joint = tensor(H_sys1, I_lub)
    Z_sys_X_lub = tensor(sigmaz(), sigmax())
    X_sys_X_lub = tensor(sigmax(), sigmax())

    H_joint_FINAL_comp = sim_comp['H_joint_inst'][-1]
    rho_joint_FINAL_compression = sim_comp['joint_rho_final']
    # print("Final state after compression strokes:\n", np.round(rho_joint_FINAL_compression.full(),4))

    # Initial joint H at start of compression (t = 0)
    Omega_start_comp = float(Omega_comp(0.0, comp_args))
    theta_start_comp = np.arctan2(2.0 * Omega_start_comp, comp_args.get('omega', omega))
    coeffZ_start_comp = float(Gamma * np.cos(theta_start_comp))
    coeffX_start_comp = float(Gamma * np.sin(theta_start_comp))
    H_joint_INITIAL_comp = (H_joint_base
                            + Omega_start_comp * H_sys1_joint
                            + coeffZ_start_comp * Z_sys_X_lub
                            + coeffX_start_comp * X_sys_X_lub)

    rho_joint_INITIAL_compression = rho_joint_init

    # print("Initial state of the compression strokes:\n", np.round(rho_joint_INITIAL_compression.full(),4))

    E_before_JOINT_comp = float(np.real((H_joint_INITIAL_comp * rho_joint_INITIAL_compression).tr()))
    E_after_JOINT_comp = float(np.real((H_joint_FINAL_comp * rho_joint_FINAL_compression).tr()))
    W_JOINT_comp = E_after_JOINT_comp - E_before_JOINT_comp
    # print(E_after_JOINT_comp)
    # print("JOINT Compression BEFORE:", E_before_JOINT_comp)
    # print("JOINT Compression AFTER:", E_after_JOINT_comp)
    # print("First JOINT term Compression BEFORE:", float(np.real((rho_joint_INITIAL_compression * (coeffZ_start_comp * Z_sys_X_lub + coeffX_start_comp * X_sys_X_lub)).tr())))
    print("Compression:", W_JOINT_comp)    

    # Dissipation energy due to tracing out the lubricant system
    # Dissipation_comp = float(np.real((H_joint_FINAL_comp * (rho_joint_FINAL_compression)).tr() ))
    # print(rho_joint_FINAL_compression.ptrace(1))
    # Convert reduced state back to computational basis
    rho_sys_after_comp = rho_eig_to_comp(rho_sys_after_comp_eig, H_sys_inst_comp)

    # System energy bookkeeping (compression)
    H_sys_inst_start = H_sys0 + Omega_comp(0, comp_args) * H_sys1
    E_before_comp = float(np.real((H_sys_inst_start * rho_sys_init).tr()))
    E_after_comp = float(np.real((H_sys_inst_comp * rho_sys_after_comp).tr()))
    W_comp = E_after_comp - E_before_comp
    TEST_coupling_term = float(np.real((rho_joint_FINAL_compression * (H_joint_FINAL_comp - tensor(H_sys_inst_comp,qeye(2)) - (omega_lub/2) *tensor(qeye(2),sigmaz()) )).tr()))
    # print("First JOINT term Compression AFTER:", TEST_coupling_term)
    # print("SYS Compression BEFORE:", E_before_comp)
    # print("SYS Compression AFTER:", E_after_comp)
    # print("JOINT Compression AFTER Theory:", float(E_after_comp+0.5 * np.real((sigmaz() * rho_joint_FINAL_compression.ptrace(1)).tr())))
    TEST_Z_term = 0.5 * np.real((sigmaz() * rho_joint_FINAL_compression.ptrace(1)).tr())
    # print("JOINT Compression AFTER New Theory:", float(TEST_coupling_term + E_after_comp + TEST_Z_term))
    
    # print("JOINT Compression BEFORE Theory:", float(E_before_comp+0.5))
    # print("JOINT Compression work:", W_JOINT_comp)
    # print("JOINT Compression work THEORY:", float(E_after_comp+0.5 * np.real((sigmaz() * rho_joint_FINAL_compression.ptrace(1)).tr()) - E_before_comp - 0.5))
    # print("Term compression:", np.real((sigmaz() * rho_joint_FINAL_compression.ptrace(1)).tr()))
    # print(rho_joint_FINAL_compression.ptrace(1))

    # 2) Hot isochore (thermalization)
    rho_sys_before_hot = rho_sys_after_comp
    rho_sys_after_hot, Q_hot = thermalize_system(rho_sys_before_hot, H_sys_inst_comp, T_hot,
                                                 gamma_th=gamma_th, t_th=tau_hot)

    # 3) Expansion stroke
    rho_joint_exp_init = tensor(rho_sys_after_hot, rho_lub_init)

    print(np.round(rho_joint_exp_init.full(),5))

    sim_exp = simulate_zeno_drive_joint(
        H_sys0, H_sys1, H_L,
        Omega_exp,
        rho_joint_exp_init,
        0.0, tau_exp,
        Gamma=Gamma,
        measurements=measurements,
        n_steps_per_interval=max(2, int(n_steps_drive / max(1, measurements))),
        selective=selective,
        args=exp_args
    )

    H_sys_inst_after_exp = sim_exp['H_sys_inst'][-1]
    rho_sys_after_exp_eig = sim_exp['rho_final']
    rho_sys_after_exp = rho_eig_to_comp(rho_sys_after_exp_eig, H_sys_inst_after_exp)

    # Expansion joint energetics
    Omega_start_exp = float(Omega_exp(0.0, exp_args))
    theta_start_exp = np.arctan2(2.0 * Omega_start_exp, exp_args.get('omega', omega))
    coeffZ_start_exp = float(Gamma * np.cos(theta_start_exp))
    coeffX_start_exp = float(Gamma * np.sin(theta_start_exp))
    H_joint_INITIAL_exp = (H_joint_base
                           + Omega_start_exp * H_sys1_joint
                           + coeffZ_start_exp * Z_sys_X_lub
                           + coeffX_start_exp * X_sys_X_lub)

    rho_joint_exp_init = tensor(rho_sys_after_hot, rho_lub_init)

    H_joint_FINAL_exp = sim_exp['H_joint_inst'][-1]
    rho_joint_FINAL_expansion = sim_exp['joint_rho_final']

    E_before_JOINT_exp = float(np.real((H_joint_INITIAL_exp * rho_joint_exp_init).tr()))
    E_after_JOINT_exp = float(np.real((H_joint_FINAL_exp * rho_joint_FINAL_expansion).tr()))
    W_JOINT_exp = E_after_JOINT_exp - E_before_JOINT_exp
    print("Expansion:", W_JOINT_exp)
    # print("Energy BEFORE JOINT: ", E_before_JOINT_exp)
    # print("Energy AFTER JOINT: ", E_after_JOINT_exp)
    

    # System expansion energetics
    E_before_exp = float(np.real((H_sys_inst_comp * rho_sys_after_hot).tr()))
    E_after_exp = float(np.real((H_sys_inst_after_exp * rho_sys_after_exp).tr()))
    W_exp = E_after_exp - E_before_exp
    # print("Energy BEFORE SYS: ", E_before_exp)
    # print("Energy AFTER SYS: ", E_after_exp)
    # print("Energy BEFORE JOINT Theory expansion:", float(E_before_exp + 0.5))
    # print("Energy AFTER JOINT Theory expansion:", float(E_after_exp + 0.5* np.real((sigmaz()*rho_joint_FINAL_expansion.ptrace(1)).tr())))
    # print("Term expansion:", float(np.real((sigmaz()*rho_joint_FINAL_expansion.ptrace(1)).tr())))
    # 4) Cold isochore
    rho_sys_before_cold = rho_sys_after_exp
    rho_sys_after_cold, Q_cold = thermalize_system(rho_sys_before_cold, H_sys_inst_after_exp, T_cold,
                                                   gamma_th=gamma_th, t_th=tau_cold)

    # totals & signs (system-level)
    W_net = W_comp + W_exp
    Q_in = Q_hot
    W_out = -W_net

    # joint totals & signs (joint-level)
    W_net_joint = W_JOINT_comp + W_JOINT_exp
    W_out_joint = -W_net_joint

    # cycle time
    total_cycle_time = tau_comp + tau_hot + tau_exp + tau_cold

    # efficiencies and powers
    eta = (W_out / Q_in) if abs(Q_in) > 1e-16 else np.nan
    power = (W_out / total_cycle_time) if total_cycle_time > 0 else np.nan

    eta_joint = (W_out_joint / Q_in) if abs(Q_in) > 1e-16 else np.nan
    power_joint = (W_out_joint / total_cycle_time) if total_cycle_time > 0 else np.nan

    # optional joint time-series (from recorded joint states)
    comp_joint_E_list = []
    comp_joint_tlist = sim_comp.get('joint_times', np.array([]))
    if 'H_joint_inst' in sim_comp and 'joint_states' in sim_comp:
        for Hj, rj in zip(sim_comp['H_joint_inst'], sim_comp['joint_states']):
            comp_joint_E_list.append(float(np.real((Hj * rj).tr())))
        comp_joint_E_list = np.array(comp_joint_E_list, dtype=float)
    else:
        comp_joint_E_list = np.array([])

    exp_joint_E_list = []
    exp_joint_tlist = sim_exp.get('joint_times', np.array([]))
    if 'H_joint_inst' in sim_exp and 'joint_states' in sim_exp:
        for Hj, rj in zip(sim_exp['H_joint_inst'], sim_exp['joint_states']):
            exp_joint_E_list.append(float(np.real((Hj * rj).tr())))
        exp_joint_E_list = np.array(exp_joint_E_list, dtype=float)
    else:
        exp_joint_E_list = np.array([])

    joint_results = {
        'H_joint_initial_comp': H_joint_INITIAL_comp,
        'H_joint_final_comp': H_joint_FINAL_comp,
        'rho_joint_initial_comp': rho_joint_INITIAL_compression,
        'rho_joint_final_comp': rho_joint_FINAL_compression,
        'E_before_comp_joint': E_before_JOINT_comp,
        'E_after_comp_joint': E_after_JOINT_comp,
        'W_comp_joint': W_JOINT_comp,
        'H_joint_initial_exp': H_joint_INITIAL_exp,
        'H_joint_final_exp': H_joint_FINAL_exp,
        'rho_joint_initial_exp': rho_joint_exp_init,
        'rho_joint_final_exp': rho_joint_FINAL_expansion,
        'E_before_exp_joint': E_before_JOINT_exp,
        'E_after_exp_joint': E_after_JOINT_exp,
        'W_exp_joint': W_JOINT_exp,
        'W_net_joint': W_net_joint,
        'W_out_joint': W_out_joint,
        'eta_joint': eta_joint,
        'power_joint': power_joint,
        'comp_joint_t': comp_joint_tlist,
        'comp_joint_E': comp_joint_E_list,
        'exp_joint_t': exp_joint_tlist,
        'exp_joint_E': exp_joint_E_list
    }

    results = {
        'W_comp': W_comp,
        'W_exp': W_exp,
        'W_net': W_net,
        'W_out': W_out,
        'Q_hot': Q_hot,
        'Q_cold': Q_cold,
        'eta': eta,
        'power': power,
        'total_time': total_cycle_time,
        'rho_final': rho_sys_after_cold,
        'energies': {
            'E_before_comp': E_before_comp,
            'E_after_comp': E_after_comp,
            'E_before_exp': E_before_exp,
            'E_after_exp': E_after_exp
        },
        'joint': joint_results
    }

    return results


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    precision = 0.05
    total_engine_compression_time = 50
    x = np.arange(5, total_engine_compression_time + precision, precision)

    # engine parameters
    omega = 1
    Omega0 = 3.01105
    T_cold = 0.5
    T_hot = 3

    # thermalization parameters
    gamma_th = 0.5
    tau_hot = 5
    tau_cold = 12

    # simulation drive detail
    n_steps_drive = 1000
    measurements_varying = 0

    for gamma_varying in np.arange(20, 1000, 20):
        scenarios = {
            'non-lubricated': {'Gamma': 0.0, 'measurements': 0},
            'lubricated': {'Gamma': gamma_varying, 'measurements': measurements_varying}
        }

        outputs = {name: {'eta': [], 'power': [], 'W_out': [], 'Q_hot': [],
                          'eta_joint': [], 'power_joint': [], 'W_out_joint': []}
                   for name in scenarios.keys()}

        for name, params in scenarios.items():
            print(f"Running scenario: {name}")
            for tau in x:
                print("tau", tau)
                res = run_otto_cycle(
                    tau_comp=tau,
                    tau_exp=tau / 2,
                    tau_hot=tau_hot,
                    tau_cold=tau_cold,
                    Omega0=Omega0, omega=omega,
                    T_cold=T_cold, T_hot=T_hot,
                    Gamma=params['Gamma'], measurements=params['measurements'],
                    gamma_th=gamma_th,
                    n_steps_drive=n_steps_drive,
                    selective=True
                )

                # append scalar outputs (system-level)
                outputs[name]['eta'].append(res['eta'])
                outputs[name]['power'].append(res['power'])
                outputs[name]['W_out'].append(res['W_out'])
                outputs[name]['Q_hot'].append(res['Q_hot'])

                # append joint-level scalars
                joint = res.get('joint', {})
                outputs[name]['eta_joint'].append(joint.get('eta_joint', np.nan))
                outputs[name]['power_joint'].append(joint.get('power_joint', np.nan))
                outputs[name]['W_out_joint'].append(joint.get('W_out_joint', np.nan))

                # # --- safely extract joint data and save per-run ---
                # comp_t = np.asarray(joint.get('comp_joint_t', np.array([])))
                # comp_E = np.asarray(joint.get('comp_joint_E', np.array([])))
                # exp_t = np.asarray(joint.get('exp_joint_t', np.array([])))
                # exp_E = np.asarray(joint.get('exp_joint_E', np.array([])))

                # fname = f'otto_joint_Gamma_{gamma_varying:.3f}_scenario_{name}_tau_{tau:.3f}.npz'
                # np.savez(fname,
                #          E_before_comp_joint=joint.get('E_before_comp_joint', np.nan),
                #          E_after_comp_joint=joint.get('E_after_comp_joint', np.nan),
                #          W_comp_joint=joint.get('W_comp_joint', np.nan),
                #          E_before_exp_joint=joint.get('E_before_exp_joint', np.nan),
                #          E_after_exp_joint=joint.get('E_after_exp_joint', np.nan),
                #          W_exp_joint=joint.get('W_exp_joint', np.nan),
                #          W_net_joint=joint.get('W_net_joint', np.nan),
                #          W_out_joint=joint.get('W_out_joint', np.nan),
                #          eta_joint=joint.get('eta_joint', np.nan),
                #          power_joint=joint.get('power_joint', np.nan),
                #          comp_t=comp_t, comp_E=comp_E,
                #          exp_t=exp_t, exp_E=exp_E)

        # Convert to numpy arrays
        for name in outputs:
            for k in outputs[name]:
                outputs[name][k] = np.array(outputs[name][k], dtype=float)

        # Save scalar outputs for this Gamma sweep (example)
        np.savetxt(f'Join_Evolution_X_basis_{gamma_varying:.3f}_measurements_{int(measurements_varying)}.txt',
                   np.column_stack((x, outputs['lubricated']['eta'], outputs['lubricated']['eta_joint'],
                                    outputs['lubricated']['W_out'], outputs['lubricated']['W_out_joint'],
                                    outputs['lubricated']['power'], outputs['lubricated']['power_joint'])),
                   header='tau eta eta_joint W_out W_out_joint power power_joint')

        # compute reference Otto efficiency
        Omega_ref = np.sqrt(omega ** 2 + Omega0 ** 2)
        eta_otto = 1.0 - (omega / Omega_ref) if Omega_ref != 0 else np.nan
        eta_line = np.full_like(x, eta_otto, dtype=float)

        # print ideal numbers
        Omega_eff, dW = ideal_otto_work(omega=omega, Omega0=Omega0, T_c=T_cold, T_h=T_hot, use_half=False)
        dQ = Qhot_ideal(omega, Omega_ref, T_cold, T_hot)
        print("Omega_eff", Omega_eff)
        print("Ideal Otto ΔW (negative -> extracted):", dW)
        print("Ideal Otto Qhot:", dQ)

        # # Plot efficiencies (system-level)
        # plt.figure(figsize=(8, 4))
        # plt.plot(x, outputs['non-lubricated']['eta'], marker='o', label='Non-lubricated')
        # plt.plot(x, outputs['lubricated']['eta'], marker='^', label='Lubricated')
        # plt.plot(x, eta_line, '--', color='k', label=f'Otto eff: $1-\\omega/\\Omega$ = {eta_otto:.3f}')
        # plt.xlabel('tau')
        # plt.ylabel('efficiency (eta)')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # # Plot power (system-level)
        # plt.figure(figsize=(8, 4))
        # plt.plot(x, outputs['non-lubricated']['power'], marker='o', label='Non-lubricated')
        # plt.plot(x, outputs['lubricated']['power'], marker='^', label='Lubricated')
        # plt.xlabel('tau')
        # plt.ylabel('power')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # # Plot work (system-level)
        # plt.figure(figsize=(8, 4))
        # plt.plot(x, outputs['non-lubricated']['W_out'], marker='o', label='Non-lubricated')
        # plt.plot(x, outputs['lubricated']['W_out'], marker='^', label='Lubricated')
        # plt.xlabel('tau')
        # plt.ylabel('W_out')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # # --- NEW: Plot joint efficiencies (joint-level)
        # plt.figure(figsize=(8, 4))
        # plt.plot(x, outputs['non-lubricated']['eta_joint'], marker='o', label='Non-lubricated (joint)')
        # plt.plot(x, outputs['lubricated']['eta_joint'], marker='^', label='Lubricated (joint)')
        # plt.plot(x, eta_line, '--', color='k', label=f'Otto eff (sys ref): {eta_otto:.3f}')
        # plt.xlabel('tau')
        # plt.ylabel('efficiency (joint eta)')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # # --- NEW: Plot joint power (joint-level)
        # plt.figure(figsize=(8, 4))
        # plt.plot(x, outputs['non-lubricated']['power_joint'], marker='o', label='Non-lubricated (joint)')
        # plt.plot(x, outputs['lubricated']['power_joint'], marker='^', label='Lubricated (joint)')
        # plt.xlabel('tau')
        # plt.ylabel('power (joint)')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # # --- NEW: Plot joint work (system-level)
        # plt.figure(figsize=(8, 4))
        # # plt.plot(x, outputs['non-lubricated']['W_out_joint'], marker='o', label='Non-lubricated')
        # plt.plot(x, outputs['lubricated']['W_out_joint'], marker='^', label='Lubricated (joint)')
        # plt.plot(x, outputs['non-lubricated']['W_out'], marker='o', label='Non-lubricated (sys)')
        # plt.plot(x, outputs['lubricated']['W_out'], marker='x', label='Lubricated (sys)')
        # plt.xlabel('tau')
        # plt.ylabel('W_out')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()