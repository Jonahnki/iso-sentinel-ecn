import tellurium as te
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# ÌṢỌ Sentinel EcN — Task 2: Full Four-Module ODE
# Module 1: Biosensor (TtrS/TtrR tetrathionate detection)
# Module 2: Regulator (thresholded Hill-function promoter)
# Module 3: Effector (MccH47 production + pathogen kill kinetics)
# Module 4: Containment (deltaDAPA escape probability)
# Parameters sourced from Palmer et al. 2017 where available
# =============================================================================

# =============================================================================
# PARAMETERS
# =============================================================================

# --- Module 1: Biosensor ---
k1          = 0.426   # TtrR production rate (J23115 + medium copy)
d_TtrR      = 0.1     # TtrR degradation
vm_sfGFP    = 0.6     # sfGFP max expression (B0031 + medium copy)
Km_sensor   = 0.3     # Hill constant for pTtr activation
n_sensor    = 2       # Hill cooperativity
d_sfGFP     = 0.05    # sfGFP degradation
k1_leak     = 0.01    # basal TtrR in absence of tetrathionate
EC50_ttr    = 10.0    # tetrathionate EC50 (uM) — Palmer 2017
n_ttr       = 2       # tetrathionate Hill coefficient

# --- Module 2: Regulator (thresholded promoter) ---
# Threshold suppresses effector below a TtrR* minimum
# Acts as a NOT gate on MccH47 expression at low TtrR*
threshold_TtrR = 1.5  # minimum TtrR* required to activate effector
n_reg          = 3    # sharpness of threshold switch
Km_reg         = threshold_TtrR

# --- Module 3: Effector (MccH47) ---
# Palmer 2017: MccH47 production ~0.1-0.3 uM/h under full induction
k_M         = 0.2     # MccH47 production rate (uM/h)
d_M         = 0.05    # MccH47 degradation
# Pathogen kill kinetics — Palmer 2017 Fig 3
# Kill rate constant: ~1e-3 per uM MccH47 per hour (bacteriostatic range)
k_kill      = 0.05    # pathogen kill rate (uM^-1 h^-1)
k_growth    = 0.5     # pathogen intrinsic growth rate (h^-1, ~35min doubling)
P0          = 1.0     # initial pathogen concentration (relative units)

# --- Module 4: Containment (deltaDAPA) ---
# Stritzker 2007: escape frequency ~1e-8 per generation
mu_escape   = 1e-8    # per-generation escape probability
gen_time    = 0.5     # hours per generation (EcN ~30min doubling)
# Burden on circuit-bearing cells
delta       = 0.05    # fitness cost of circuit (fraction of growth rate)

# --- Simulation parameters ---
T_END   = 200
N_PTS   = 2000
time    = np.linspace(0, T_END, N_PTS)

# =============================================================================
# MODULE 1+2+3: Full ODE system via solve_ivp
# State vector: [TtrR, sfGFP, MccH47, Pathogen]
# Tetrathionate concentration is an external input parameter
# =============================================================================

def iso_ode(t, y, S):
    """
    Full four-module ODE.
    S = tetrathionate concentration (uM)
    y = [TtrR, sfGFP, MccH47, Pathogen]
    """
    TtrR, sfGFP, MccH47, Pathogen = y
    TtrR    = max(TtrR, 0)
    sfGFP   = max(sfGFP, 0)
    MccH47  = max(MccH47, 0)
    Pathogen = max(Pathogen, 0)

    # Module 1: Sensor activation
    # TtrS senses S, phosphorylates TtrR — modelled as Hill scaling of k1
    k1_eff = k1 * (S**n_ttr) / (EC50_ttr**n_ttr + S**n_ttr) + k1_leak

    dTtrR = k1_eff - d_TtrR * TtrR

    # sfGFP reporter (for validation only — not part of effector path)
    dsfGFP = (vm_sfGFP * TtrR**n_sensor) / (Km_sensor**n_sensor + TtrR**n_sensor) \
             - d_sfGFP * sfGFP

    # Module 2: Regulator threshold
    # MccH47 only produced above TtrR threshold — suppresses leaky effector
    reg_gate = (TtrR**n_reg) / (Km_reg**n_reg + TtrR**n_reg)

    # Module 3: Effector
    dMccH47 = k_M * reg_gate - d_M * MccH47

    # Pathogen kill kinetics — logistic growth minus MccH47-mediated killing
    dPathogen = k_growth * Pathogen * (1 - Pathogen) - k_kill * MccH47 * Pathogen

    return [dTtrR, dsfGFP, dMccH47, dPathogen]

# --- Simulate homeostatic state (no tetrathionate) ---
y0 = [0.0, 0.0, 0.0, P0]
sol_home = solve_ivp(iso_ode, [0, T_END], y0, args=(0.0,),
                     t_eval=time, method='LSODA', rtol=1e-8, atol=1e-10)

# --- Simulate pathogen-present state (S = 50 uM tetrathionate) ---
sol_path = solve_ivp(iso_ode, [0, T_END], y0, args=(50.0,),
                     t_eval=time, method='LSODA', rtol=1e-8, atol=1e-10)

# --- Simulate threshold effect (S = 2 uM — sub-threshold) ---
sol_sub = solve_ivp(iso_ode, [0, T_END], y0, args=(2.0,),
                    t_eval=time, method='LSODA', rtol=1e-8, atol=1e-10)

# Unpack results
t = sol_path.t
TtrR_path  = sol_path.y[0]
sfGFP_path = sol_path.y[1]
MccH47_path = sol_path.y[2]
Path_path  = sol_path.y[3]

TtrR_home  = sol_home.y[0]
MccH47_home = sol_home.y[2]
Path_home  = sol_home.y[3]

TtrR_sub   = sol_sub.y[0]
MccH47_sub = sol_sub.y[2]
Path_sub   = sol_sub.y[3]

# --- Print key metrics ---
print("=== Four-Module Simulation Results ===")
print(f"\n[Homeostatic — no tetrathionate]")
print(f"  TtrR ss:    {TtrR_home[-1]:.4f}")
print(f"  MccH47 ss:  {MccH47_home[-1]:.4f}")
print(f"  Pathogen:   {Path_home[-1]:.4f} (should remain ~1.0)")

print(f"\n[Sub-threshold — 2 uM tetrathionate]")
print(f"  TtrR ss:    {TtrR_sub[-1]:.4f}")
print(f"  MccH47 ss:  {MccH47_sub[-1]:.4f}")
print(f"  Pathogen:   {Path_sub[-1]:.4f}")

print(f"\n[Pathogen-present — 50 uM tetrathionate]")
print(f"  TtrR ss:    {TtrR_path[-1]:.4f}")
print(f"  sfGFP ss:   {sfGFP_path[-1]:.4f}")
print(f"  MccH47 ss:  {MccH47_path[-1]:.4f}")
print(f"  Pathogen:   {Path_path[-1]:.6f} (suppression target: <0.01)")

suppression = (P0 - Path_path[-1]) / P0 * 100
print(f"  Suppression: {suppression:.1f}%")

# =============================================================================
# MODULE 4: Containment — deltaDAPA escape probability
# Moran process analytical solution (Nowak 2006)
# =============================================================================

def moran_fixation_prob(delta, N):
    """
    Analytical fixation probability of loss-of-function mutant
    in population of size N, with circuit fitness cost delta.
    Mutant fitness = 1, circuit fitness = 1 - delta
    """
    if delta == 0:
        return 1.0 / N
    r = 1.0 / (1.0 - delta)  # relative fitness of mutant
    return (1 - 1/r) / (1 - 1/r**N)

N_values    = [100, 1000, 10000]
delta_range = np.linspace(0, 0.3, 100)

fix_probs = {N: [moran_fixation_prob(d, N) for d in delta_range]
             for N in N_values}

# Stochastic Moran trajectories at delta=0.05 (our design point)
def moran_trajectory(N, delta, n_steps=10000):
    """Single stochastic Moran trajectory."""
    n_mutant = 1  # start with 1 loss-of-function mutant
    w_circuit = 1 - delta
    w_mutant  = 1.0
    trajectory = [n_mutant]
    for _ in range(n_steps):
        if n_mutant == 0 or n_mutant == N:
            break
        n_circuit = N - n_mutant
        fit_total = n_circuit * w_circuit + n_mutant * w_mutant
        p_birth_mutant = (n_mutant * w_mutant) / fit_total
        p_death_circuit = n_circuit / N
        p_death_mutant  = n_mutant / N
        # Birth of mutant, death of circuit
        p_increase = p_birth_mutant * p_death_circuit
        # Birth of circuit, death of mutant
        p_circuit_birth = (n_circuit * w_circuit) / fit_total
        p_decrease = p_circuit_birth * p_death_mutant
        r = np.random.random()
        if r < p_increase:
            n_mutant += 1
        elif r < p_increase + p_decrease:
            n_mutant -= 1
        trajectory.append(n_mutant)
    return trajectory

np.random.seed(42)
n_traj = 200
N_moran = 1000
trajectories = [moran_trajectory(N_moran, delta) for _ in range(n_traj)]
fixed   = sum(1 for tr in trajectories if tr[-1] == N_moran)
extinct = sum(1 for tr in trajectories if tr[-1] == 0)

print(f"\n=== Moran Process (N={N_moran}, delta={delta}) ===")
print(f"  Trajectories run:     {n_traj}")
print(f"  Mutant fixed:         {fixed}  ({fixed/n_traj*100:.1f}%)")
print(f"  Mutant extinct:       {extinct} ({extinct/n_traj*100:.1f}%)")
print(f"  Analytical fix prob:  {moran_fixation_prob(delta, N_moran):.4f}")

# Escape time estimation
escape_rate = mu_escape / gen_time   # escapes per hour
t_escape_50 = np.log(2) / escape_rate
print(f"\n=== Containment Escape Estimate ===")
print(f"  Escape rate:          {escape_rate:.2e} per hour")
print(f"  t50 escape:           {t_escape_50:.2e} hours ({t_escape_50/8760:.1f} years)")

# =============================================================================
# PLOTTING — 2x3 grid
# =============================================================================

fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# ---- Panel 1: Full module time course ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, TtrR_path,   color='steelblue',  linewidth=2, label='TtrR (50µM S)')
ax1.plot(t, sfGFP_path,  color='darkorange', linewidth=2, label='sfGFP reporter')
ax1.plot(t, MccH47_path, color='seagreen',   linewidth=2, label='MccH47 (effector)')
ax1.plot(t, TtrR_home,   color='steelblue',  linewidth=1.5, linestyle='--',
         alpha=0.5, label='TtrR (homeostatic)')
ax1.plot(t, MccH47_home, color='seagreen',   linewidth=1.5, linestyle='--',
         alpha=0.5, label='MccH47 (homeostatic)')
ax1.set_xlabel('Time (hours)', fontsize=10)
ax1.set_ylabel('Concentration (relative units)', fontsize=10)
ax1.set_title('Module 1+2+3 Time Course\nPathogen-present vs Homeostatic', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, T_END)
ax1.set_ylim(0)

# ---- Panel 2: Pathogen suppression ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(t, np.maximum(Path_path, 1e-10),  color='crimson',    linewidth=2,
             label='50µM tetrathionate')
ax2.semilogy(t, np.maximum(Path_sub, 1e-10),   color='darkorange', linewidth=2,
             linestyle='--', label='2µM (sub-threshold)')
ax2.semilogy(t, np.maximum(Path_home, 1e-10),  color='gray',       linewidth=2,
             linestyle=':', label='0µM (homeostatic)')
ax2.axhline(y=0.01, color='black', linewidth=1, linestyle='--', alpha=0.5,
            label='Target suppression (<1%)')
ax2.set_xlabel('Time (hours)', fontsize=10)
ax2.set_ylabel('Pathogen load (log scale)', fontsize=10)
ax2.set_title('Module 3: Pathogen Suppression\nMccH47 kill kinetics', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, T_END)

# ---- Panel 3: Regulator threshold effect ----
ax3 = fig.add_subplot(gs[0, 2])
S_sweep = np.logspace(-1, 2, 80)
MccH47_ss = []
TtrR_ss_sweep = []
for S in S_sweep:
    sol = solve_ivp(iso_ode, [0, 500], [0,0,0,P0], args=(S,),
                    t_eval=[500], method='LSODA', rtol=1e-8, atol=1e-10)
    TtrR_ss_sweep.append(sol.y[0][0])
    MccH47_ss.append(sol.y[2][0])

ax3.semilogx(S_sweep, TtrR_ss_sweep,  color='steelblue', linewidth=2, label='TtrR*')
ax3_r = ax3.twinx()
ax3_r.semilogx(S_sweep, MccH47_ss, color='seagreen', linewidth=2,
               linestyle='--', label='MccH47')
ax3.axvline(x=EC50_ttr, color='gray', linestyle='--', linewidth=1, alpha=0.7,
            label='EC50=10µM')
ax3.axvline(x=100, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax3.fill_betweenx([0, max(TtrR_ss_sweep)], 10, 100,
                  alpha=0.07, color='green')
ax3.set_xlabel('[Tetrathionate] (µM)', fontsize=10)
ax3.set_ylabel('TtrR* steady state', fontsize=10, color='steelblue')
ax3_r.set_ylabel('MccH47 steady state', fontsize=10, color='seagreen')
ax3.set_title('Module 2: Regulator Threshold\nTtrR* → MccH47 gating', fontsize=10)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_r.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax3.grid(True, alpha=0.3)

# ---- Panel 4: Moran fixation probability ----
ax4 = fig.add_subplot(gs[1, 0])
colors_N = {100: 'cornflowerblue', 1000: 'seagreen', 10000: 'tomato'}
for N in N_values:
    ax4.plot(delta_range, fix_probs[N], color=colors_N[N],
             linewidth=2, label=f'N={N}')
ax4.axvline(x=delta, color='black', linewidth=1.5, linestyle='--',
            label=f'Design point δ={delta}')
ax4.axvline(x=0.1, color='gray', linewidth=1, linestyle=':',
            label='δ=0.1 (critical threshold)')
ax4.set_xlabel('Circuit fitness cost (δ)', fontsize=10)
ax4.set_ylabel('Fixation probability', fontsize=10)
ax4.set_title('Module 4: Moran Fixation Probability\nLoss-of-function mutant', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 0.3)
ax4.set_ylim(0)

# ---- Panel 5: Stochastic Moran trajectories ----
ax5 = fig.add_subplot(gs[1, 1])
for tr in trajectories[:50]:
    color = 'tomato' if tr[-1] == N_moran else ('steelblue' if tr[-1] == 0 else 'lightgray')
    ax5.plot(range(len(tr)), tr, color=color, linewidth=0.5, alpha=0.4)
ax5.axhline(y=N_moran, color='tomato',    linewidth=1.5, linestyle='--',
            label=f'Fixation (n={fixed})')
ax5.axhline(y=0,       color='steelblue', linewidth=1.5, linestyle='--',
            label=f'Extinction (n={extinct})')
ax5.set_xlabel('Moran steps', fontsize=10)
ax5.set_ylabel('Mutant cell count', fontsize=10)
ax5.set_title(f'Module 4: Stochastic Moran Trajectories\nN={N_moran}, δ={delta}, n=200', fontsize=10)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ---- Panel 6: Containment escape over generations ----
ax6 = fig.add_subplot(gs[1, 2])
generations = np.logspace(0, 12, 200)
# Probability at least one escape in a population of 1e8 cells
N_cells = 1e8
p_no_escape  = (1 - mu_escape) ** (generations * N_cells / N_cells)
p_escape_cum = 1 - (1 - mu_escape) ** generations

ax6.semilogx(generations, p_escape_cum * 100, color='crimson',
             linewidth=2.5, label='Single ΔdapA')
# Dual auxotrophy: escape requires two independent events
p_dual = 1 - (1 - mu_escape**2) ** generations
ax6.semilogx(generations, p_dual * 100, color='steelblue',
             linewidth=2.5, linestyle='--', label='Dual auxotrophy (ΔdapA + ΔthyA)')
ax6.axvline(x=1e8, color='gray', linewidth=1, linestyle='--', alpha=0.7,
            label='~1e8 generations (est. gut)')
ax6.axhline(y=1,   color='orange', linewidth=1, linestyle=':', alpha=0.7,
            label='1% escape probability')
ax6.set_xlabel('Generations', fontsize=10)
ax6.set_ylabel('Cumulative escape probability (%)', fontsize=10)
ax6.set_title('Module 4: Containment Escape\nΔdapA vs dual auxotrophy', fontsize=10)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0, 105)

fig.suptitle('ÌṢỌ Sentinel EcN — Full Four-Module Characterisation (Task 2)',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('iso_four_module_sim.png', dpi=300, bbox_inches='tight')
plt.savefig('iso_four_module_sim.svg', format='svg', bbox_inches='tight')
plt.show()

print("\nFigures saved: iso_four_module_sim.png and iso_four_module_sim.svg")
