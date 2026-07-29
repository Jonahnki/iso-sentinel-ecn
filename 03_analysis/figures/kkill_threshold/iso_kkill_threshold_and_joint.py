import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# =============================================================================
# ISO Sentinel EcN — k_kill threshold localisation and joint fragility mapping
# =============================================================================

# --- Parameters, locked ---
k1, k1_leak, d_TtrR, n_ttr, EC50_ttr = 0.426, 0.002, 0.1, 2, 20.0
n_reg, Km_reg, d_M, k_growth = 3, 2.0, 0.05, 0.5
copy_number, delta_per_copy = 20, 0.0015
delta_design = copy_number * delta_per_copy
mu_escape, gen_time = 1e-8, 0.5
P0, T_END, S_ON = 1.0, 200, 50.0
k_M_design = 0.2

# =============================================================================
# PART A — k_kill threshold via root-finding
# =============================================================================

def suppression_at(k_kill_val, k_M_val=k_M_design, delta_val=delta_design):
    def ode(t, y):
        TtrR, MccH47, Pathogen = [max(v, 0) for v in y]
        k1_eff   = k1 * (S_ON**n_ttr / (EC50_ttr**n_ttr + S_ON**n_ttr)) + k1_leak
        dTtrR    = k1_eff - d_TtrR * TtrR
        reg_gate = TtrR**n_reg / (Km_reg**n_reg + TtrR**n_reg)
        dMccH47  = k_M_val * reg_gate - d_M * MccH47
        dPath    = k_growth * Pathogen * (1 - Pathogen) - k_kill_val * MccH47 * Pathogen
        return [dTtrR, dMccH47, dPath]
    sol = solve_ivp(ode, [0, T_END], [0, 0, P0], t_eval=[T_END],
                    method='LSODA', rtol=1e-8, atol=1e-10)
    return max(0, (P0 - sol.y[2][0]) / P0 * 100)

print("=== k_kill threshold localisation ===")

def suppression_minus_90(k_kill_val):
    return suppression_at(k_kill_val) - 90.0

k_kill_threshold = brentq(suppression_minus_90, 0.08, 0.20, xtol=1e-5)
print(f"  k_kill* (90% crossing) = {k_kill_threshold:.5f} uM^-1 h^-1")
print(f"  Manuscript value 0.30 gives a {0.30/k_kill_threshold:.2f}x margin above threshold")

k_kill_fine = np.linspace(0.05, 0.30, 200)
supp_fine   = [suppression_at(kk) for kk in k_kill_fine]

def supp_minus(target):
    return lambda kk: suppression_at(kk) - target

k_kill_50 = brentq(supp_minus(50), 0.05, 0.20, xtol=1e-5)
k_kill_99 = brentq(supp_minus(99), 0.05, 0.20, xtol=1e-5)
print(f"  k_kill for 50% suppression: {k_kill_50:.5f}")
print(f"  k_kill for 99% suppression: {k_kill_99:.5f}")

# =============================================================================
# PART B — joint efficacy / containment sweep
# =============================================================================

print("\n=== Joint efficacy-containment sweep ===")

def moran_fixation_prob(delta_val, N):
    if delta_val == 0: return 1.0 / N
    if delta_val >= 1.0: return 1.0
    numerator = delta_val
    log_term  = N * np.log1p(-delta_val)
    denominator = 1.0 if log_term < -500 else 1.0 - np.exp(log_term)
    return numerator / denominator if denominator != 0 else 1.0/N

k_kill_joint = np.linspace(0.05, 0.30, 30)
N_joint      = [50, 100, 250, 500, 1000]

# Single shared colour map for N across every panel in this figure
colors_N = {50: '#bcbd22', 100: '#17becf', 250: '#9467bd', 500: '#8c564b', 1000: '#2ca02c'}

joint_supp = {N: [suppression_at(kk) for kk in k_kill_joint] for N in N_joint}
fix_at_N   = {N: moran_fixation_prob(delta_design, N) for N in N_joint}

print("  Suppression does not depend on N in this model.")
print("  Fixation probability by N (design delta=0.03):")
for N in N_joint:
    print(f"    N={N}: P_fix={fix_at_N[N]:.4f}")

fragility_by_N = {}
for N in N_joint:
    supp_norm = 1 - (np.array(joint_supp[N]) / 100)
    fix_norm  = fix_at_N[N] / max(fix_at_N.values())
    fragility_by_N[N] = np.maximum(supp_norm, fix_norm)

# =============================================================================
# FIGURE — threshold localisation
# =============================================================================

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes1[0]
ax.plot(k_kill_fine, supp_fine, color='steelblue', lw=2.5)
ax.axhline(y=90, color='black', lw=1.5, ls='--', label='90% suppression target')
ax.axvline(x=k_kill_threshold, color='crimson', lw=2, ls='-',
           label=f'k_kill* = {k_kill_threshold:.4f} (threshold)')
ax.axvline(x=0.05, color='gray', lw=1, ls=':', label='Palmer 2017 in vitro (0.05)')
ax.axvline(x=0.30, color='seagreen', lw=1.5, ls='--', label='Manuscript value (0.30)')
ax.axvspan(k_kill_threshold, 0.30, alpha=0.08, color='seagreen',
           label=f'Margin ({0.30/k_kill_threshold:.1f}x)')
ax.plot(k_kill_threshold, 90, 'o', color='crimson', ms=10, zorder=5)
ax.set(xlabel='k_kill (uM^-1 h^-1)', ylabel='Suppression at design point (%)',
       title='A. Suppression threshold')
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(True, alpha=0.3)

ax2 = axes1[1]
milestones = [(k_kill_50, '50%'), (k_kill_threshold, '90%'), (k_kill_99, '99%')]
labels = [m[1] for m in milestones]
values = [m[0] for m in milestones]
bars = ax2.bar(labels, values, color=['#ff7f0e', '#d62728', '#2ca02c'],
               edgecolor='black', linewidth=0.5)
ax2.axhline(y=0.05, color='gray', lw=1, ls=':', label='Palmer 2017 in vitro')
ax2.axhline(y=0.30, color='seagreen', lw=1.5, ls='--', label='Manuscript value')
for bar, v in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}',
             ha='center', fontsize=10, fontweight='bold')
ax2.set(xlabel='Suppression milestone', ylabel='Required k_kill (uM^-1 h^-1)',
        title='B. k_kill required per milestone')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

fig1.suptitle('ISO Sentinel EcN -- k_kill Threshold Localisation', fontsize=13, fontweight='bold')
fig1.tight_layout()
plt.savefig('iso_kkill_threshold.png', dpi=300, bbox_inches='tight')
plt.savefig('iso_kkill_threshold.svg', format='svg', bbox_inches='tight')
plt.close()
print("\nSaved: iso_kkill_threshold.png / .svg")

# =============================================================================
# FIGURE — joint fragility, consistent colour mapping across all three panels
# =============================================================================

fig2 = plt.figure(figsize=(16, 6))
gs2  = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)

axA = fig2.add_subplot(gs2[0])
for N in N_joint:
    axA.plot(k_kill_joint, joint_supp[N], color=colors_N[N], lw=2, label=f'N={N}')
axA.axhline(y=90, color='black', lw=1, ls='--', alpha=0.6)
axA.axvline(x=k_kill_threshold, color='crimson', lw=1.5, ls=':', label='k_kill* threshold')
axA.set(xlabel='k_kill (uM^-1 h^-1)', ylabel='Suppression (%)',
        title='A. Efficacy across k_kill,\nby population size N')
axA.legend(fontsize=8)
axA.grid(True, alpha=0.3)

axB = fig2.add_subplot(gs2[1])
Ns_arr = N_joint  # explicit, matches axA and axC ordering exactly
fix_vals = [fix_at_N[N] for N in Ns_arr]
bar_colors = [colors_N[N] for N in Ns_arr]
bars_B = axB.bar(range(len(Ns_arr)), fix_vals, color=bar_colors,
                  edgecolor='black', linewidth=0.5)
axB.set_xticks(range(len(Ns_arr)))
axB.set_xticklabels([str(N) for N in Ns_arr])
for i, v in enumerate(fix_vals):
    axB.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
# Explicit legend matching axA/axC colour-to-N mapping
handles_B = [plt.Rectangle((0,0),1,1, color=colors_N[N]) for N in Ns_arr]
axB.legend(handles_B, [f'N={N}' for N in Ns_arr], fontsize=7.5, loc='upper right')
axB.set(xlabel='Population size N', ylabel='Fixation probability (delta=0.03)',
        title='B. Fixation probability by\npopulation size N')
axB.grid(True, alpha=0.3, axis='y')

axC = fig2.add_subplot(gs2[2])
for N in N_joint:
    axC.plot(k_kill_joint, fragility_by_N[N], color=colors_N[N], lw=2, label=f'N={N}')
axC.axvline(x=k_kill_threshold, color='crimson', lw=1.5, ls=':', alpha=0.7)
axC.axvline(x=0.30, color='seagreen', lw=1.5, ls='--', alpha=0.7, label='Manuscript value')
axC.set(xlabel='k_kill (uM^-1 h^-1)', ylabel='Combined fragility score (0=robust, 1=fragile)',
        title='C. Combined fragility across\nk_kill and N')
axC.legend(fontsize=8)
axC.grid(True, alpha=0.3)

fig2.suptitle('ISO Sentinel EcN -- Joint Efficacy-Containment Fragility Map',
              fontsize=13, fontweight='bold', y=1.03)
plt.savefig('iso_joint_fragility.png', dpi=300, bbox_inches='tight')
plt.savefig('iso_joint_fragility.svg', format='svg', bbox_inches='tight')
plt.close()
print("Saved: iso_joint_fragility.png / .svg")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n=== Summary ===")
print(f"  Suppression threshold: k_kill={k_kill_threshold:.4f}")
print(f"  Fixation probability: N=1000 -> {fix_at_N[1000]:.4f}, N=50 -> {fix_at_N[50]:.4f}")
print(f"  Ratio: {fix_at_N[50]/fix_at_N[1000]:.1f}x")
print(f"  Suppression has no N-dependence in this model; fixation probability does.")
print(f"  A population bottleneck therefore raises the probability that a")
print(f"  loss-of-function mutant fixes, without directly changing kill efficacy")
print(f"  while the circuit remains intact.")
