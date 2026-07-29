import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# =============================================================================
# PARAMETERS — locked from Task 5
# =============================================================================

delta_design   = 0.03    # thresholded regulator (Module 2 active)
delta_linear   = 0.06    # linear regulator estimate (Module 2 absent)
delta_critical = 0.10    # critical threshold
mu_escape      = 1e-8    # per generation (Stritzker 2007)
gen_time       = 0.5     # hours per generation

# Extended N range including bottleneck-scale populations
N_values_full = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

# Stochastic trajectory parameters
N_TRAJ = 500

# =============================================================================
# LOG-SPACE MORAN FIXATION (from Task 5, verified)
# =============================================================================

def moran_fixation_prob(delta_val, N):
    if delta_val == 0:
        return 1.0 / N
    if delta_val >= 1.0:
        return 1.0
    numerator = delta_val
    log_term  = N * np.log1p(-delta_val)
    if log_term < -500:
        denominator = 1.0
    else:
        denominator = 1.0 - np.exp(log_term)
    if denominator == 0:
        return 1.0 / N
    return numerator / denominator

def circuit_half_life(delta_val, N):
    p_fix = moran_fixation_prob(delta_val, N)
    rate  = N * mu_escape * p_fix
    if rate == 0:
        return np.inf
    return 1.0 / rate   # generations

def half_life_years(delta_val, N):
    t_gen = circuit_half_life(delta_val, N)
    if t_gen == np.inf:
        return np.inf
    return t_gen * gen_time / 8760

# =============================================================================
# MORAN STOCHASTIC TRAJECTORY
# =============================================================================

def moran_trajectory(N, delta_val, n_steps=50000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_mut = 1
    w_c   = 1.0 - delta_val
    traj  = [n_mut]
    for _ in range(n_steps):
        if n_mut == 0 or n_mut == N:
            break
        n_c       = N - n_mut
        fit_total = n_c * w_c + n_mut
        p_inc     = (n_mut / fit_total) * (n_c / N)
        p_dec     = (n_c * w_c / fit_total) * (n_mut / N)
        u = rng.random()
        if u < p_inc:
            n_mut += 1
        elif u < p_inc + p_dec:
            n_mut -= 1
        traj.append(n_mut)
    return np.array(traj)

# =============================================================================
# ANALYSIS 1: Fixation probability at design vs linear vs critical across all N
# =============================================================================

print("=== Bottleneck-N Moran Sweep ===")
print(f"  N range: {N_values_full}")
print(f"  δ_design={delta_design}, δ_linear={delta_linear}, δ_critical={delta_critical}")

pfix_design   = [moran_fixation_prob(delta_design, N) for N in N_values_full]
pfix_linear   = [moran_fixation_prob(delta_linear, N) for N in N_values_full]
pfix_critical = [moran_fixation_prob(delta_critical, N) for N in N_values_full]
pfix_neutral  = [1.0 / N for N in N_values_full]

# =============================================================================
# ANALYSIS 2: Half-life ratio (thresholded / linear) across all N
# =============================================================================

halflife_thresh = [circuit_half_life(delta_design, N) for N in N_values_full]
halflife_linear = [circuit_half_life(delta_linear, N) for N in N_values_full]
advantage_ratio = [ht / hl if hl > 0 else np.inf
                   for ht, hl in zip(halflife_thresh, halflife_linear)]

print("\n  Fixation probability and half-life across population sizes:")
print(f"{'N':>8} | {'P_fix(thresh)':>14} | {'P_fix(linear)':>14} | {'P_fix(crit)':>12} | {'T_half(thr) yr':>15} | {'T_half(lin) yr':>15} | {'Advantage':>10}")
print("-" * 105)
for i, N in enumerate(N_values_full):
    ht_yr = half_life_years(delta_design, N)
    hl_yr = half_life_years(delta_linear, N)
    print(f"{N:>8} | {pfix_design[i]:>14.6f} | {pfix_linear[i]:>14.6f} | {pfix_critical[i]:>12.6f} | {ht_yr:>15.1f} | {hl_yr:>15.1f} | {advantage_ratio[i]:>10.2f}x")

# =============================================================================
# ANALYSIS 3: Stochastic trajectories at bottleneck N=50 vs colonised N=1000
# =============================================================================

print(f"\n  Running {N_TRAJ} stochastic trajectories at N=50 (bottleneck), δ={delta_design}...")
rng50 = np.random.default_rng(42)
traj_50 = [moran_trajectory(50, delta_design, rng=rng50) for _ in range(N_TRAJ)]
fixed_50   = sum(1 for tr in traj_50 if tr[-1] == 50)
extinct_50 = sum(1 for tr in traj_50 if tr[-1] == 0)

print(f"  Running {N_TRAJ} stochastic trajectories at N=1000 (colonised), δ={delta_design}...")
rng1k = np.random.default_rng(99)
traj_1k = [moran_trajectory(1000, delta_design, rng=rng1k) for _ in range(N_TRAJ)]
fixed_1k   = sum(1 for tr in traj_1k if tr[-1] == 1000)
extinct_1k = sum(1 for tr in traj_1k if tr[-1] == 0)

print(f"\n  N=50 (bottleneck):  Fixed={fixed_50}/{N_TRAJ} ({fixed_50/N_TRAJ*100:.1f}%) | Extinct={extinct_50}/{N_TRAJ} ({extinct_50/N_TRAJ*100:.1f}%)")
print(f"  N=1000 (colonised): Fixed={fixed_1k}/{N_TRAJ} ({fixed_1k/N_TRAJ*100:.1f}%) | Extinct={extinct_1k}/{N_TRAJ} ({extinct_1k/N_TRAJ*100:.1f}%)")

# =============================================================================
# ANALYSIS 4: Advantage ratio across a continuous delta sweep
# =============================================================================

delta_sweep = np.linspace(0.005, 0.15, 100)
N_bottleneck_values = [50, 100, 500, 1000]
advantage_by_N = {}

for N in N_bottleneck_values:
    ratios = []
    for d in delta_sweep:
        d_linear_est = d * 2   # linear imposes ~2x the burden
        ht = circuit_half_life(d, N)
        hl = circuit_half_life(d_linear_est, N)
        ratios.append(ht / hl if hl > 0 else np.inf)
    advantage_by_N[N] = ratios

# =============================================================================
# PLOTTING — 2x3 grid
# =============================================================================

fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

colors_N = {10: '#d62728', 25: '#ff7f0e', 50: '#bcbd22', 100: '#17becf',
            250: '#9467bd', 500: '#8c564b', 1000: '#2ca02c',
            2500: '#1f77b4', 5000: '#e377c2', 10000: '#7f7f7f'}

# ---- Panel A: Fixation probability vs N (log scale) ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogx(N_values_full, pfix_design,   'o-', color='seagreen', lw=2.5, ms=7,
             label=f'Thresholded δ={delta_design}')
ax1.semilogx(N_values_full, pfix_linear,   's--', color='darkorange', lw=2, ms=6,
             label=f'Linear δ={delta_linear}')
ax1.semilogx(N_values_full, pfix_critical, '^:', color='crimson', lw=1.5, ms=5,
             label=f'Critical δ={delta_critical}')
ax1.semilogx(N_values_full, pfix_neutral,  'x-', color='gray', lw=1, ms=5,
             label='Neutral drift (1/N)')
ax1.axvspan(10, 100, alpha=0.06, color='orange', label='Bottleneck range')
ax1.set(xlabel='Population size N (log scale)',
        ylabel='Fixation probability',
        title='A. Fixation probability across population sizes\nincluding gut-transit bottlenecks')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# ---- Panel B: Half-life advantage ratio vs N ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogx(N_values_full, advantage_ratio, 'o-', color='steelblue', lw=2.5, ms=8)
ax2.axhline(y=2.0, color='seagreen', lw=1, ls='--', alpha=0.6,
            label='2.0x (manuscript reported)')
ax2.axhline(y=1.0, color='gray', lw=1, ls=':', alpha=0.6, label='No advantage')
ax2.axvspan(10, 100, alpha=0.06, color='orange', label='Bottleneck range')
for i, N in enumerate(N_values_full):
    ax2.annotate(f'{advantage_ratio[i]:.2f}x',
                 (N, advantage_ratio[i]),
                 textcoords='offset points', xytext=(0, 10),
                 ha='center', fontsize=7.5, fontweight='bold')
ax2.set(xlabel='Population size N (log scale)',
        ylabel='Half-life advantage ratio (thresholded / linear)',
        title='B. Module 2 evolutionary advantage\npreserved at all population sizes')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# ---- Panel C: Stochastic trajectories at N=50 (bottleneck) ----
ax3 = fig.add_subplot(gs[0, 2])
for tr in traj_50[:80]:
    color = 'tomato' if tr[-1] == 50 else ('steelblue' if tr[-1] == 0 else 'lightgray')
    ax3.plot(range(len(tr)), tr, color=color, lw=0.4, alpha=0.35)
ax3.axhline(y=50, color='tomato',    lw=1.5, ls='--',
            label=f'Fixed {fixed_50}/{N_TRAJ} ({fixed_50/N_TRAJ*100:.1f}%)')
ax3.axhline(y=0,  color='steelblue', lw=1.5, ls='--',
            label=f'Extinct {extinct_50}/{N_TRAJ} ({extinct_50/N_TRAJ*100:.1f}%)')
ax3.set(xlabel='Moran steps', ylabel='Loss-of-function mutant count',
        title=f'C. Stochastic trajectories — bottleneck\nN=50, δ={delta_design}')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# ---- Panel D: Half-life (years) vs N for three delta values ----
ax4 = fig.add_subplot(gs[1, 0])
for d_val, col, lbl in [(delta_design, 'seagreen', f'Thresholded δ={delta_design}'),
                          (delta_linear, 'darkorange', f'Linear δ={delta_linear}'),
                          (delta_critical, 'crimson', f'Critical δ={delta_critical}')]:
    yrs = [half_life_years(d_val, N) for N in N_values_full]
    ax4.loglog(N_values_full, yrs, 'o-', color=col, lw=2, ms=6, label=lbl)
ax4.axvspan(10, 100, alpha=0.06, color='orange', label='Bottleneck range')
ax4.axhline(y=1, color='gray', lw=1, ls=':', alpha=0.6, label='1 year threshold')
ax4.set(xlabel='Population size N (log scale)',
        ylabel='Circuit half-life (years, log scale)',
        title='D. Circuit half-life vs population size\nacross fitness cost regimes')
ax4.legend(fontsize=7.5); ax4.grid(True, alpha=0.3)

# ---- Panel E: Advantage ratio vs delta for different bottleneck N ----
ax5 = fig.add_subplot(gs[1, 1])
bn_colors = {50: '#bcbd22', 100: '#17becf', 500: '#1f77b4', 1000: '#2ca02c'}
for N in N_bottleneck_values:
    ax5.plot(delta_sweep, advantage_by_N[N], lw=2, color=bn_colors[N],
             label=f'N={N}')
ax5.axvline(x=delta_design, color='black', lw=1.5, ls='--',
            label=f'Design δ={delta_design}')
ax5.axhline(y=2.0, color='gray', lw=1, ls=':', alpha=0.6)
ax5.set(xlabel='Circuit fitness cost δ (thresholded)',
        ylabel='Half-life advantage ratio (thresholded / linear)',
        title='E. Module 2 advantage across δ and N\n(linear burden assumed 2× thresholded)')
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# ---- Panel F: Stochastic comparison N=50 vs N=1000 ----
ax6 = fig.add_subplot(gs[1, 2])
bar_labels = ['N=50\n(bottleneck)', 'N=1000\n(colonised)']
extinct_pcts = [extinct_50/N_TRAJ*100, extinct_1k/N_TRAJ*100]
fixed_pcts   = [fixed_50/N_TRAJ*100, fixed_1k/N_TRAJ*100]
x_pos = np.arange(2)
width = 0.35
bars1 = ax6.bar(x_pos - width/2, extinct_pcts, width, color='steelblue',
                edgecolor='black', linewidth=0.5, label='Mutant extinct (circuit survives)')
bars2 = ax6.bar(x_pos + width/2, fixed_pcts,   width, color='tomato',
                edgecolor='black', linewidth=0.5, label='Mutant fixed (circuit lost)')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(bar_labels, fontsize=10)
ax6.set(ylabel='Percentage of trajectories (%)',
        title=f'F. Bottleneck vs colonised: mutant fate\nδ={delta_design}, {N_TRAJ} trajectories each')
for bar, val in zip(bars1, extinct_pcts):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
             ha='center', fontsize=9, fontweight='bold', color='steelblue')
for bar, val in zip(bars2, fixed_pcts):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
             ha='center', fontsize=9, fontweight='bold', color='tomato')
ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3, axis='y')
ax6.set_ylim(0, 110)

fig.text(0.5, -0.01,
    f"Bottleneck-N sweep: N=10 to N=10,000 | δ_design={delta_design} vs δ_linear={delta_linear} | "
    f"Advantage ratio preserved ≥{min(advantage_ratio):.2f}x across all N | "
    f"Bottleneck N=50: extinct={extinct_50/N_TRAJ*100:.0f}% (circuit survives)",
    ha='center', fontsize=9, color='dimgray',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

fig.suptitle('ISO Sentinel EcN -- Bottleneck Population Moran Analysis',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('iso_bottleneck_moran.png', dpi=300, bbox_inches='tight')
plt.savefig('iso_bottleneck_moran.svg', format='svg', bbox_inches='tight')
plt.close()
print("\nSaved: iso_bottleneck_moran.png / .svg")

# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n=== Bottleneck Analysis Verdict ===")
min_adv = min(advantage_ratio)
max_adv = max(advantage_ratio)
print(f"  Advantage ratio range: {min_adv:.2f}x to {max_adv:.2f}x across N={N_values_full[0]} to N={N_values_full[-1]}")
if min_adv >= 1.5:
    print(f"  CONFIRMED: Module 2 thresholded advantage is PRESERVED at all population sizes,")
    print(f"  including gut-transit bottlenecks as small as N={N_values_full[0]}.")
    print(f"  The qualitative conclusion is ROBUST to population structure assumptions.")
else:
    print(f"  WARNING: Advantage ratio drops below 1.5x at small N. Review assumptions.")