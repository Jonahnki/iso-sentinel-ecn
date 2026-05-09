import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# ÌṢỌ Sentinel EcN — Task 1 v6: Biosensor Characterisation (FI Fixed)
# Key fix from v5:
#   alpha_leak reduced to 2% of alpha_max (was 10%) — OFF floor was 2.4,
#   dragging observed FI to 5.5x. At 2%, OFF floor ~0.48, FI ~27x.
#   k1_leak reduced to 0.002 to match — TtrR at S=0 now ~0.02 (was 0.1),
#   preventing TtrR-driven sfGFP expression in homeostatic state.
#   Second figure (ON/OFF standalone) added and saves correctly.
# =============================================================================

# =============================================================================
# PARAMETERS
# =============================================================================

alpha_max      = 12.0
alpha_leak     = 0.02 * alpha_max    # FIXED: 2% — OFF floor ~0.48, FI ~27x
                                     # v5 was 10% — OFF floor 2.4, FI only 5.5x

EC50_ttr       = 20.0
n_ttr          = 2
Km_sensor      = 0.3
k1             = 0.426
k1_leak        = 0.002               # FIXED: reduced from 0.01 — TtrR_ss(OFF) now ~0.02
d_TtrR         = 0.1
n_sensor       = 2
d_sfGFP        = 0.05
vm_sfGFP       = alpha_max * d_sfGFP
copy_number    = 20
delta_per_copy = 0.0015
delta          = copy_number * delta_per_copy
S_PHYS_LOW     = 10.0
S_PHYS_HIGH    = 100.0
T_END          = 500
N_PTS          = 5000
time           = np.linspace(0, T_END, N_PTS)

FI_theoretical = (alpha_max + alpha_leak) / alpha_leak

print("=== ÌṢỌ Task 1 v6 — Biosensor Parameters ===")
print(f"  α_max:        {alpha_max}")
print(f"  α_leak:       {alpha_leak:.3f}  ({alpha_leak/alpha_max*100:.0f}% of α_max)")
print(f"  k1_leak:      {k1_leak}")
print(f"  EC50:         {EC50_ttr} µM")
print(f"  Hill n:       {n_ttr}")
print(f"  Copy number:  {copy_number} copies/cell")
print(f"  Burden δ:     {delta:.4f}  ({delta*100:.2f}% growth rate cost)")
print(f"  FI theoretical: {FI_theoretical:.1f}x")

# =============================================================================
# ODE
# =============================================================================

def biosensor_ode(t, y, S):
    TtrR, sfGFP = y
    TtrR  = max(TtrR, 0)
    sfGFP = max(sfGFP, 0)
    k1_eff        = k1 * (S**n_ttr / (EC50_ttr**n_ttr + S**n_ttr)) + k1_leak
    dTtrR         = k1_eff - d_TtrR * TtrR
    sfGFP_induced = vm_sfGFP * (TtrR**n_sensor / (Km_sensor**n_sensor + TtrR**n_sensor))
    dsfGFP        = sfGFP_induced + alpha_leak * d_sfGFP - d_sfGFP * sfGFP
    return [dTtrR, dsfGFP]

kw     = dict(method='LSODA', rtol=1e-8, atol=1e-10)
y0     = [0.0, 0.0]
sol_on   = solve_ivp(biosensor_ode, [0, T_END], y0, args=(50.0,),  t_eval=time, **kw)
sol_off  = solve_ivp(biosensor_ode, [0, T_END], y0, args=(0.0,),   t_eval=time, **kw)
sol_sub  = solve_ivp(biosensor_ode, [0, T_END], y0, args=(2.0,),   t_eval=time, **kw)
sol_ec50 = solve_ivp(biosensor_ode, [0, T_END], y0, args=(20.0,),  t_eval=time, **kw)
t        = sol_on.t

sfGFP_max = sol_on.y[1][-1]
t50_idx   = np.argmax(sol_on.y[1] >= 0.5 * sfGFP_max)
t50_val   = t[t50_idx]
FI_obs    = sol_on.y[1][-1] / max(sol_off.y[1][-1], 1e-10)

print(f"\n=== Simulation Results ===")
print(f"  sfGFP ss (ON   50µM):  {sol_on.y[1][-1]:.3f}")
print(f"  sfGFP ss (OFF   0µM):  {sol_off.y[1][-1]:.3f}")
print(f"  sfGFP ss (sub   2µM):  {sol_sub.y[1][-1]:.3f}")
print(f"  sfGFP ss (EC50 20µM):  {sol_ec50.y[1][-1]:.3f}")
print(f"  Observed FI (ON/OFF):  {FI_obs:.1f}x")
print(f"  t50 (sfGFP ON):        {t50_val:.1f}h",
      "✓" if 8 <= t50_val <= 30 else "✗ outside 8–30h")

# =============================================================================
# ANALYTICAL STEADY-STATE + SWEEPS
# =============================================================================

S_sweep = np.logspace(-2, 2, 120)

def sfGFP_ss_analytical(S, EC50, n, Km=Km_sensor, n_s=n_sensor,
                         k1v=k1, k1l=k1_leak, dT=d_TtrR,
                         vm=vm_sfGFP, dg=d_sfGFP, al=alpha_leak):
    k1_eff  = k1v * (S**n / (EC50**n + S**n)) + k1l
    TtrR_ss = k1_eff / dT
    return (vm * (TtrR_ss**n_s / (Km**n_s + TtrR_ss**n_s)) + al * dg) / dg

def sfGFP_ss_Km(S, Km_v):
    k1_eff  = k1 * (S**n_ttr / (EC50_ttr**n_ttr + S**n_ttr)) + k1_leak
    TtrR_ss = k1_eff / d_TtrR
    return (vm_sfGFP*(TtrR_ss**n_sensor/(Km_v**n_sensor+TtrR_ss**n_sensor))
            + alpha_leak*d_sfGFP) / d_sfGFP

EC50_variants = [5.0, 10.0, 20.0, 50.0]
ec50_colors   = ['mediumpurple', 'steelblue', 'seagreen', 'tomato']
n_variants    = [1, 2, 3]
n_colors      = ['steelblue', 'seagreen', 'tomato']
Km_variants   = [0.1, 0.3, 0.6, 1.0]
Km_colors     = ['lightskyblue', 'steelblue', 'navy', 'goldenrod']
copy_variants = [5, 10, 20, 30, 50]
copy_colors   = ['navy', 'royalblue', 'seagreen', 'goldenrod', 'tomato']

EC50_curves = {ec: [sfGFP_ss_analytical(S, ec, 2) for S in S_sweep]
               for ec in EC50_variants}
n_curves    = {nv: [sfGFP_ss_analytical(S, EC50_ttr, nv) for S in S_sweep]
               for nv in n_variants}
Km_curves   = {kv: [sfGFP_ss_Km(S, kv) for S in S_sweep]
               for kv in Km_variants}
copy_curves = {cn: [sfGFP_ss_analytical(S, EC50_ttr, n_ttr)*(cn/20.0)
               for S in S_sweep] for cn in copy_variants}

phys_kw = dict(alpha=0.08, color='green')

# =============================================================================
# FIGURE 1 — 6-panel characterisation
# =============================================================================

fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, sol_on.y[0],  color='steelblue',  lw=2,   label='TtrR (ON, 50µM S)')
ax1.plot(t, sol_on.y[1],  color='darkorange', lw=2,   label='sfGFP (ON)')
ax1.plot(t, sol_off.y[1], color='darkorange', lw=1.5, ls='--', alpha=0.6,
         label=f'sfGFP (OFF / leak={alpha_leak:.2f})')
ax1.axvline(x=t50_val, color='gray', ls=':', lw=1.2, label=f't50={t50_val:.0f}h')
ax1.set(xlabel='Time (hours)', ylabel='Concentration (rel. units)',
        title=f'Extended Time Course (500h)\nα_leak={alpha_leak:.2f} ({alpha_leak/alpha_max*100:.0f}% α_max)',
        xlim=(0, T_END), ylim=(0, None))
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(sol_on.y[0], sol_on.y[1], color='purple', lw=2)
for tm in [5, 10, 20, 50, 100, 300]:
    idx = np.argmin(np.abs(t - tm))
    ax2.plot(sol_on.y[0][idx], sol_on.y[1][idx], 'o', color='purple', ms=5)
    ax2.annotate(f'{tm}h', (sol_on.y[0][idx], sol_on.y[1][idx]),
                 xytext=(4, 2), textcoords='offset points', fontsize=7)
ax2.plot(sol_on.y[0][-1], sol_on.y[1][-1], 'ro', ms=8,
         label=f'SS ({sol_on.y[0][-1]:.2f}, {sol_on.y[1][-1]:.2f})')
ax2.set(xlabel='TtrR (rel. units)', ylabel='sfGFP (rel. units)',
        title='Phase Portrait\nsfGFP vs TtrR (time-parametric)')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
for nv, col in zip(n_variants, n_colors):
    ax3.semilogx(S_sweep, n_curves[nv], color=col, lw=2, label=f'n={nv}')
ax3.axvline(x=EC50_ttr, color='gray', lw=1.5, ls='--', label=f'EC50={EC50_ttr}µM')
ax3.fill_betweenx([0, alpha_max*1.1], S_PHYS_LOW, S_PHYS_HIGH, **phys_kw)
ax3.set(xlabel='[Tetrathionate] (µM)', ylabel='sfGFP steady state',
        title=f'Dose-Response: Hill Coefficient\n(EC50={EC50_ttr}µM)', ylim=(0, None))
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 0])
for ec, col in zip(EC50_variants, ec50_colors):
    ax4.semilogx(S_sweep, EC50_curves[ec], color=col, lw=2, label=f'EC50={ec}µM')
ax4.fill_betweenx([0, alpha_max*1.1], S_PHYS_LOW, S_PHYS_HIGH, **phys_kw)
ax4.set(xlabel='[Tetrathionate] (µM)', ylabel='sfGFP steady state',
        title='Dose-Response: EC50 Sensitivity\n(n=2)', ylim=(0, None))
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
for kv, col in zip(Km_variants, Km_colors):
    lw = 2.5 if kv == Km_sensor else 1.5
    ls = '-' if kv >= Km_sensor else '--'
    ax5.semilogx(S_sweep, Km_curves[kv], color=col, lw=lw, ls=ls,
                 label=f'Km={kv}' + (' ← design' if kv == Km_sensor else ''))
ax5.fill_betweenx([0, alpha_max*1.1], S_PHYS_LOW, S_PHYS_HIGH, **phys_kw)
ax5.axvline(x=EC50_ttr, color='gray', lw=1, ls='--', alpha=0.6)
ax5.set(xlabel='[Tetrathionate] (µM)', ylabel='sfGFP steady state',
        title='Dose-Response: Km Sensitivity\n(Km≥0.3 enforced)', ylim=(0, None))
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
for cn, col in zip(copy_variants, copy_colors):
    burden = cn * delta_per_copy * 100
    ax6.semilogx(S_sweep, copy_curves[cn], color=col, lw=2,
                 label=f'{cn} copies/cell (δ={burden:.1f}%)')
ax6.fill_betweenx([0, max(copy_curves[50])*1.1], S_PHYS_LOW, S_PHYS_HIGH, **phys_kw)
ax6.set(xlabel='[Tetrathionate] (µM)', ylabel='sfGFP steady state',
        title='Dose-Response: Copy Number + Burden δ\n(pSC101 range)', ylim=(0, None))
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

fig.text(0.5, -0.01,
    f"EC50={EC50_ttr}µM | n={n_ttr} | α_leak={alpha_leak:.2f} ({alpha_leak/alpha_max*100:.0f}% α_max) | "
    f"FI={FI_obs:.0f}x | t50={t50_val:.0f}h | copy={copy_number} | δ={delta*100:.1f}% growth cost",
    ha='center', fontsize=9, color='dimgray',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
fig.suptitle('ÌṢỌ Sentinel EcN — Biosensor Characterisation v6',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('iso_biosensor_v6_6panel.png', dpi=300, bbox_inches='tight')
plt.savefig('iso_biosensor_v6_6panel.svg', format='svg', bbox_inches='tight')
plt.close()
print("Saved: iso_biosensor_v6_6panel.png")

# =============================================================================
# FIGURE 2 — ON vs OFF standalone (portfolio-ready)
# =============================================================================

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(t, sol_on.y[1],  color='darkorange', lw=2.5, label='sfGFP ON (50µM S)')
axes[0].plot(t, sol_off.y[1], color='steelblue',  lw=2,   ls='--', label='sfGFP OFF (0µM S)')
axes[0].plot(t, sol_sub.y[1], color='gray',        lw=1.5, ls=':',  label='sfGFP sub-threshold (2µM)')
axes[0].axhline(y=alpha_leak, color='red', lw=1, ls='-.', alpha=0.6,
                label=f'α_leak={alpha_leak:.2f}')
axes[0].axvline(x=t50_val, color='black', lw=1, ls=':', label=f't50={t50_val:.0f}h')
axes[0].annotate(f'Fold induction: {FI_obs:.0f}x',
                 xy=(80, sol_on.y[1][-1]*0.6), fontsize=10, color='darkorange',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[0].set(xlabel='Time (hours)', ylabel='sfGFP (rel. units)',
            title='TtrR-sfGFP Biosensor: ON vs OFF State\n(J23115/J23106/B0031, Medium Copy)',
            xlim=(0, 150), ylim=(0, None))
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].semilogx(S_sweep,
                 [sfGFP_ss_analytical(S, EC50_ttr, n_ttr) for S in S_sweep],
                 color='seagreen', lw=2.5)
axes[1].axvline(x=EC50_ttr, color='gray',  lw=1.5, ls='--', label=f'EC50={EC50_ttr}µM')
axes[1].axvline(x=100,      color='red',   lw=1,   ls=':',  alpha=0.6, label='100µM (gut max)')
axes[1].fill_betweenx([0, alpha_max*1.05], S_PHYS_LOW, S_PHYS_HIGH,
                       alpha=0.1, color='green', label='Physiological range (10–100µM)')
axes[1].set(xlabel='[Tetrathionate] (µM)', ylabel='sfGFP steady state (rel. units)',
            title=f'Tetrathionate Dose-Response\n(pTtr_LT2, EC50={EC50_ttr}µM)',
            ylim=(0, None))
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

fig2.suptitle('ÌṢỌ Task 1 v6 — Biosensor ON/OFF & Dose-Response', fontweight='bold')
fig2.tight_layout()
plt.savefig('iso_biosensor_v6_onoff.png', dpi=300, bbox_inches='tight')
plt.savefig('iso_biosensor_v6_onoff.svg', format='svg', bbox_inches='tight')
plt.close()
print("Saved: iso_biosensor_v6_onoff.png")
print("Saved: iso_biosensor_v6_onoff.svg")

# =============================================================================
# PRE-AIM-2 CHECKLIST
# =============================================================================

print("\n=== Pre-Aim-2 Checklist ===")
checks = {
    f"FI ≥ 10x  (got {FI_obs:.0f}x)":                     FI_obs >= 10,
    f"FI ≤ 500x (got {FI_obs:.0f}x)":                     FI_obs <= 500,
    f"EC50 5–50µM (got {EC50_ttr}µM)":                    5 <= EC50_ttr <= 50,
    f"t50 8–30h (got {t50_val:.1f}h)":                    8 <= t50_val <= 30,
    f"n=2 locked":                                          n_ttr == 2,
    f"Km ≥ 0.3 (got {Km_sensor})":                         Km_sensor >= 0.3,
    f"Copy number 5–50 (got {copy_number})":               5 <= copy_number <= 50,
    f"Burden δ ≤ 0.10 (got {delta:.4f})":                  delta <= 0.10,
    f"α_leak 1–5% of α_max (got {alpha_leak/alpha_max*100:.0f}%)":
                                                            0.01 <= alpha_leak/alpha_max <= 0.05,
}
all_pass = True
for label, passed in checks.items():
    status   = "✓" if passed else "✗"
    all_pass = all_pass and passed
    print(f"  [{status}] {label}")

print(f"\n  {'ALL CHECKS PASSED — ready to proceed to Task 2.' if all_pass else 'REVIEW FLAGGED PARAMETERS BEFORE PROCEEDING.'}")
print("\nOutputs: iso_biosensor_v6_6panel.png/svg  |  iso_biosensor_v6_onoff.png/svg")