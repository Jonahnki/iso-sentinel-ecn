import tellurium as te
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# =============================================================================
# ÌṢỌ Sentinel EcN — Biosensor Simulation v4
# Extended run: full variable relationships, 500h time course
# =============================================================================

copy_factor = 6.0
k1          = 0.071 * copy_factor   # 0.426
vm          = 0.10  * copy_factor   # 0.600
d1          = 0.1
d2          = 0.05
Km          = 0.3
k1_leak     = 0.01

def make_model(k1_val, n_val, vm_val, km_val=0.3):
    return te.loada(f'''
      J1: -> TtrR;  k1_val
      J2: TtrR -> ; d1 * TtrR
      J3: -> sfGFP; (vm_val * TtrR^n_val) / (km_val^n_val + TtrR^n_val)
      J4: sfGFP -> ; d2 * sfGFP
      k1_val = {k1_val:.6f}
      d1     = {d1}
      vm_val = {vm_val:.6f}
      km_val = {km_val:.4f}
      n_val  = {n_val}
      d2     = {d2}
      TtrR   = 0
      sfGFP  = 0
    ''')

# =============================================================================
# PART 1: Extended ON/OFF time course — 500h to show full steady state
# =============================================================================

T_END  = 500
N_PTS  = 2000

model_ON  = make_model(k1,      n_val=2, vm_val=vm)
model_OFF = make_model(k1_leak, n_val=2, vm_val=vm)

result_ON  = model_ON.simulate(0,  T_END, N_PTS)
result_OFF = model_OFF.simulate(0, T_END, N_PTS)

time      = result_ON[:, 0]
TtrR_ON   = result_ON[:, 1]
sfGFP_ON  = result_ON[:, 2]
sfGFP_OFF = result_OFF[:, 2]

fold      = sfGFP_ON[-1] / (sfGFP_OFF[-1] + 1e-12)
TtrR_ss   = TtrR_ON[-1]
sfGFP_ss  = sfGFP_ON[-1]
leak_ss   = sfGFP_OFF[-1]

def compute_t50(time_arr, signal):
    half = signal[-1] / 2.0
    idx  = np.where(signal >= half)[0]
    return time_arr[idx[0]] if len(idx) > 0 else np.nan

t50_ttrR  = compute_t50(time, TtrR_ON)
t50_sfGFP = compute_t50(time, sfGFP_ON)

print("=== Extended Simulation Results (500h) ===")
print(f"  TtrR steady state:      {TtrR_ss:.3f}")
print(f"  sfGFP steady state ON:  {sfGFP_ss:.3f}")
print(f"  sfGFP steady state OFF: {leak_ss:.4f}")
print(f"  Fold induction:         {fold:.1f}x")
print(f"  t50 TtrR:               {t50_ttrR:.1f}h")
print(f"  t50 sfGFP:              {t50_sfGFP:.1f}h")
print(f"  TtrR lag (sfGFP-TtrR):  {t50_sfGFP - t50_ttrR:.1f}h")

# =============================================================================
# PART 2: TtrR-sfGFP phase portrait — shows relationship directly
# =============================================================================

# Phase portrait: sfGFP vs TtrR over time (parametric)
# Shows how sfGFP responds to TtrR accumulation

# =============================================================================
# PART 3: Dose-response sweeps
# =============================================================================

tetrathionate_conc = np.logspace(-2, 2, 80)
EC50_values        = [5.0, 10.0, 20.0]
n_values           = [1, 2, 3]

results = {ec50: {n: [] for n in n_values} for ec50 in EC50_values}

for EC50 in EC50_values:
    for n_val in n_values:
        for conc in tetrathionate_conc:
            k1_eff = k1 * (conc**2) / (EC50**2 + conc**2)
            m = make_model(k1_eff, n_val=n_val, vm_val=vm)
            r = m.simulate(0, 500, 600)
            results[EC50][n_val].append(r[-1, 2])
        results[EC50][n_val] = np.array(results[EC50][n_val])

# =============================================================================
# PART 4: Km sensitivity sweep — shows effect of Hill threshold
# =============================================================================

Km_values = [0.1, 0.3, 0.6, 1.0]
sfGFP_km  = {}
for km_val in Km_values:
    row = []
    for conc in tetrathionate_conc:
        k1_eff = k1 * (conc**2) / (10.0**2 + conc**2)
        m = make_model(k1_eff, n_val=2, vm_val=vm, km_val=km_val)
        r = m.simulate(0, 500, 600)
        row.append(r[-1, 2])
    sfGFP_km[km_val] = np.array(row)

# =============================================================================
# PART 5: Copy number sensitivity
# =============================================================================

copy_numbers = [5, 10, 20, 30, 50]   # pSC101 through medium copy range
sfGFP_copy   = {}
for cn in copy_numbers:
    cn_factor = cn / 5.0
    k1_cn     = 0.071 * cn_factor
    vm_cn     = 0.10  * cn_factor
    row = []
    for conc in tetrathionate_conc:
        k1_eff = k1_cn * (conc**2) / (10.0**2 + conc**2)
        m = make_model(k1_eff, n_val=2, vm_val=vm_cn)
        r = m.simulate(0, 500, 600)
        row.append(r[-1, 2])
    sfGFP_copy[cn] = np.array(row)

# =============================================================================
# PLOTTING — 2x3 grid
# =============================================================================

fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# ---- Panel 1: Extended time course ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time, TtrR_ON,   color='steelblue',  linewidth=2,   label='TtrR (ON)')
ax1.plot(time, sfGFP_ON,  color='darkorange', linewidth=2,   label='sfGFP (ON)')
ax1.plot(time, sfGFP_OFF, color='darkorange', linewidth=1.5,
         linestyle='--', label='sfGFP (OFF / leak)')
ax1.axhline(y=TtrR_ss,  color='steelblue',  linewidth=0.8, linestyle=':', alpha=0.5)
ax1.axhline(y=sfGFP_ss, color='darkorange', linewidth=0.8, linestyle=':', alpha=0.5)
ax1.axhline(y=leak_ss,  color='gray',       linewidth=0.8, linestyle=':', alpha=0.5)
ax1.axvline(x=t50_ttrR,  color='steelblue',  linewidth=1, linestyle='--', alpha=0.5,
            label=f'TtrR t50={t50_ttrR:.0f}h')
ax1.axvline(x=t50_sfGFP, color='darkorange', linewidth=1, linestyle='--', alpha=0.5,
            label=f'sfGFP t50={t50_sfGFP:.0f}h')
ax1.annotate(f'Fold: {fold:.0f}x\nLeak: {leak_ss:.2f}',
             xy=(300, sfGFP_ss * 0.45), fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
ax1.set_xlabel('Time (hours)', fontsize=10)
ax1.set_ylabel('Concentration (relative units)', fontsize=10)
ax1.set_title('Extended Time Course (500h)\nTtrR → sfGFP dynamics', fontsize=10)
ax1.legend(fontsize=8, loc='center right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, T_END)
ax1.set_ylim(0)

# ---- Panel 2: Phase portrait — sfGFP vs TtrR ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(TtrR_ON, sfGFP_ON, color='purple', linewidth=2)
# Mark time points along trajectory
for t_mark in [5, 10, 20, 50, 100, 200]:
    idx = np.argmin(np.abs(time - t_mark))
    ax2.scatter(TtrR_ON[idx], sfGFP_ON[idx], s=40, zorder=5,
                color='purple', alpha=0.7)
    ax2.annotate(f'{t_mark}h', (TtrR_ON[idx], sfGFP_ON[idx]),
                 fontsize=7, xytext=(4, 2), textcoords='offset points')
ax2.scatter(TtrR_ON[-1], sfGFP_ON[-1], s=80, color='red', zorder=6,
            label=f'Steady state\n({TtrR_ss:.2f}, {sfGFP_ss:.2f})')
ax2.set_xlabel('TtrR (relative units)', fontsize=10)
ax2.set_ylabel('sfGFP (relative units)', fontsize=10)
ax2.set_title('Phase Portrait\nsfGFP vs TtrR (time-parametric)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0)
ax2.set_ylim(0)

# ---- Panel 3: Dose-response by Hill coefficient ----
ax3 = fig.add_subplot(gs[0, 2])
colors_n = {1: 'cornflowerblue', 2: 'seagreen', 3: 'tomato'}
for n_val in n_values:
    ax3.semilogx(tetrathionate_conc, results[10.0][n_val],
                 color=colors_n[n_val], linewidth=2, label=f'n={n_val}')
ymax3 = max(results[10.0][3]) * 1.15
ax3.axvline(x=10,  color='gray', linestyle='--', linewidth=1, label='EC50=10µM')
ax3.axvline(x=100, color='red',  linestyle=':',  linewidth=1, alpha=0.5)
ax3.fill_betweenx([0, ymax3], 10, 100, alpha=0.07, color='green',
                  label='Physiological range')
ax3.set_xlabel('[Tetrathionate] (µM)', fontsize=10)
ax3.set_ylabel('sfGFP steady state', fontsize=10)
ax3.set_title('Dose-Response: Hill Coefficient\n(EC50=10µM)', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, ymax3)

# ---- Panel 4: Dose-response by EC50 ----
ax4 = fig.add_subplot(gs[1, 0])
colors_ec = {5.0: 'mediumorchid', 10.0: 'seagreen', 20.0: 'coral'}
for EC50 in EC50_values:
    ax4.semilogx(tetrathionate_conc, results[EC50][2],
                 color=colors_ec[EC50], linewidth=2, label=f'EC50={EC50}µM')
ymax4 = max(results[5.0][2]) * 1.15
ax4.axvline(x=10,  color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax4.axvline(x=100, color='red',  linestyle=':',  linewidth=1, alpha=0.5)
ax4.fill_betweenx([0, ymax4], 10, 100, alpha=0.07, color='green',
                  label='Physiological range')
ax4.set_xlabel('[Tetrathionate] (µM)', fontsize=10)
ax4.set_ylabel('sfGFP steady state', fontsize=10)
ax4.set_title('Dose-Response: EC50 Sensitivity\n(n=2)', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, ymax4)

# ---- Panel 5: Km sensitivity ----
ax5 = fig.add_subplot(gs[1, 1])
colors_km = ['#8ecae6', '#219ebc', '#023047', '#ffb703']
for i, km_val in enumerate(Km_values):
    ax5.semilogx(tetrathionate_conc, sfGFP_km[km_val],
                 color=colors_km[i], linewidth=2, label=f'Km={km_val}')
ymax5 = max(sfGFP_km[0.1]) * 1.15
ax5.axvline(x=10,  color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax5.axvline(x=100, color='red',  linestyle=':',  linewidth=1, alpha=0.5)
ax5.fill_betweenx([0, ymax5], 10, 100, alpha=0.07, color='green',
                  label='Physiological range')
ax5.set_xlabel('[Tetrathionate] (µM)', fontsize=10)
ax5.set_ylabel('sfGFP steady state', fontsize=10)
ax5.set_title('Dose-Response: Km Sensitivity\n(n=2, EC50=10µM)', fontsize=10)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.set_ylim(0, ymax5)

# ---- Panel 6: Copy number sensitivity ----
ax6 = fig.add_subplot(gs[1, 2])
cmap  = plt.cm.viridis
cnorm = plt.Normalize(vmin=min(copy_numbers), vmax=max(copy_numbers))
for cn in copy_numbers:
    ax6.semilogx(tetrathionate_conc, sfGFP_copy[cn],
                 color=cmap(cnorm(cn)), linewidth=2, label=f'{cn} copies/cell')
ymax6 = max(sfGFP_copy[50]) * 1.15
ax6.axvline(x=10,  color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax6.axvline(x=100, color='red',  linestyle=':',  linewidth=1, alpha=0.5)
ax6.fill_betweenx([0, ymax6], 10, 100, alpha=0.07, color='green',
                  label='Physiological range')
ax6.set_xlabel('[Tetrathionate] (µM)', fontsize=10)
ax6.set_ylabel('sfGFP steady state', fontsize=10)
ax6.set_title('Dose-Response: Copy Number Effect\n(pSC101 → Medium Copy range)', fontsize=10)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0, ymax6)

fig.suptitle('ÌṢỌ Sentinel EcN — Full Biosensor Characterisation (v4)',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('iso_biosensor_sim_v4.png', dpi=300, bbox_inches='tight')
plt.savefig('iso_biosensor_sim_v4.svg', format='svg', bbox_inches='tight')
plt.show()

print("\n=== Final Performance Summary ===")
print(f"  Fold induction:       {fold:.1f}x")
print(f"  TtrR t50:             {t50_ttrR:.1f}h")
print(f"  sfGFP t50:            {t50_sfGFP:.1f}h")
print(f"  Phosphorelay lag:     {t50_sfGFP - t50_ttrR:.1f}h")
print(f"  Leakage (OFF):        {leak_ss:.4f}")
print(f"  TtrR steady state:    {TtrR_ss:.3f}")
print(f"  sfGFP steady state:   {sfGFP_ss:.3f}")
print("\nFigures saved: iso_biosensor_sim_v4.png and iso_biosensor_sim_v4.svg")