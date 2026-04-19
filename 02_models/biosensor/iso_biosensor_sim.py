import tellurium as te
import matplotlib.pyplot as plt
import numpy as np

# ON state model: tetrathionate present, TtrR produced constitutively
model_ON = te.loada('''
  J1: -> TtrR; k1
  J2: TtrR -> ; d1*TtrR
  J3: -> sfGFP; (vm * TtrR^n) / (Km^n + TtrR^n)
  J4: sfGFP -> ; d2*sfGFP

  k1 = 0.5; d1 = 0.1
  vm = 1.0; Km = 0.3; n = 2
  d2 = 0.05
  TtrR = 0; sfGFP = 0
''')

# OFF state model: no tetrathionate, TtrR not produced
model_OFF = te.loada('''
  J1: -> TtrR; k1
  J2: TtrR -> ; d1*TtrR
  J3: -> sfGFP; (vm * TtrR^n) / (Km^n + TtrR^n)
  J4: sfGFP -> ; d2*sfGFP

  k1 = 0.0; d1 = 0.1
  vm = 1.0; Km = 0.3; n = 2
  d2 = 0.05
  TtrR = 0; sfGFP = 0
''')

# simulate both to 150 hours for full steady state
result_ON  = model_ON.simulate(0, 150, 1000)
result_OFF = model_OFF.simulate(0, 150, 1000)

time = result_ON[:, 0]

TtrR_ON   = result_ON[:, 1]
sfGFP_ON  = result_ON[:, 2]
sfGFP_OFF = result_OFF[:, 2]

# calculate fold induction at steady state
fold = sfGFP_ON[-1] / (sfGFP_OFF[-1] + 1e-10)
print(f"TtrR steady state (ON):   {TtrR_ON[-1]:.3f}")
print(f"sfGFP steady state (ON):  {sfGFP_ON[-1]:.3f}")
print(f"sfGFP steady state (OFF): {sfGFP_OFF[-1]:.6f}")
print(f"Fold induction (ON/OFF):  {fold:.1f}x")

# plot
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(time, TtrR_ON,   color='steelblue',   linewidth=2,   label='TtrR (tetrathionate present)')
ax.plot(time, sfGFP_ON,  color='darkorange',  linewidth=2,   label='sfGFP ON (tetrathionate present)')
ax.plot(time, sfGFP_OFF, color='darkorange',  linewidth=2,
        linestyle='--', label='sfGFP OFF (no tetrathionate)')

ax.axhline(y=sfGFP_ON[-1],  color='darkorange', linewidth=0.8, linestyle=':', alpha=0.5)
ax.axhline(y=TtrR_ON[-1],   color='steelblue',  linewidth=0.8, linestyle=':', alpha=0.5)

ax.annotate(f'Steady state: {sfGFP_ON[-1]:.1f}',
            xy=(150, sfGFP_ON[-1]), xytext=(120, sfGFP_ON[-1] + 1),
            fontsize=9, color='darkorange')

ax.annotate(f'Fold induction: {fold:.0f}x',
            xy=(75, sfGFP_ON[-1]/2),
            fontsize=10, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Concentration (relative units)', fontsize=12)
ax.set_title('TtrR-sfGFP Biosensor: ON vs OFF State', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 150)
ax.set_ylim(0)

plt.tight_layout()
plt.savefig('iso_biosensor_ON_OFF.png', dpi=300, bbox_inches='tight')
plt.savefig('iso_biosensor_ON_OFF.svg', format='svg', bbox_inches='tight')
plt.show()

print("Figures saved: iso_biosensor_ON_OFF.png and iso_biosensor_ON_OFF.svg")