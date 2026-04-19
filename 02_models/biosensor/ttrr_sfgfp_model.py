import tellurium as te
import matplotlib.pyplot as plt

model = te.loada("""
  J1: -> TtrR; k1
  J2: TtrR -> ; d1*TtrR
  J3: -> sfGFP; (vm * TtrR^n) / (Km^n + TtrR^n)
  J4: sfGFP -> ; d2*sfGFP

  k1 = 0.5
  d1 = 0.1

  vm = 1.0
  Km = 0.3
  n = 2

  d2 = 0.05

  TtrR = 0
  sfGFP = 0
""")

result = model.simulate(0, 72, 500)

plt.plot(result[:, 0], result[:, 1], label="TtrR")
plt.plot(result[:, 0], result[:, 2], label="sfGFP")

plt.xlabel("Time (hours)")
plt.ylabel("Concentration")
plt.title("TtrR-sfGFP Biosensor Simulation")
plt.legend()
plt.grid(True)

plt.show()