# ÌṢỌ Sentinel EcN

> *Yoruba: to be well; to recover.*

A model-first, constraint-aware approach to engineering *E. coli* Nissle 1917 (EcN) as a gut sentinel: sensing context, responding with targeted antimicrobials, and remaining governable through built-in containment.

## Project documentation

Full project documentation, methodology, aims, and results are maintained on the HTGAA 2026 course portfolio:

**[ÌṢỌ Individual Final Project — HTGAA Spring 2026](https://pages.htgaa.org/2026a/john-adeyemo-adedeji/projects/individual-final-project/-_index/index.html)**

The portfolio covers system architecture, ODE modelling rationale, construct design decisions, experimental design, key figures, and references. This repository contains all associated code, sequences, and simulation outputs.

## Repository structure

```
01_sequences/
  parts/              Individual BioBrick and regulatory parts
  constructs/         Assembled constructs and GenBank files
  protein/            ttrR and ttrS protein FAA files
  twist_delivered/    Authoritative Twist-delivered sequences

02_models/
  biosensor/          Biosensor module ODE simulations (v1-v4)
  four_module/        Full four-module ODE (Task 2)
  pareto/             Pareto landscape sweep (Task 3)
  sensitivity/        PRCC and Sobol sensitivity analysis (Task 4)
  moran/              Evolutionary stability simulation (Task 5)
  sbml/               SBML exports for citability

03_analysis/
  qc_pipeline/        Construct QC pipeline (iso_qc_pipeline_v4.sh)
  figures/            All simulation and construct map outputs

04_data/
  ncbi/               ttrR and ttrS NCBI datasets

06_references/        Palmer 2017 and sequence PDFs
```

## Quick start

```bash
python3.11 -m venv ~/.envs/iso-ecn
source ~/.envs/iso-ecn/bin/activate
pip install -r requirements.txt
python 02_models/biosensor/iso_biosensor_sim_v4.py
python 02_models/four_module/iso_four_module_sim.py
```

## Interactive Simulator

A live browser-based ODE simulator is embedded on the HTGAA project page under the **Simulator** tab:  
[Project Simulation Website](https://pages.htgaa.org/2026a/john-adeyemo-adedeji/projects/individual-final-project/-_index/index.html)

The system implements a fully client-side four-module Runge–Kutta (RK4) integration framework with no server dependency. Nine gut-system parameters are exposed as interactive sliders, enabling real-time control of system dynamics.

On each parameter update, the model recomputes the coupled ODE system and updates five key emergent metrics: fold induction, t₅₀, pathogen suppression, system burden (δ), and Moran fixation probability.

**Parameters exposed:**

| Slider | Range | Design default |
|--------|-------|----------------|
| Tetrathionate [S] | 0–100 µM | 50 µM |
| EC50 | 1–50 µM | 20 µM |
| Hill coefficient n | 1–3 | 2 |
| Leaky expression α_leak | 1–20% α_max | 2% |
| MccH47 production k_M | 0.01–0.5 µM/h | 0.2 µM/h |
| Pathogen kill rate k_kill | 0.01–1.0 µM⁻¹h⁻¹ | 0.3 µM⁻¹h⁻¹ |
| Plasmid copy number | 5–50 copies/cell | 20 |
| Gate sharpness n_reg | 1–5 | 3 |
| Time horizon | 24–500 h | 150 h |

**Outputs displayed in real time:**
- Biosensor and effector time course (sfGFP ON vs OFF, MccH47)
- Pathogen suppression time course
- Dose-response sigmoid with physiological window
- Containment escape probability (single ΔdapA vs dual ΔdapA+ΔthyA)

**ODE solver:** RK4 fixed-step, dt=0.1 h. Sufficient for this parameter regime; no LSODA dependency required in-browser.

**Parameters grounded in:** Palmer et al. 2017 (ACS Infect. Dis.); Stritzker et al. 2007 (Int. J. Med. Microbiol.); Scott et al. 2010 (Science).

**Licence:** MIT. Cite as: Adedeji, J.A. (2026). ÌṢỌ Sentinel EcN computational framework. GitHub: Jonahnki/iso-sentinel-ecn. DOI: 10.5281/zenodo.20098747.

## Citation

If you use this work, please cite:

[![DOI](https://zenodo.org/badge/1187462265.svg)](https://doi.org/10.5281/zenodo.20098747)

or:

> DOI: 10.5281/zenodo.20098747 

## License

MIT
