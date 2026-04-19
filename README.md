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

## Citation

See `CITATION.cff`. ORCID: [0009-0004-1257-4551](https://orcid.org/0009-0004-1257-4551)

## License

MIT
