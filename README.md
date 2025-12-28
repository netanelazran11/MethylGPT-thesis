# MethylGPT-thesis
Thesis repository extending MethylGPT, a transformer-based foundation model for DNA methylation, designed to learn contextual CpG representations and support diverse downstream epigenetic analyses.


## Overview
This repository contains the codebase for an MSc thesis that extends MethylGPT,
a transformer-based foundation model for DNA methylation. The goal is to study
representation learning and downstream adaptability of contextual CpG embeddings
under controlled experimental settings.

## Relation to Original MethylGPT
This work builds directly upon the official MethylGPT implementation:
https://github.com/albert-ying/MethylGPT

All core architectural components and pretrained weights are preserved.
Extensions in this repository focus on fine-tuning strategies, experimental
design, and systematic analysis of model behavior.

## Repository Structure
- `methylgpt/` – core MethylGPT model (upstream or submodule)
- `tutorials/` – original tutorials from the upstream repository
- `thesis/` – thesis-specific code (fine-tuning, configs, experiments)
- `scripts/` – reproducible run scripts
- `data/` – placeholder and documentation only (no raw data included)

## Status
This repository is under active development as part of an MSc thesis.
