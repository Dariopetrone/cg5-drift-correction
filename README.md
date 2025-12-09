# CG-5 Drift Correction: Harmonic Analysis Toolbox

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

Automated harmonic correction workflow for instrumental drift in **Scintrex CG-5 gravimeters**.

This repository provides the reference implementation of the **four-harmonic drift model** described in Petrone et al. (2025) to remove tidal-induced and long-period drift components from microgravity base-station time series.

**Key result**: base-station scatter is reduced from **76–2000 µGal** to **10–50 µGal**.

## Key Features

- Weighted least-squares fitting of four periodic components (semi-diurnal, diurnal and long-period terms)
- Coordinate-descent optimization within prescribed frequency/period bands
- Automatic memory-reset jump detection and segment-wise offset correction
- Uncertainty estimation for harmonic frequencies and periods
- Residual analysis with Gaussian diagnostics
- Full transparency: all analysis code and example data available

## Installation

### Requirements

- Python 3.8 or higher
- `numpy`, `scipy`, `pandas`, `matplotlib`, `Pillow`

### Quick Install

```bash
git clone https://github.com/Dariopetrone/cg5-drift-correction.git
cd cg5-drift-correction

# create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate    # su Windows: .venv\Scripts\activate

# install required packages
python -m pip install numpy scipy pandas matplotlib Pillow
