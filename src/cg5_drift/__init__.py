"""
cg5_drift: tools for harmonic drift correction of CG-5 base-station time series.
"""

from .io import load_dataset, to_seconds, site_label_from_filename
from .fitting import (
    correct_memory_reset,
    stima_ampiezza_fase_wls,
    chi2_reduced_global,
    best_omega_in_band,
    optimize_omegas_cd,
    estimate_omega_uncertainties,
)

__all__ = [
    "load_dataset",
    "to_seconds",
    "site_label_from_filename",
    "correct_memory_reset",
    "stima_ampiezza_fase_wls",
    "chi2_reduced_global",
    "best_omega_in_band",
    "optimize_omegas_cd",
    "estimate_omega_uncertainties",
]
