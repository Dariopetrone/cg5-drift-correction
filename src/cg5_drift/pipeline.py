# -*- coding: utf-8 -*-
"""
End-to-end pipeline for CG-5 drift correction with 4-harmonic fit.

This script:
- loads all CSV files from BASE_PATH,
- applies memory-reset correction,
- fits the 4-harmonic model,
- estimates frequency/period uncertainties,
- generates combined panel figures and Gaussian residual PDFs.
"""

import os
import glob
from pathlib import Path

import numpy as np
import matplotlib
# Non-interactive backend (safe on headless machines / CI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from scipy.stats import norm
from PIL import Image

from .io import load_dataset, to_seconds, site_label_from_filename
from .fitting import (
    correct_memory_reset,
    optimize_omegas_cd,
    estimate_omega_uncertainties,
)

# Disable Pillow safety limit for high-resolution TIFF export
Image.MAX_IMAGE_PIXELS = None

# ======== Settings ======== #

# Repository root = .../cg5-drift-correction (two levels above src/cg5_drift)
ROOT = Path(__file__).resolve().parents[2]

# Where the CSVs used in the paper are stored
BASE_PATH = os.fspath(ROOT / "data")

# Where figures will be written
OUT_DIR = os.fspath(ROOT / "figures")

# Threshold for detecting memory-reset jumps in gravity (mGal)
JUMP_THRESHOLD = 1.0

# Minimum number of samples between two jumps to consider them distinct
MIN_GAP_POINTS = 3

# Maximum number of sites (CSV files) to process in the combined figures
MAX_SITES = 8


# ======== Target harmonics (period bands) ======== #
targets = [
    ("12h", 12 * 3600, 2.4 * 3600),
    ("24h", 24 * 3600, 4.8 * 3600),
    ("27d", 27 * 24 * 3600, 5.4 * 24 * 3600),
    ("54d", 54 * 24 * 3600, 10.8 * 24 * 3600),
]


def main(base_path: str = BASE_PATH):
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Figures will be saved in: {OUT_DIR}")

    files = sorted(glob.glob(os.path.join(base_path, "*.csv")))[:MAX_SITES]
    print(
        f"Found {len(files)} CSV files; "
        f"{2 * len(files) + 1} figures will be generated (2 per site + combined Gaussians)."
    )

    # Global style parameters (Geophysics-ready figure layout)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    # Combined panel figures (fit and residuals)
    fig_fit, axes_fit = plt.subplots(
        4, 2, figsize=(8.66, 10), constrained_layout=True
    )
    fig_resid, axes_resid = plt.subplots(
        4, 2, figsize=(8.66, 10), constrained_layout=True
    )
    axes_fit = axes_fit.ravel()
    axes_resid = axes_resid.ravel()

    results = []

    for i, fpath in enumerate(files[:8]):
        df = load_dataset(fpath)
        label = site_label_from_filename(fpath)

        # Robust σ: replace non-positive / non-finite SD values by median SD
        sigma_raw = df["sigma"].to_numpy()
        if np.any((sigma_raw > 0) & np.isfinite(sigma_raw)):
            sref = np.median(
                sigma_raw[(sigma_raw > 0) & np.isfinite(sigma_raw)]
            )
        else:
            sref = 1.0
        sigma = np.where(
            (~np.isfinite(sigma_raw)) | (sigma_raw <= 0), sref, sigma_raw
        )

        # Correct memory-reset jumps in gravity series
        g_corr, _, _ = correct_memory_reset(
            df["g"].to_numpy(),
            threshold=JUMP_THRESHOLD,
            min_gap=MIN_GAP_POINTS,
        )

        # Diagnostic printout of corrected gravity statistics
        g_mean = float(np.mean(g_corr))
        g_std = float(np.std(g_corr, ddof=1))
        print(
            f"\nSite {label}: g_corr mean = {g_mean:.5f} mGal, "
            f"std = {g_std:.5f} mGal"
        )

        # Time axis in seconds from first sample
        t0 = df["DATETIME"].values[0]
        t_sec = to_seconds(df["DATETIME"].values, t0).astype(float)

        # Optimize the four angular frequencies and fit the 4-sinusoid model
        w_opt, (fit, A, phi, off), chi2_red = optimize_omegas_cd(
            t_sec,
            g_corr,
            sigma,
            targets,
            n_steps=400,
            max_iter=8,
            tol=1e-4,
        )
        resid = g_corr - fit

        # Gaussian fit to residuals
        mu, sd = norm.fit(resid)

        # Estimate uncertainties on ω and T
        w_err = estimate_omega_uncertainties(
            t_sec, g_corr, sigma, w_opt, targets, n_steps=160
        )

        T_sec = 2.0 * np.pi / w_opt
        # Error propagation: σ_T = (2π / ω²) σ_ω
        T_err_sec = (2.0 * np.pi / (w_opt ** 2)) * w_err

        print(f"Site {label}: reduced χ² = {chi2_red:.3f}")
        for (name, _, _), wi, dwi, Ti, dTi in zip(
            targets, w_opt, w_err, T_sec, T_err_sec
        ):
            wi_h = wi * 3600.0
            dwi_h = dwi * 3600.0
            Ti_h = Ti / 3600.0
            dTi_h = dTi / 3600.0

            if np.isfinite(dwi_h) and np.isfinite(dTi_h):
                print(
                    f"  {name}: ω = {wi_h:.6e} ± {dwi_h:.6e} rad/h, "
                    f"T = {Ti_h:.3f} ± {dTi_h:.3f} h"
                )
            else:
                print(
                    f"  {name}: ω = {wi_h:.6e} rad/h (uncertainty not resolved), "
                    f"T = {Ti_h:.3f} h"
                )

        # ---- Fit panel (gravity + 4-sinusoid model) ---- #
        ax = axes_fit[i]
        t_days = t_sec / 86400.0

        t_dense_days = np.linspace(t_days.min(), t_days.max(), 1000)
        fit_dense = (
            np.sum(
                [
                    Ai * np.sin(wi * t_dense_days * 86400.0 + phii)
                    for Ai, wi, phii in zip(A, w_opt, phi)
                ],
                axis=0,
            )
            + off
        )

        ax.errorbar(
            t_days,
            g_corr,
            yerr=sigma,
            fmt="o",
            markersize=3,
            markeredgecolor="blue",
            markerfacecolor="blue",
            alpha=0.7,
            ecolor="gray",
            elinewidth=0.8,
            capsize=1.5,
            label="measured gravity",
        )

        ax.plot(
            t_dense_days,
            fit_dense,
            color="red",
            lw=1.2,
            label="4-sinusoid drift model",
        )

        ax.set_title(f"{label}", fontsize=8)
        ax.set_xlabel("time (days)")
        ax.set_ylabel("gravity (mGal)")
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.grid(False)

        if i == 0:
            ax.legend(
                loc="upper left",
                fontsize=7,
                frameon=True,
                facecolor="white",
                framealpha=0.8,
                edgecolor="gray",
            )

        # ---- Residuals panel (histogram + Gaussian PDF) ---- #
        axr = axes_resid[i]
        x_vals = np.linspace(resid.min(), resid.max(), 600)

        axr.hist(
            resid,
            bins=30,
            density=True,
            color="lightgray",
            edgecolor="black",
            alpha=0.7,
        )
        axr.plot(x_vals, norm.pdf(x_vals, mu, sd), "r-", lw=1)

        axr.set_title(f"{label}", fontsize=8)
        axr.set_xlabel("residuals (mGal)")
        axr.set_ylabel("density")
        axr.tick_params(labelsize=7)
        axr.grid(False)

        results.append({"label": label, "mu": float(mu), "sd": float(sd)})

    # ======== Save combined panel figures (PNG + CMYK TIFF) ======== #
    for name, fig in [
        ("combined_fit_panels", fig_fit),
        ("combined_residuals_panels", fig_resid),
    ]:
        png_path = os.path.join(OUT_DIR, f"{name}.png")
        tiff_path = os.path.join(OUT_DIR, f"{name}.tiff")

        fig.savefig(png_path, dpi=1200, bbox_inches="tight")
        plt.close(fig)

        with Image.open(png_path) as im:
            im.convert("CMYK").save(
                tiff_path, compression="tiff_lzw", dpi=(1200, 1200)
            )

        print(f"✅ Saved figure {name} (PNG + CMYK TIFF) in {OUT_DIR}")

    # ======== Final figure: all Gaussian PDFs of residuals ======== #
    if results:
        xmin = min(r["mu"] - 4 * r["sd"] for r in results)
        xmax = max(r["mu"] + 4 * r["sd"] for r in results)
        xgrid = np.linspace(xmin, xmax, 900)

        figG, axG = plt.subplots(
            figsize=(4.33, 3.0), constrained_layout=True
        )
        colors = plt.get_cmap("tab10").colors

        for i, r in enumerate(results):
            axG.plot(
                xgrid,
                norm.pdf(xgrid, r["mu"], r["sd"]),
                color=colors[i % len(colors)],
                lw=1.3,
                label=f"{i + 1}–{r['label']}",
            )

        axG.set_xlabel("residuals (mGal)")
        axG.set_ylabel("probability density")
        axG.grid(False)
        axG.legend(
            loc="upper left",
            fontsize=7,
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            edgecolor="gray",
        )

        png_path = os.path.join(OUT_DIR, "combined_gaussians.png")
        tiff_path = os.path.join(OUT_DIR, "combined_gaussians.tiff")

        figG.savefig(png_path, dpi=1200, bbox_inches="tight")
        plt.close(figG)

        with Image.open(png_path) as im:
            im.convert("CMYK").save(
                tiff_path, compression="tiff_lzw", dpi=(1200, 1200)
            )

        print(
            f"✅ Saved figure combined_gaussians (PNG + CMYK TIFF) in {OUT_DIR}"
        )

    print(f"\nAll figures successfully written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
