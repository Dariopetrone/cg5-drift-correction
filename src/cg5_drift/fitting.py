# -*- coding: utf-8 -*-
"""
Harmonic drift correction and fitting routines for CG-5 time series.
"""

import numpy as np


# ======== Memory-reset correction ======== #
def correct_memory_reset(
    g: np.ndarray,
    threshold: float,
    min_gap: int,
):
    """
    Detect and correct memory-reset jumps in a CG-5 time series.

    Parameters
    ----------
    g : array_like
        Gravity values (mGal).
    threshold : float
        Minimum absolute step (mGal) to flag a jump.
    min_gap : int
        Minimum number of samples between two jumps to consider them
        independent (compacts clusters of detected jumps).

    Returns
    -------
    g_corr : ndarray
        Gravity series with segment-wise offsets corrected.
    jumps : ndarray of int
        Indices where memory-reset jumps are detected.
    deltas : ndarray of float
        Applied offsets (mGal) for each corrected segment.
    """
    g = np.asarray(g, float)
    n = len(g)
    if n < 2:
        return g.copy(), np.array([], dtype=int), np.array([], dtype=float)

    diffs = np.diff(g)
    jumps = np.where(np.abs(diffs) > threshold)[0] + 1

    # Compact nearby jumps into a single index per cluster
    if jumps.size:
        compact = [jumps[0]]
        for j in jumps[1:]:
            if j - compact[-1] > min_gap:
                compact.append(j)
        jumps = np.array(compact, dtype=int)
    else:
        jumps = np.array([], dtype=int)

    # Build segment bounds and correct each segment to the previous one
    bounds = np.r_[0, jumps, n]
    g_corr = g.copy()
    deltas = []

    for si in range(1, len(bounds) - 1):
        e_prev = bounds[si]
        s_cur = bounds[si]
        e_cur = bounds[si + 1]

        # Offset required to match previous segment endpoint to new segment start
        delta = g_corr[e_prev - 1] - g_corr[s_cur]
        g_corr[s_cur:e_cur] += delta
        deltas.append(delta)

    return g_corr, jumps, np.array(deltas, float)


# ======== 4-harmonic weighted least-squares fit ======== #
def stima_ampiezza_fase_wls(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    sigma: np.ndarray,
):
    """
    Weighted least-squares fit of a sum of sinusoids plus constant offset.

    Model:
        y_fit(t) = sum_i A_i * sin(w_i * t + phi_i) + offset
    """
    y_mean = np.mean(y)
    y0 = y - y_mean

    # Design matrix: [sin(w1 t), cos(w1 t), ..., sin(wN t), cos(wN t), 1]
    cols = []
    for wi in w:
        cols.append(np.sin(wi * t))
        cols.append(np.cos(wi * t))
    X = np.column_stack(cols)
    X = np.hstack([X, np.ones((len(t), 1))])

    sigma = np.asarray(sigma, float)
    if np.any((sigma > 0) & np.isfinite(sigma)):
        sref = np.median(sigma[(sigma > 0) & np.isfinite(sigma)])
    else:
        sref = 1.0

    sigma_safe = np.where((~np.isfinite(sigma)) | (sigma <= 0), sref, sigma)
    wgt = 1.0 / (sigma_safe**2)

    WX = X * wgt[:, None]
    Wy = y0 * wgt

    beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)

    # Amplitude-phase conversion
    A = np.hypot(beta[:-1:2], beta[1:-1:2])
    phi = np.arctan2(beta[1:-1:2], beta[:-1:2])
    offset = beta[-1] + y_mean

    y_fit = np.sum(
        [Ai * np.sin(wi * t + ph) for Ai, wi, ph in zip(A, w, phi)],
        axis=0,
    ) + offset

    return y_fit, A, phi, offset, sigma_safe


def chi2_reduced_global(
    t_sec: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    w: np.ndarray,
):
    """
    Compute reduced chi-square for the global 4-sinusoid fit.
    """
    fit, A, phi, off, sigma_safe = stima_ampiezza_fase_wls(
        t_sec, data, w, sigma
    )
    resid = data - fit
    N = len(data)
    p = 2 * len(w) + 1  # parameters: 2 per harmonic + offset
    chi2 = np.sum((resid / sigma_safe) ** 2) / max(1, (N - p))
    return chi2, (fit, A, phi, off)


def best_omega_in_band(
    t_sec: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    w_current: np.ndarray,
    k: int,
    T_center: float,
    delta_T: float,
    n_steps: int = 400,
):
    """
    Grid search for the best angular frequency ω_k within a period band.
    """
    f_min = 1.0 / (T_center + delta_T)
    f_max = 1.0 / (T_center - delta_T)
    f_grid = np.linspace(f_min, f_max, n_steps)
    w_grid = 2 * np.pi * f_grid

    best = None
    for wk in w_grid:
        w_try = w_current.copy()
        w_try[k] = wk
        chi2, _ = chi2_reduced_global(t_sec, data, sigma, w_try)
        if (best is None) or (chi2 < best[0]):
            best = (chi2, wk)

    return best


def optimize_omegas_cd(
    t_sec: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    targets,
    w_init=None,
    n_steps: int = 400,
    max_iter: int = 6,
    tol: float = 1e-3,
):
    """
    Coordinate-descent optimization of the 4 angular frequencies ω_i.
    """
    if w_init is None:
        w = np.array([2 * np.pi * (1.0 / T0) for _, T0, _ in targets], float)
    else:
        w = np.array(w_init, float)

    chi2, _ = chi2_reduced_global(t_sec, data, sigma, w)
    prev_chi2 = chi2

    for _ in range(max_iter):
        improved = False
        for k, (_, T_center, delta_T) in enumerate(targets):
            chi2_k, wk_best = best_omega_in_band(
                t_sec, data, sigma, w, k, T_center, delta_T, n_steps=n_steps
            )
            if chi2_k + 1e-12 < chi2:
                w[k] = wk_best
                chi2 = chi2_k
                improved = True

        if (not improved) or abs(prev_chi2 - chi2) < tol:
            break
        prev_chi2 = chi2

    chi2_final, (fit, A, phi, off) = chi2_reduced_global(
        t_sec, data, sigma, w
    )
    return w, (fit, A, phi, off), chi2_final


def estimate_omega_uncertainties(
    t_sec: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    w_opt: np.ndarray,
    targets,
    n_steps: int = 160,
):
    """
    Estimate 1σ uncertainties on angular frequencies and periods.

    For each harmonic k, evaluate χ²_red(ω_k) on a grid within the
    prescribed period band and define the 1σ interval as the set of
    ω_k values for which Δχ² <= 1.
    """
    w_err = np.zeros_like(w_opt, dtype=float)

    for k, (_, T_center, delta_T) in enumerate(targets):
        f_min = 1.0 / (T_center + delta_T)
        f_max = 1.0 / (T_center - delta_T)
        f_grid = np.linspace(f_min, f_max, n_steps)
        w_grid = 2 * np.pi * f_grid

        chi2_vals = np.empty_like(w_grid)
        for i_w, wk in enumerate(w_grid):
            w_try = w_opt.copy()
            w_try[k] = wk
            chi2_k, _ = chi2_reduced_global(t_sec, data, sigma, w_try)
            chi2_vals[i_w] = chi2_k

        chi2_min = chi2_vals.min()
        mask = chi2_vals <= chi2_min + 1.0  # 1σ interval (Δχ² <= 1)

        if np.sum(mask) >= 2:
            w_in = w_grid[mask]
            w_err[k] = 0.5 * (w_in.max() - w_in.min())
        else:
            # If the interval cannot be resolved within the band, mark as NaN
            w_err[k] = np.nan

    return w_err
