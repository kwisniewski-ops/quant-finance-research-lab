"""Numerical linear algebra for covariance and correlation matrices.

Estimated covariance matrices are routinely indefinite (missing data,
pairwise estimation, stress overrides) and ill-conditioned when the number
of assets approaches the number of observations. This module provides
Higham-style positive-semidefinite repair, PSD checks, Ledoit-Wolf
constant-correlation shrinkage, and a Cholesky factorization that degrades
gracefully.

References
----------
Higham, N. J. (2002). "Computing the Nearest Correlation Matrix — A Problem
from Finance." *IMA Journal of Numerical Analysis*, 22(3), 329-343.

Ledoit, O. and Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance
Matrix." *Journal of Portfolio Management*, 30(4), 110-119.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["nearest_psd", "is_psd", "ledoit_wolf_shrinkage", "safe_cholesky"]


def _validate_square(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square 2-D matrix, got shape {A.shape}")
    return A


def is_psd(A: np.ndarray, tol: float = 1e-10) -> bool:
    r"""Check whether a symmetric matrix is positive semidefinite.

    Parameters
    ----------
    A : np.ndarray
        Square matrix (symmetrized internally as
        :math:`(A + A^\top)/2` before the eigenvalue test).
    tol : float, optional
        Eigenvalues greater than ``-tol`` count as non-negative.
        Default ``1e-10``.

    Returns
    -------
    bool
        ``True`` if the smallest eigenvalue exceeds ``-tol``.
    """
    A = _validate_square(A)
    sym = 0.5 * (A + A.T)
    eigvals = np.linalg.eigvalsh(sym)
    return bool(eigvals.min() >= -tol)


def nearest_psd(A: np.ndarray, eps: float = 1e-10, corr: bool = False) -> np.ndarray:
    r"""Project a symmetric matrix onto the positive-semidefinite cone.

    Higham-style eigenvalue clipping: symmetrize, decompose
    :math:`A = Q \Lambda Q^\top`, floor the spectrum at ``eps``, and
    reconstruct:

    .. math::

        A_+ = Q \max(\Lambda, \varepsilon) Q^\top .

    This is the Frobenius-norm projection onto the PSD cone (the first step
    of Higham's 2002 alternating-projections algorithm). With
    ``corr=True`` the unit diagonal is additionally restored after each
    clip, alternating a few times between the two constraint sets — a light
    version of Higham's full algorithm suitable for correlation matrices.

    Parameters
    ----------
    A : np.ndarray
        Square (nearly) symmetric matrix.
    eps : float, optional
        Eigenvalue floor. Default ``1e-10``.
    corr : bool, optional
        If ``True``, treat ``A`` as a correlation matrix and preserve the
        unit diagonal. Default ``False``.

    Returns
    -------
    np.ndarray
        PSD matrix close to ``A`` in Frobenius norm; exactly symmetric.

    Raises
    ------
    ValueError
        If ``A`` is not square.
    """
    A = _validate_square(A)
    out = 0.5 * (A + A.T)

    n_sweeps = 5 if corr else 1
    for _ in range(n_sweeps):
        eigvals, eigvecs = np.linalg.eigh(out)
        if eigvals.min() >= eps and not corr:
            break
        clipped = np.maximum(eigvals, eps)
        out = (eigvecs * clipped) @ eigvecs.T
        out = 0.5 * (out + out.T)
        if corr:
            # Restore the unit diagonal (rescale rows/cols).
            d = np.sqrt(np.clip(np.diag(out), eps, None))
            out = out / np.outer(d, d)
            np.fill_diagonal(out, 1.0)
            if np.linalg.eigvalsh(out).min() >= -eps:
                break
    return out


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    r"""Ledoit-Wolf shrinkage toward the constant-correlation target.

    Computes the optimal convex combination

    .. math::

        \hat{\Sigma}_{\text{shrunk}} = \delta^\ast F + (1 - \delta^\ast) S,

    where :math:`S` is the sample covariance matrix and :math:`F` the
    constant-correlation target (sample variances on the diagonal,
    off-diagonals implied by the average sample correlation
    :math:`\bar{r}`). The intensity minimizes the expected Frobenius loss:

    .. math::

        \delta^\ast = \max\!\left(0, \min\!\left(1,
            \frac{\hat{\pi} - \hat{\rho}}{\hat{\gamma}\, T}\right)\right),

    with :math:`\hat{\pi}` the sum of asymptotic variances of the entries
    of :math:`S`, :math:`\hat{\rho}` the covariance between the estimation
    errors of :math:`S` and :math:`F`, and
    :math:`\hat{\gamma} = \lVert F - S \rVert_F^2`
    (Ledoit and Wolf 2004).

    Parameters
    ----------
    returns : pd.DataFrame
        Return observations, rows = dates, columns = assets. NaN rows are
        dropped.

    Returns
    -------
    tuple of (np.ndarray, float)
        ``(shrunk_cov, shrinkage_intensity)`` — the shrunk covariance as an
        ``(n, n)`` array and :math:`\delta^\ast \in [0, 1]`.

    Raises
    ------
    ValueError
        If fewer than 2 observations or fewer than 2 assets remain.
    """
    X = returns.dropna().to_numpy(dtype=float)
    t, n = X.shape
    if t < 2 or n < 2:
        raise ValueError(
            f"Need at least 2 observations and 2 assets, got shape {(t, n)}"
        )

    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / t

    var = np.diag(S).copy()
    std = np.sqrt(var)
    corr = S / np.outer(std, std)
    r_bar = (corr.sum() - n) / (n * (n - 1))

    F = r_bar * np.outer(std, std)
    np.fill_diagonal(F, var)

    # pi-hat: sum of asymptotic variances of the entries of S.
    Y = Xc**2
    pi_mat = (Y.T @ Y) / t - S**2
    pi_hat = pi_mat.sum()

    # rho-hat: diagonal part plus the constant-correlation cross term.
    theta_ii = (Xc**3).T @ Xc / t - var[:, None] * S
    rho_off = (std[None, :] / std[:, None]) * theta_ii \
        + (std[:, None] / std[None, :]) * theta_ii.T
    np.fill_diagonal(rho_off, 0.0)
    rho_hat = np.trace(pi_mat) + 0.5 * r_bar * rho_off.sum()

    gamma_hat = np.linalg.norm(S - F, "fro") ** 2

    if gamma_hat <= 0.0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa / t, 0.0, 1.0))

    shrunk = delta * F + (1.0 - delta) * S
    return shrunk, delta


def safe_cholesky(A: np.ndarray) -> np.ndarray:
    r"""Cholesky factorization with a PSD-repair fallback.

    Attempts ``np.linalg.cholesky``; on failure (matrix indefinite or
    numerically singular), projects ``A`` onto the PSD cone via
    :func:`nearest_psd`, adds a small diagonal jitter scaled to the matrix,
    and retries.

    Parameters
    ----------
    A : np.ndarray
        Square (nearly) symmetric matrix.

    Returns
    -------
    np.ndarray
        Lower-triangular ``L`` with :math:`L L^\top \approx A` (exactly
        equal when ``A`` is already positive definite).

    Raises
    ------
    ValueError
        If ``A`` is not square.
    np.linalg.LinAlgError
        If factorization fails even after repair (pathological input).
    """
    A = _validate_square(A)
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        pass

    repaired = nearest_psd(A)
    scale = max(float(np.trace(repaired)) / repaired.shape[0], 1.0)
    jitter = 1e-12 * scale
    for _ in range(8):
        try:
            return np.linalg.cholesky(
                repaired + jitter * np.eye(repaired.shape[0])
            )
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError(
        "safe_cholesky: factorization failed even after PSD repair"
    )
