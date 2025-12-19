import numpy as np
from scipy.stats import spearmanr


def spearman_per_feature(X1, X2):
    """
    Compute Spearman rank correlation for each feature between
    two feature matrices X1 and X2.

    Parameters
    ----------
    X1, X2 : array-like, shape (n_samples, n_features)
        Feature matrices for the SAME samples but different electronic levels.

    Returns
    -------
    rhos : np.ndarray, shape (n_features,)
        Spearman rho for each feature (column).
    """
    X1 = np.asarray(X1, float)
    X2 = np.asarray(X2, float)
    assert X1.shape == X2.shape, "X1 and X2 must have the same shape"
    n_samples, n_features = X1.shape

    rhos = np.empty(n_features, dtype=float)
    for j in range(n_features):
        # spearmanr returns (rho, pvalue)
        rho, _ = spearmanr(X1[:, j], X2[:, j])
        rhos[j] = rho
    return rhos


def summarize_spearman_by_block(rhos, block_slices):
    """
    Summarize Spearman correlations per block, given per-feature rhos
    and a dict of block slices/indices.

    Parameters
    ----------
    rhos : array-like, shape (n_features,)
        Spearman rho per feature (column).
    block_slices : dict
        Mapping block_name -> slice or index array defining which columns
        belong to that block.

    Returns
    -------
    summary : dict
        { block_name: { 'mean': ..., 'min': ..., 'max': ..., 'std': ..., 'n_feat': ... } }
    """
    rhos = np.asarray(rhos, float)
    summary = {}

    for name, sl in block_slices.items():
        # sl can be a slice, int, list, or np.ndarray
        if isinstance(sl, slice):
            block_rhos = rhos[sl]
        else:
            block_rhos = rhos[np.asarray(sl)]

        if block_rhos.size == 0:
            summary[name] = {
                'mean': np.nan,
                'min': np.nan,
                'max': np.nan,
                'std': np.nan,
                'n_feat': 0,
            }
        else:
            summary[name] = {
                'mean': float(np.nanmean(block_rhos)),
                'min': float(np.nanmin(block_rhos)),
                'max': float(np.nanmax(block_rhos)),
                'std': float(np.nanstd(block_rhos)),
                'n_feat': int(block_rhos.size),
            }

    return summary

