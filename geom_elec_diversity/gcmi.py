"""
Taken from:
    https://github.com/robince/gcmi


Minimal Gaussian-Copula Mutual Information tools for:
  I(X_block ; y)

Assumes:
  - X_all: (N_samples, D_total)
  - y_array: (N_samples,)
  - block_slices: dict mapping block_name -> slice or index array
"""

import numpy as np
import scipy as sp
import scipy.special





# ------------------------------------------------------------------
# 0) Compatibility patch for NumPy >= 2.0 (original GCMI code uses np.float)
# ------------------------------------------------------------------
if not hasattr(np, "float"):
    np.float = float


# ------------------------------------------------------------------
# 1) Core GCMI (continuous–continuous) from Ince et al.
#    (only the parts you actually use)
# ------------------------------------------------------------------

def ctransform(x):
    """
    Copula transformation (empirical CDF).
    Input shape: (N_var, N_trials) or (N_trials,)
    Output shape: same, values in (0,1).
    """
    xi = np.argsort(np.atleast_2d(x))
    xr = np.argsort(xi)
    cx = (xr + 1).astype(np.float) / (xr.shape[-1] + 1)
    return cx


def copnorm(x):
    """
    Copula normalization:
      - empirical CDF (ctransform)
      - inverse standard normal CDF (ndtri)
    Returns standard-normal samples with same rank structure.
    """
    cx = sp.special.ndtri(ctransform(x))
    return cx


def mi_gg(x, y, biascorrect=True, demeaned=False):
    """
    Mutual information (MI) between two Gaussian variables in bits.

    x, y : arrays with shape (N_var, N_trials)
           (variables in rows, trials in columns)
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2D")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx + Nvary

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xy = np.vstack((x, y))
    if not demeaned:
        xy = xy - xy.mean(axis=1)[:, np.newaxis]

    Cxy = np.dot(xy, xy.T) / float(Ntrl - 1)

    # sub-covariances
    Cx = Cxy[:Nvarx, :Nvarx]
    Cy = Cxy[Nvarx:, Nvarx:]

    chCxy = np.linalg.cholesky(Cxy)
    chCx = np.linalg.cholesky(Cx)
    chCy = np.linalg.cholesky(Cy)

    # entropies in nats (normalization constants cancel in MI)
    HX = np.sum(np.log(np.diagonal(chCx)))
    HY = np.sum(np.log(np.diagonal(chCy)))
    HXY = np.sum(np.log(np.diagonal(chCxy)))

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi(
            (Ntrl - np.arange(1, Nvarxy + 1)).astype(np.float) / 2.0
        ) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0

        HX = HX - Nvarx * dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary * dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy * dterm - psiterms[:Nvarxy].sum()

    I = (HX + HY - HXY) / ln2  # in bits
    return I


def gcmi_cc(x, y):
    """
    Gaussian-Copula Mutual Information between two continuous variables.

    x : (N_varx, N_trials)
    y : (N_vary, N_trials)

    Steps:
      - copula-normalize x and y (rank -> Gaussian)
      - compute Gaussian MI (mi_gg) on the normalized data
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2D")

    Ntrl = x.shape[1]
    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # warnings about repeated values (optional)
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    for xi in range(Nvarx):
        if (np.unique(x[xi, :]).size / float(Ntrl)) < 0.9:
            # you can replace with warnings.warn if you want
            print("Warning: x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi, :]).size / float(Ntrl)) < 0.9:
            print("Warning: y has more than 10% repeated values")
            break

    # copula normalization
    cx = copnorm(x)
    cy = copnorm(y)

    # parametric Gaussian MI
    I = mi_gg(cx, cy, biascorrect=True, demeaned=True)
    return I


# ------------------------------------------------------------------
# 2) Helpers to use PRECOMPUTED feature matrix X_all
# ------------------------------------------------------------------

def _jitter_rows_for_gcmi(A, eps=1e-10, seed=0):
    """
    A : array (N_var, N_trials)

    GCMI expects each variable (row) to have mostly unique values across trials.
    We add tiny noise if a row has exact repeats, to break ties without
    changing the statistics in any meaningful way.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float)
    n_var, n_trials = A.shape

    for i in range(n_var):
        row = A[i, :]
        if np.unique(row).size < n_trials:
            A[i, :] = row + eps * rng.standard_normal(size=n_trials)
    return A


def prepare_gcmi_from_precomputed(
    X_all,
    y_array,
    col_slice,
    jitter_eps=1e-10,
    seed=0,
):
    """
    Prepare (X_gcmi, y_gcmi) in the shape gcmi_cc expects, from
    a PRECOMPUTED feature matrix.

    Inputs:
      X_all    : (N_samples, D_total)
      y_array  : 1D array-like of length N_samples
      col_slice: slice or index array selecting the columns for this block

    Returns:
      X_gcmi : (N_var, N_trials) = (D_block, N_samples)
      y_gcmi : (1, N_trials)     = (1, N_samples)
    """
    X_all = np.asarray(X_all, float)
    y = np.asarray(y_array, float).ravel()

    if X_all.shape[0] != y.shape[0]:
        raise ValueError("X_all rows and y_array length mismatch")

    # select the block columns
    X_sel = X_all[:, col_slice]     # (N_samples, D_block)

    # gcmi_cc expects (variables, trials) = (D_block, N_samples)
    X_gcmi = X_sel.T                # (D_block, N_samples)
    y_gcmi = y.reshape(1, -1)       # (1, N_samples)

    # jitter to avoid exact ties along trials
    X_gcmi = _jitter_rows_for_gcmi(X_gcmi, eps=jitter_eps, seed=seed)
    y_gcmi = _jitter_rows_for_gcmi(y_gcmi, eps=jitter_eps, seed=seed + 1)

    return X_gcmi, y_gcmi


def gcmi_from_precomputed(
    X_all,
    y_array,
    block_slices,
    block_key,
    jitter_eps=1e-10,
    seed=0,
    ):
    """
    Compute I( X_block ; y ) using gcmi_cc, where X_block is a subset of
    columns from the PRECOMPUTED feature matrix X_all.

    Inputs:
      X_all        : (N_samples, D_total)
      y_array      : 1D array of property values (N_samples,)
      block_slices : dict mapping block_name -> slice (or index array)
      block_key    : which block to use, e.g.
                       'all', 'geom_all', 'elec_all',
                       'composition', 'shape', 'guess_frontier_energies', ...

    Returns:
      I_bits : float, mutual information in bits.
    """
    if block_key not in block_slices:
        raise KeyError(
            f"Unknown block key '{block_key}'. Available keys: {list(block_slices.keys())}"
        )

    col_slice = block_slices[block_key]

    X_gcmi, y_gcmi = prepare_gcmi_from_precomputed(
        X_all,
        y_array,
        col_slice=col_slice,
        jitter_eps=jitter_eps,
        seed=seed,
    )

    I_bits = gcmi_cc(X_gcmi, y_gcmi)
    return float(I_bits)


def ent_g(x, biascorrect=True):
    """
    Entropy of a Gaussian variable in bits

    H = ent_g(x) returns the entropy of a (possibly
    multidimensional) Gaussian variable x with bias correction.
    Columns of x correspond to samples, rows to dimensions/variables.
    (Samples last axis)
    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    # demean data
    x = x - x.mean(axis=1)[:, np.newaxis]
    # covariance
    C = np.dot(x, x.T) / float(Ntrl - 1)
    chC = np.linalg.cholesky(C)

    # entropy in nats
    HX = np.sum(np.log(np.diagonal(chC))) + 0.5 * Nvarx * (np.log(2 * np.pi) + 1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi(
            (Ntrl - np.arange(1, Nvarx + 1).astype(np.float)) / 2.0
        ) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HX = HX - Nvarx * dterm - psiterms.sum()

    # convert to bits
    return HX / ln2


def gc_entropy(x, biascorrect=True, jitter_eps=1e-10, seed=0):
    """
    Gaussian-Copula entropy H(X) in bits for a continuous variable X.

    x : array, shape (N_var, N_trials) or (N_trials,)
        Variables in rows, trials in columns (same convention as gcmi_cc).

    Steps:
      - optional tiny jitter across trials to break exact ties (as in gcmi_from_precomputed)
      - copula-normalize x (rank -> Gaussian)
      - compute Gaussian entropy via ent_g on the normalized data

    Returns:
      H_bits : float, entropy in bits.
    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be at most 2D")

    # jitter along trials, as in _jitter_rows_for_gcmi
    x = _jitter_rows_for_gcmi(x, eps=jitter_eps, seed=seed)

    # copula normalization (same as gcmi_cc)
    cx = copnorm(x)

    # Gaussian entropy with the same bias correction scheme
    H_bits = ent_g(cx, biascorrect=biascorrect)
    return float(H_bits)


def gc_entropy_from_precomputed(
    X_all,
    block_slices,
    block_key,
    jitter_eps=1e-10,
    seed=0,
):
    """
    Gaussian-Copula entropy H(X_block) in bits, where X_block is a subset of
    columns from the PRECOMPUTED feature matrix X_all.

    Inputs:
      X_all        : (N_samples, D_total)
      block_slices : dict mapping block_name -> slice (or index array)
      block_key    : which block to use (same keys as in gcmi_from_precomputed)

    Returns:
      H_bits : float, entropy in bits.
    """
    if block_key not in block_slices:
        raise KeyError(
            f"Unknown block key '{block_key}'. Available keys: {list(block_slices.keys())}"
        )

    col_slice = block_slices[block_key]

    # reuse your precomputed-prep helper
    X_gcmi, _ = prepare_gcmi_from_precomputed(
        X_all,
        y_array=np.zeros(X_all.shape[0]),  # dummy, not used
        col_slice=col_slice,
        jitter_eps=jitter_eps,
        seed=seed,
    )
    # y_gcmi is ignored; we only want X_gcmi
    H_bits = gc_entropy(X_gcmi, biascorrect=True, jitter_eps=0.0, seed=seed)
    return float(H_bits)


def gcmi_between_blocks(                                                                                                          
     X_all,                                                                                                                        
     block_slices,                                                                                                                 
     block_key_x,                                                                                                                  
     block_key_y,                                                                                                                  
     jitter_eps=1e-10,                                                                                                             
     seed=0,                                                                                                                       
 ):                                                                                                                                
     """                                                                                                                           
     Gaussian-Copula MI between two feature blocks:                                                                                
         I( X_block_x ; X_block_y )                                                                                                
                                                                                                                                   
     Inputs:                                                                                                                       
       X_all        : (N_samples, D_total)                                                                                         
       block_slices : dict mapping block_name -> slice or index array                                                              
       block_key_x  : name of first block, e.g. 'geom_all'                                                                         
       block_key_y  : name of second block, e.g. 'elec_all'                                                                        
                                                                                                                                   
     Returns:                                                                                                                      
       I_bits : float, mutual information in bits.                                                                                 
     """                                                                                                                           
     if block_key_x not in block_slices:                                                                                           
         raise KeyError(                                                                                                           
             f"Unknown block key '{block_key_x}'. Available keys: {list(block_slices.keys())}"                                     
         )                                                                                                                         
     if block_key_y not in block_slices:                                                                                           
         raise KeyError(                                                                                                           
             f"Unknown block key '{block_key_y}'. Available keys: {list(block_slices.keys())}"                                     
         )                                                                                                                         
                                                                                                                                   
     col_slice_x = block_slices[block_key_x]                                                                                       
     col_slice_y = block_slices[block_key_y]                                                                                       
                                                                                                                                   
     X_all = np.asarray(X_all, float)                                                                                              
                                                                                                                                   
     # select columns for each block: (N_samples, D_block)                                                                         
     Xx = X_all[:, col_slice_x]                                                                                                    
     Xy = X_all[:, col_slice_y]                                                                                                    
                                                                                                                                   
     # gcmi_cc expects (variables, trials) = (D_block, N_samples)                                                                  
     Xx_gcmi = Xx.T   # (D_x, N_samples)                                                                                           
     Xy_gcmi = Xy.T   # (D_y, N_samples)                                                                                           
                                                                                                                                   
     # jitter rows to avoid exact ties, same logic as prepare_gcmi_from_precomputed                                                
     Xx_gcmi = _jitter_rows_for_gcmi(Xx_gcmi, eps=jitter_eps, seed=seed)                                                           
     Xy_gcmi = _jitter_rows_for_gcmi(Xy_gcmi, eps=jitter_eps, seed=seed + 1)                                                       
                                                                                                                                   
     # Gaussian-copula MI between the two blocks                                                                                   
     I_bits = gcmi_cc(Xx_gcmi, Xy_gcmi)                                                                                            
     return float(I_bits)                     




def gcmi_from_precomputed_blocks_union(
    X_all,
    y_array,
    block_slices,
    block_keys,
    jitter_eps=1e-10,
    seed=0,
):
    """
    Compute I( [X_block1, X_block2, ...] ; y ) using gcmi_cc,
    where [X_block1, X_block2, ...] is the concatenation of several
    column blocks from X_all.

    Inputs
    ------
    X_all        : (N_samples, D_total)
    y_array      : 1D array of property values (N_samples,)
    block_slices : dict mapping block_name -> slice or index array
    block_keys   : iterable of block names to combine, e.g.
                   ['geom_all', 'spahm'] or ('composition','shape','slatm')

    Returns
    -------
    I_bits : float, mutual information in bits.
    """
    X_all = np.asarray(X_all, float)
    y = np.asarray(y_array, float).ravel()

    # make sure block_keys is a list
    if isinstance(block_keys, (str, bytes)):
        block_keys = [block_keys]
    else:
        block_keys = list(block_keys)

    if len(block_keys) == 0:
        raise ValueError("block_keys must contain at least one block name")

    # collect all column indices for the union of requested blocks
    col_indices_list = []
    for key in block_keys:
        if key not in block_slices:
            raise KeyError(
                f"Unknown block key '{key}'. Available keys: {list(block_slices.keys())}"
            )
        sl = block_slices[key]
        if isinstance(sl, slice):
            idx = np.arange(sl.start, sl.stop)
        else:
            idx = np.asarray(sl, int)
        col_indices_list.append(idx)

    col_indices = np.concatenate(col_indices_list, axis=0)

    # Prepare data for GCMI: (D_block_union, N_samples)
    X_sel = X_all[:, col_indices]          # (N, D_union)
    X_gcmi = X_sel.T                       # (D_union, N)
    y_gcmi = y.reshape(1, -1)              # (1, N)

    # Jitter rows to handle ties
    X_gcmi = _jitter_rows_for_gcmi(X_gcmi, eps=jitter_eps, seed=seed)
    y_gcmi = _jitter_rows_for_gcmi(y_gcmi, eps=jitter_eps, seed=seed + 1)

    I_bits = gcmi_cc(X_gcmi, y_gcmi)
    return float(I_bits)


def gcmi_per_feature(
    X,
    y_array,
    jitter_eps=1e-10,
    seed=0,
):
    """
    Compute Gaussian-Copula MI I(x_j ; y) for each individual feature column x_j.

    Inputs
    ------
    X        : array-like, shape (N_samples, D_features)
               Feature matrix (same format as for your plots).
    y_array  : array-like, shape (N_samples,)
               Scalar property values.
    jitter_eps : float
               Magnitude of jitter noise to break ties in ranks.
    seed       : int
               Random seed for jitter.

    Returns
    -------
    I_vec : np.ndarray, shape (D_features,)
            I(x_j ; y) in bits for each feature j.
    """
    X = np.asarray(X, float)
    y = np.asarray(y_array, float).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows and y_array length mismatch")

    N_samples, D_features = X.shape

    # y_gcmi is shared across features
    y_gcmi = y.reshape(1, -1)  # (1, N_samples)
    # jitter y once
    y_gcmi = _jitter_rows_for_gcmi(y_gcmi, eps=jitter_eps, seed=seed + 1)

    I_vec = np.empty(D_features, dtype=float)

    for j in range(D_features):
        # single feature -> shape (1, N_samples)
        x_j = X[:, j].reshape(1, -1)
        # jitter x_j
        x_j = _jitter_rows_for_gcmi(x_j, eps=jitter_eps, seed=seed + j + 2)

        # Gaussian-copula MI (bias-corrected) between x_j and y
        I_vec[j] = gcmi_cc(x_j, y_gcmi)

    return I_vec





#############################################

import numpy as np


# ============================================================
# PCA-reduction helpers for HUGE blocks (stable, copula-based)
# ============================================================

#def _as_var_trial_matrix(X):
#    """
#    Ensure X is (n_var, n_trials).
#    Accepts:
#      - (n_trials,) -> (1, n_trials)
#      - (n_trials, n_var) -> transpose to (n_var, n_trials)
#      - (n_var, n_trials) -> unchanged
#    """
#    X = np.asarray(X, float)
#    if X.ndim == 1:
#        return X.reshape(1, -1)
#    if X.ndim != 2:
#        raise ValueError("X must be 1D or 2D")
#    # Heuristic: if rows are trials and cols are variables, transpose
#    # In your project: feature matrices are usually (N_samples, D_features),
#    # while gcmi expects (D, N). So we transpose if rows > cols.
#    if X.shape[0] >= X.shape[1]:
#        # likely (n_trials, n_var) -> transpose
#        return X.T
#    return X

def _to_var_trial(X, assume_trials_first=True):
    """
    Convert to (n_var, n_trials) for GCMI.

    If assume_trials_first=True (default), interpret X as (n_trials, n_var)
    and transpose.

    If assume_trials_first=False, interpret X as already (n_var, n_trials).
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        # 1D -> treat as a single variable measured over trials
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("X must be 1D or 2D")

    return X.T if assume_trials_first else X


def _drop_constant_rows(X, eps=1e-12):
    """Drop rows (variables) with ~zero variance across trials."""
    X = np.asarray(X, float)
    if X.size == 0:
        return X, np.zeros((0,), dtype=bool)
    sd = X.std(axis=1)
    keep = sd > eps
    return X[keep, :], keep


def _pca_whiten_gaussian_rows(G, var_thresh=0.99, max_dims=None, whiten=True, eps=1e-12):
    """
    PCA (via SVD) on a *Gaussian* matrix G of shape (n_var, n_trials),
    where trials are samples, variables are dimensions.

    Returns:
      Z: (k, n_trials) reduced (and optionally whitened) representation
      meta: dict with kept dims and explained variance
    """
    G = np.asarray(G, float)
    if G.ndim != 2:
        raise ValueError("G must be 2D (n_var, n_trials)")
    n_var, n_trials = G.shape
    if n_var == 0:
        return np.zeros((0, n_trials), float), {"kept_dims": 0, "var_explained": 0.0}

    # Center across trials (demean each variable)
    Gc = G - G.mean(axis=1, keepdims=True)

    # SVD of (n_var x n_trials)
    # We want PCs in variable-space; SVD is stable:
    # Gc = U S V^T, with U (n_var,k), V (n_trials,k)
    U, S, Vt = np.linalg.svd(Gc, full_matrices=False)

    # eigenvalues of covariance in variable-space
    # Cov = Gc Gc^T / (n_trials - 1) -> eigvals = S^2/(n_trials-1)
    denom = max(n_trials - 1, 1)
    eig = (S * S) / denom
    total = float(eig.sum() + eps)
    cum = np.cumsum(eig) / total

    k = int(np.searchsorted(cum, var_thresh) + 1)
    k = max(1, min(k, eig.size))

    if max_dims is not None:
        k = min(k, int(max_dims))

    # Reduced scores: PC coordinates across trials are proportional to V^T
    # Standard PCA scores: T = U^T Gc = S V^T -> shape (k, n_trials)
    Z = (S[:k, None] * Vt[:k, :])

    if whiten:
        Z = Z / np.sqrt(eig[:k] + eps)[:, None]

    var_expl = float(eig[:k].sum() / (eig.sum() + eps))
    meta = {"kept_dims": int(k), "var_explained": var_expl}
    return Z, meta


def stable_gaussian_repr_with_pca(
    X_block,
    jitter_eps=1e-10,
    seed=0,
    var_thresh=0.99,
    max_dims=None,
    whiten=True,
    assume_trials_first=True,
):
    """
    Build a *stable* representation for a (possibly huge) block:
      1) convert to (n_var, n_trials)
      2) jitter to break ties
      3) Gaussian-copula normalize (copnorm)  -> Gaussian rows
      4) drop constant rows
      5) PCA to var_thresh (+ optional max_dims) and optional whitening

    Returns:
      Z : (k, n_trials) Gaussian reduced representation
      meta : dict
    """
    #X = _as_var_trial_matrix(X_block)
    #X = _jitter_rows_for_gcmi(X, eps=jitter_eps, seed=seed)

    ## Gaussian-copula normalization (already in gcmi.py)
    #G = copnorm(X)

	# ALWAYS convert to (n_var, n_trials) for GCMI
    X_vt = _to_var_trial(X_block, assume_trials_first=assume_trials_first)

	# jitter across trials
    X_vt = _jitter_rows_for_gcmi(X_vt, eps=jitter_eps, seed=seed)

	# copula-normalize -> Gaussian rows
    G = copnorm(X_vt)	

    # Drop constant rows (rare but can happen after copnorm with many ties)
    G, keep = _drop_constant_rows(G)

    # PCA on Gaussian rows
    Z, pmeta = _pca_whiten_gaussian_rows(
        G,
        var_thresh=var_thresh,
        max_dims=max_dims,
        whiten=whiten,
    )
    meta = {
        "input_dims": int(X_vt.shape[0]),
        "kept_nonconst_dims": int(G.shape[0]),
        "kept_mask_rows": keep,
        **pmeta,
    }
    return Z, meta


# ============================================================
# Precomputed X_all + block_slices APIs (with PCA reduction)
# ============================================================

def _cols_from_slice_or_idx(sl):
    if isinstance(sl, slice):
        return np.arange(sl.start, sl.stop, dtype=int)
    return np.asarray(sl, dtype=int)


def _get_block_matrix_trials_first(X_all, block_slices, block_key):
    if block_key not in block_slices:
        raise KeyError(
            f"Unknown block key '{block_key}'. Available keys: {list(block_slices.keys())}"
        )
    X_all = np.asarray(X_all, float)
    cols = _cols_from_slice_or_idx(block_slices[block_key])
    return X_all[:, cols]  # (n_trials, d_block)


def gc_entropy_from_precomputed_pca(
    X_all,
    block_slices,
    block_key,
    jitter_eps=1e-10,
    seed=0,
    var_thresh=0.99,
    max_dims=None,
    whiten=True,
    biascorrect=True,
):
    """
    H(X_block) in bits, but using copula + PCA reduction for huge blocks.
    """
    Xblk = _get_block_matrix_trials_first(X_all, block_slices, block_key)  # (N, D)
    Z, meta = stable_gaussian_repr_with_pca(
        Xblk,
        jitter_eps=jitter_eps,
        seed=seed,
        var_thresh=var_thresh,
        max_dims=max_dims,
        whiten=whiten,
    )
    # Z is already Gaussian (copnorm), so entropy is Gaussian entropy
    H_bits = ent_g(Z, biascorrect=biascorrect)
    return float(H_bits), meta


def gcmi_between_blocks_pca(
    X_all,
    block_slices,
    block_key_x,
    block_key_y,
    jitter_eps=1e-10,
    seed=0,
    var_thresh=0.99,
    max_dims=None,
    whiten=True,
    biascorrect=True,
):
    """
    I(X_block_x ; X_block_y) in bits, using copula + PCA reduction on each block.
    """
    Xx = _get_block_matrix_trials_first(X_all, block_slices, block_key_x)  # (N, Dx)
    Xy = _get_block_matrix_trials_first(X_all, block_slices, block_key_y)  # (N, Dy)

    Zx, mx = stable_gaussian_repr_with_pca(
        Xx,
        jitter_eps=jitter_eps,
        seed=seed,
        var_thresh=var_thresh,
        max_dims=max_dims,
        whiten=whiten,
    )
    Zy, my = stable_gaussian_repr_with_pca(
        Xy,
        jitter_eps=jitter_eps,
        seed=seed + 1,
        var_thresh=var_thresh,
        max_dims=max_dims,
        whiten=whiten,
    )

    # Zx, Zy are Gaussian => use Gaussian MI directly (mi_gg is in gcmi.py)
    I_bits = mi_gg(Zx, Zy, biascorrect=biascorrect, demeaned=True)
    return float(I_bits), {"x": mx, "y": my}


def gcmi_union_vs_block_pca(
    X_all,
    block_slices,
    union_block_keys,
    other_block_key,
    jitter_eps=1e-10,
    seed=0,
    var_thresh=0.99,
    max_dims=None,
    whiten=True,
    biascorrect=True,
):
    """
    I([union of blocks] ; other_block) using copula + PCA reduction.

    union_block_keys: iterable of block names to concatenate in feature-space first.
    """
    if isinstance(union_block_keys, (str, bytes)):
        union_block_keys = [union_block_keys]
    union_block_keys = list(union_block_keys)
    if len(union_block_keys) == 0:
        raise ValueError("union_block_keys must not be empty")

    # Build union columns
    cols_all = []
    for k in union_block_keys:
        if k not in block_slices:
            raise KeyError(f"Unknown block key '{k}'. Available: {list(block_slices.keys())}")
        cols_all.append(_cols_from_slice_or_idx(block_slices[k]))
    cols = np.unique(np.concatenate(cols_all))
    X_union = np.asarray(X_all, float)[:, cols]  # (N, D_union)

    X_other = _get_block_matrix_trials_first(X_all, block_slices, other_block_key)  # (N, D_other)

    Zu, mu = stable_gaussian_repr_with_pca(
        X_union, jitter_eps=jitter_eps, seed=seed,
        var_thresh=var_thresh, max_dims=max_dims, whiten=whiten
    )
    Zo, mo = stable_gaussian_repr_with_pca(
        X_other, jitter_eps=jitter_eps, seed=seed + 1,
        var_thresh=var_thresh, max_dims=max_dims, whiten=whiten
    )

    I_bits = mi_gg(Zu, Zo, biascorrect=biascorrect, demeaned=True)
    return float(I_bits), {"union": mu, "other": mo}

import numpy as np

def external_descriptor_to_pca(
    A,
    var_thresh=0.99,
    max_dims=None,
    whiten=False,
    center=True,
    eps=1e-12,
    return_model=False,
):
    """
    Reduce an external descriptor A using PCA.

    Parameters
    ----------
    A : array-like, shape (N_samples, D_dims)
        External descriptor matrix (rows=samples, cols=dimensions).
    var_thresh : float
        Keep the smallest number of PCs whose cumulative explained variance >= var_thresh.
    max_dims : int or None
        Optional hard cap on #PCs kept.
    whiten : bool
        If True, scale PCs to have unit variance (useful sometimes, optional).
    center : bool
        If True, subtract column means before PCA.
    return_model : bool
        If True, also return a model dict to transform future A matrices consistently.

    Returns
    -------
    A_pca : ndarray, shape (N_samples, K)
    meta  : dict
    model : dict (only if return_model=True)
    """
    A = np.asarray(A, float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array with shape (N_samples, D_dims).")

    n, d = A.shape
    if n < 2 or d == 0:
        A_pca = np.zeros((n, 0), float)
        meta = {"kept_dims": 0, "var_explained": 0.0, "n_samples": int(n), "n_input_dims": int(d)}
        if return_model:
            model = {"mean": A.mean(axis=0, keepdims=True) if center else None,
                     "components": np.zeros((0, d), float),
                     "eigvals": np.zeros((0,), float),
                     "whiten": bool(whiten),
                     "center": bool(center),
                     "eps": float(eps)}
            return A_pca, meta, model
        return A_pca, meta

    mu = A.mean(axis=0, keepdims=True) if center else np.zeros((1, d), float)
    Ac = A - mu

    # SVD: Ac = U S Vt
    U, S, Vt = np.linalg.svd(Ac, full_matrices=False)

    # eigenvalues of covariance
    denom = max(n - 1, 1)
    eig = (S * S) / denom

    total = float(eig.sum() + eps)
    cum = np.cumsum(eig) / total
    k = int(np.searchsorted(cum, var_thresh) + 1)
    k = max(1, min(k, eig.size))
    if max_dims is not None:
        k = min(k, int(max_dims))

    # scores: (N, k)
    A_pca = U[:, :k] * S[:k][None, :]

    if whiten:
        A_pca = A_pca / np.sqrt(eig[:k] + eps)[None, :]

    meta = {
        "kept_dims": int(k),
        "var_explained": float(eig[:k].sum() / (eig.sum() + eps)),
        "n_samples": int(n),
        "n_input_dims": int(d),
        "whiten": bool(whiten),
        "center": bool(center),
    }

    if return_model:
        model = {
            "mean": mu,
            "components": Vt[:k, :],   # (k, d)
            "eigvals": eig[:k],
            "whiten": bool(whiten),
            "center": bool(center),
            "eps": float(eps),
        }
        return A_pca, meta, model

    return A_pca, meta



def id_pca_on_copula(
    X,
    var_thresh=0.99,
    max_dims=None,
    jitter_eps=1e-10,
    seed=0,
    whiten=False,
):
    """
    Effective dimension via PCA in *copula-Gaussian space*, consistent with GCMI.

    Parameters
    ----------
    X : array, shape (N_samples, D_dims)
        Samples-first matrix.
    var_thresh : float
        Keep smallest #PCs reaching this cumulative variance in Gaussian space.
    max_dims : int or None
        Optional cap on kept PCs (useful for huge descriptors).
    whiten : bool
        If True, whiten PCs (not necessary for ID; keep False by default).

    Returns
    -------
    meta : dict
        Includes kept_dims, var_explained, and input dims.
    """
    # stable_gaussian_repr_with_pca expects samples-first by default (assume_trials_first=True)
    Z, meta = stable_gaussian_repr_with_pca(
        X,
        jitter_eps=jitter_eps,
        seed=seed,
        var_thresh=var_thresh,
        max_dims=max_dims,
        whiten=whiten,
        assume_trials_first=True,   # X is (N_samples, D)
    )

    # meta already contains: input_dims, kept_nonconst_dims, kept_dims, var_explained, ...
    return {
        "kept_dims": int(meta["kept_dims"]),
        "var_explained": float(meta["var_explained"]),
        "n_samples": int(np.asarray(X).shape[0]),
        "n_input_dims": int(np.asarray(X).shape[1]),
        "kept_nonconst_dims": int(meta["kept_nonconst_dims"]),
    }
