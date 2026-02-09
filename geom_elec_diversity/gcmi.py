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
