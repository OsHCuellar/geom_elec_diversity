import numpy as np
from .gcmi import _jitter_rows_for_gcmi, gcmi_cc

def select_features_by_mi(X, I_features, threshold):
    """
    Selects feature columns whose MI >= threshold.

    Parameters
    ----------
    X : array-like, shape (N_samples, D_features)
        Full descriptor matrix.
    I_features : array-like, shape (D_features,)
        Mutual information of each feature.
    threshold : float
        Minimum MI required to keep a feature.

    Returns
    -------
    X_selected : array (N_samples, D_selected)
        Features whose MI >= threshold.
    selected_indices : array (D_selected,)
        Indices of selected features.
    I_selected : array (D_selected,)
        MI values of selected features.
    """
    I_features = np.asarray(I_features, float)
    X = np.asarray(X, float)

    if X.shape[1] != I_features.size:
        raise ValueError("X.shape[1] must match I_features.size")

    # Boolean mask of features to keep
    mask = I_features >= threshold

    selected_indices = np.where(mask)[0]
    X_selected = X[:, mask]
    I_selected = I_features[mask]

    return X_selected, selected_indices, I_selected



def greedy_forward_gcmi(
    X,
    y_array,
    max_features=20,
    preselect_k=None,
    jitter_eps=1e-10,
    seed=0,
    min_delta_MI=1e-3,
):
    """
    Greedy forward selection using Gaussian-Copula MI.

    Parameters
    ----------
    X : array-like, shape (N_samples, D_features)
        Full feature matrix.
    y_array : array-like, shape (N_samples,)
        Scalar property values.
    max_features : int
        Maximum number of features to select.
    preselect_k : int or None
        If not None, preselect only the top-k features by univariate MI
        and run greedy selection on that reduced set.
    jitter_eps : float
        Jitter amplitude for GCMI rank tie-breaking.
    seed : int
        Random seed for jitter.
    min_delta_MI : float
        Stop if the best new feature improves MI by less than this.

    Returns
    -------
    selected_indices : list of int
        Indices (in original feature space) of selected features, in the
        order they were added.
    MI_history : list of float
        MI(X_selected[:k] ; y) after adding the k-th feature.
    I_features : np.ndarray, shape (D_features,)
        Univariate MI of each original feature (for inspection).
    """
    X = np.asarray(X, float)
    y = np.asarray(y_array, float).ravel()

    N_samples, D_features = X.shape
    if y.shape[0] != N_samples:
        raise ValueError("X rows and y length mismatch")

    # -------------------------------------------------
    # 1) Univariate MI for all features
    # -------------------------------------------------
    I_features = gcmi_per_feature(X, y, jitter_eps=jitter_eps, seed=seed)

    # -------------------------------------------------
    # 2) Candidate pool (optional preselection)
    # -------------------------------------------------
    if preselect_k is not None and preselect_k < D_features:
        # Take top-k by univariate MI
        candidate_order = np.argsort(I_features)[::-1]   # descending
        candidate_indices = candidate_order[:preselect_k]
    else:
        candidate_indices = np.arange(D_features)

    # keep a set for fast membership tests
    candidate_set = set(candidate_indices.tolist())

    selected_indices = []
    MI_history = []

    # y for gcmi_cc: shape (1, N_samples)
    y_gcmi = y.reshape(1, -1)
    y_gcmi = _jitter_rows_for_gcmi(y_gcmi, eps=jitter_eps, seed=seed + 1)

    current_MI = 0.0

    # -------------------------------------------------
    # 3) Greedy forward loop
    # -------------------------------------------------
    rng = np.random.default_rng(seed + 1234)

    for step in range(max_features):
        best_delta = 0.0
        best_feature = None
        best_MI_for_feature = None

        # loop over still-available candidates
        for j in list(candidate_set):
            # temp subset = already selected + candidate j
            subset_indices = selected_indices + [j]
            X_sub = X[:, subset_indices]            # (N_samples, n_sub)
            X_gcmi = X_sub.T                        # (n_sub, N_samples)

            # jitter rows for GCMI
            X_gcmi = _jitter_rows_for_gcmi(
                X_gcmi,
                eps=jitter_eps,
                seed=seed + 10_000 + j
            )

            # MI(X_subset ; y)
            MI_sub = gcmi_cc(X_gcmi, y_gcmi)

            delta = MI_sub - current_MI
            if delta > best_delta:
                best_delta = delta
                best_feature = j
                best_MI_for_feature = MI_sub

        # check stopping condition
        if best_feature is None or best_delta < min_delta_MI:
            print(f"Stopping at step {step}: ΔMI = {best_delta:.4g} < min_delta_MI")
            break

        # accept best feature
        selected_indices.append(best_feature)
        candidate_set.remove(best_feature)
        current_MI = best_MI_for_feature
        MI_history.append(current_MI)

        print(
            f"Step {step+1:2d}: added feature {best_feature}, "
            f"MI = {current_MI:.4f} bits (Δ = {best_delta:.4f})"
        )

        # safety break if no more candidates
        if not candidate_set:
            break

    return selected_indices, MI_history, I_features


def select_features_by_indices(X, indices):
    """
    Select feature columns of X by integer indices.

    Parameters
    ----------
    X : array-like, shape (N_samples, D_features)
    indices : array-like of ints

    Returns
    -------
    X_selected : array, shape (N_samples, len(indices))
    """
    X = np.asarray(X)
    indices = np.asarray(indices, dtype=int)
    return X[:, indices]



def beam_search_gcmi(
    X,
    y_array,
    max_features=10,
    beam_width=5,
    preselect_k=200,
    jitter_eps=1e-10,
    seed=0,
    min_delta_MI=1e-4
):
    """
    Beam Search feature selection using Gaussian-Copula Mutual Information.

    Parameters
    ----------
    X : array-like (N_samples, D_features)
    y_array : (N_samples,)
    max_features : max final subset size
    beam_width : number of subsets kept at each step
    preselect_k : prefilter using top-k univariate MI features (reduces search space!)
    jitter_eps : jitter for ties
    seed : RNG seed
    min_delta_MI : stop if best improvement < this

    Returns
    -------
    best_subset : list[int]
        Feature indices (original order) of the best subset found.
    best_MI : float
        MI of the best final subset.
    history : list of (subset, MI)
        Best subset at each depth.
    """
    X = np.asarray(X, float)
    y = np.asarray(y_array, float).ravel()
    N, D = X.shape

    # -------------------------------------------
    # 1) Univariate MI — ranking before search
    # -------------------------------------------
    I_uni = gcmi_per_feature(X, y, jitter_eps=jitter_eps, seed=seed)

    # Pre-select top-k candidates
    if preselect_k is not None and preselect_k < D:
        top_idx = np.argsort(I_uni)[::-1][:preselect_k]
    else:
        top_idx = np.arange(D)

    candidate_set = set(top_idx.tolist())

    # y in gcmi format (variables × samples)
    y_gcmi = y.reshape(1, -1)
    y_gcmi = _jitter_rows_for_gcmi(y_gcmi, eps=jitter_eps, seed=seed + 1)

    # -------------------------------------------
    # 2) Beam initialization
    # Each element: (subset_list, MI_value)
    # Start with empty subset (MI = 0)
    # -------------------------------------------
    beam = [([], 0.0)]
    history = []

    rng = np.random.default_rng(seed + 5000)

    # -------------------------------------------
    # 3) Iterative beam expansion
    # -------------------------------------------
    for step in range(1, max_features + 1):
        new_candidates = []

        for subset, subset_MI in beam:
            used = set(subset)
            remaining = candidate_set - used

            for j in remaining:
                new_subset = subset + [j]
                X_sub = X[:, new_subset].T  # (n_vars, N_samples)

                # jitter rows for GCMI
                X_sub = _jitter_rows_for_gcmi(
                    X_sub, eps=jitter_eps, seed=seed + 10000 + j
                )

                MI_new = gcmi_cc(X_sub, y_gcmi)
                delta = MI_new - subset_MI

                if delta >= min_delta_MI:
                    new_candidates.append((new_subset, MI_new))

        if not new_candidates:
            print(f"Stopping at step {step}: No valid MI improvements above threshold.")
            break

        # Keep only top beam_width subsets by MI
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        beam = new_candidates[:beam_width]

        # Record best subset so far
        best_subset, best_MI = beam[0]
        history.append((best_subset, best_MI))

        print(f"Step {step}: Best MI = {best_MI:.5f} bits, Subset = {best_subset}")

    # Best final subset
    best_subset, best_MI = max(beam, key=lambda x: x[1])
    return best_subset, best_MI, history



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

