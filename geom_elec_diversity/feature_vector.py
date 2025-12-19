import numpy as np
from pyscf import scf, dft
from scipy.spatial import cKDTree, ConvexHull

# Use LB guess from qstack
from qstack.spahm.guesses import LB, solveF

import numpy as np
from pyscf import scf, dft
from scipy.spatial import cKDTree, ConvexHull
import glob
import random as rnd
from qstack.compound import xyz_to_mol
import time
from qstack.spahm.guesses import LB, solveF



if not hasattr(np, "float"):
    np.float = float

# constants
_COMPOSITION_ELEMS = ['C', 'H', 'O', 'N', 'S']
_HEAVY_ELEMS = ['C', 'N', 'O', 'S']

# approximate atomic masses (amu) for CHONS
_ATOMIC_MASSES = {
    'H': 1.008,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'S': 32.06,
}

# covalent radii (Å-ish, only ratios matter if units are consistent)
_COVALENT_RADII = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'S': 1.05,
}

def _get_coords_symbols(mol):
    """
    Returns:
      coords  : (natm,3) array
      symbols : list length natm
    """
    coords = mol.atom_coords()  # PySCF: in Bohr by default; consistent across molecules
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    return np.asarray(coords, float), symbols


def _composition_block(symbols):
    """Counts of [C, H, O, N, S]."""
    counts = []
    for elem in _COMPOSITION_ELEMS:
        counts.append(sum(1 for s in symbols if s == elem))
    return np.array(counts, float)


def _shape_block(coords, symbols):
    """
    Shape / size (6 features):
      - radius of gyration
      - inertia tensor eigenvalues (λ1 >= λ2 >= λ3)
      - asphericity = λ1 - 0.5(λ2+λ3)
      - convex hull volume
    """
    natm = coords.shape[0]
    if natm == 0:
        return np.zeros(6, float)

    # mass
    m = np.array([_ATOMIC_MASSES.get(sym, 12.0) for sym in symbols], float)
    Mtot = m.sum() + 1e-12

    # center of mass
    r_cm = (m[:, None] * coords).sum(axis=0) / Mtot
    rel = coords - r_cm[None, :]

    # radius of gyration
    rg2 = (m * np.sum(rel**2, axis=1)).sum() / Mtot
    rg = float(np.sqrt(max(rg2, 0.0)))

    # inertia tensor
    I = np.zeros((3, 3), float)
    for mi, ri in zip(m, rel):
        r2 = np.dot(ri, ri)
        I += mi * (r2 * np.eye(3) - np.outer(ri, ri))

    evals, _ = np.linalg.eigh(I)
    evals = np.sort(evals)[::-1]  # descending λ1 >= λ2 >= λ3
    lam1, lam2, lam3 = evals
    asph = float(lam1 - 0.5 * (lam2 + lam3))

    # convex hull volume
    if natm >= 4:
        try:
            hull = ConvexHull(coords)
            vol = float(hull.volume)
        except Exception:
            vol = 0.0
    else:
        vol = 0.0

    return np.array([rg, lam1, lam2, lam3, asph, vol], float)


def _build_bond_graph(coords, symbols, scale=1.25):
    """
    Build adjacency list using covalent radii and scale factor.
    Edge i-j if distance < scale * (r_cov[i] + r_cov[j]).
    Returns:
      neighbors: list of lists, length natm
    """
    natm = len(symbols)
    neighbors = [[] for _ in range(natm)]
    if natm <= 1:
        return neighbors

    # precompute radii
    r_cov = np.array([_COVALENT_RADII.get(sym, 0.7) for sym in symbols], float)

    for i in range(natm):
        for j in range(i + 1, natm):
            rij = coords[j] - coords[i]
            dist = np.linalg.norm(rij)
            cutoff = scale * (r_cov[i] + r_cov[j])
            if dist < cutoff:
                neighbors[i].append(j)
                neighbors[j].append(i)

    return neighbors


def _two_body_block(coords, symbols):
    """
    Two-body block (24 features):
      For each E in ['C','N','O','S']:
        - For each atom of element E:
          * use KDTree over heavy atoms (C,N,O,S)
          * distances d1,d2 to nearest heavy neighbors (excluding self)
        - Collect all d1 and all d2 for that element.
        - Take percentiles (25,50,75) of d1 list, and same for d2 list.
      -> 4 elements x 2 (d1,d2) x 3 percentiles = 24 features.
    """
    coords = np.asarray(coords, float)
    natm = coords.shape[0]
    if natm == 0:
        return np.zeros(24, float)

    symbols = list(symbols)

    # heavy atoms
    heavy_idx = [i for i, s in enumerate(symbols) if s in _HEAVY_ELEMS]
    if len(heavy_idx) == 0:
        return np.zeros(24, float)

    heavy_coords = coords[heavy_idx]
    tree = cKDTree(heavy_coords)

    feats = []
    for elem in _HEAVY_ELEMS:
        # collect d1,d2 per central atom of type elem
        d1_list = []
        d2_list = []
        for i, s in enumerate(symbols):
            if s != elem:
                continue
            # query nearest heavy neighbors
            ci = coords[i]
            if i in heavy_idx:
                # if central is heavy, need to skip self; query k=3
                k = min(3, len(heavy_idx))
            else:
                k = min(2, len(heavy_idx))

            dists, idxs = tree.query(ci, k=k)
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            # map heavy indices back to global indices
            global_idxs = [heavy_idx[j] for j in idxs]
            # drop self if present
            filtered = [(d, g) for d, g in zip(dists, global_idxs) if g != i]

            if len(filtered) == 0:
                continue
            # we need up to two distances
            d_sorted = sorted(d for d, g in filtered)
            if len(d_sorted) >= 1:
                d1_list.append(d_sorted[0])
            if len(d_sorted) >= 2:
                d2_list.append(d_sorted[1])

        # percentiles; if list empty, zeros
        if len(d1_list) == 0:
            d1_feats = np.zeros(3, float)
        else:
            d1_feats = np.percentile(d1_list, [25, 50, 75]).astype(float)
        if len(d2_list) == 0:
            d2_feats = np.zeros(3, float)
        else:
            d2_feats = np.percentile(d2_list, [25, 50, 75]).astype(float)

        feats.append(d1_feats)
        feats.append(d2_feats)

    return np.concatenate(feats, axis=0)  # 4 * (3+3) = 24


def _circular_mean_and_variance(angles):
    """
    angles in radians, 1D array.
    Returns (mean_angle, circular_variance).
    """
    if len(angles) == 0:
        return 0.0, 0.0
    ang = np.asarray(angles, float)
    C = np.mean(np.cos(ang))
    S = np.mean(np.sin(ang))
    mean_angle = float(np.arctan2(S, C))
    R = np.sqrt(C**2 + S**2)
    circ_var = float(1.0 - R)
    return mean_angle, circ_var


def _three_body_block(coords, symbols, neighbors):
    """
    Three-body block (8 features):
      - build angles a-j-c for all unordered neighbor pairs around central j
      - only for central element E in ['C','N','O','S']
      - for each E:
          * compute circular mean angle and circular variance
      -> 4 elements x 2 stats = 8 features.
    """
    coords = np.asarray(coords, float)
    natm = coords.shape[0]
    if natm == 0:
        return np.zeros(8, float)

    feats = []
    for elem in _HEAVY_ELEMS:
        angle_list = []
        for j, s in enumerate(symbols):
            if s != elem:
                continue
            neigh = neighbors[j]
            if len(neigh) < 2:
                continue
            # unordered neighbor pairs
            for a_i in range(len(neigh)):
                for c_i in range(a_i + 1, len(neigh)):
                    a = neigh[a_i]
                    c = neigh[c_i]
                    v1 = coords[a] - coords[j]
                    v2 = coords[c] - coords[j]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 < 1e-8 or n2 < 1e-8:
                        continue
                    cosang = np.dot(v1, v2) / (n1 * n2)
                    cosang = np.clip(cosang, -1.0, 1.0)
                    ang = np.arccos(cosang)  # in [0, pi]
                    angle_list.append(ang)
        mu, var = _circular_mean_and_variance(angle_list)
        feats.extend([mu, var])

    return np.array(feats, float)  # 4 * 2 = 8


def _dihedral_angle(p0, p1, p2, p3):
    """
    Compute dihedral angle (in radians) for 4 points p0-p1-p2-p3.
    Range ~ (-pi, pi].
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    p3 = np.asarray(p3, float)

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1
    b1 /= np.linalg.norm(b1) + 1e-12

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    angle = np.arctan2(y, x)
    return float(angle)


def _torsion_block(coords, neighbors):
    """
    Torsion block (4 features):
      - collect all torsion angles i-j-k-l where j-k is a bond and
        i in neighbors[j]{k}, l in neighbors[k]{j}
      - angles in (-pi, pi]
      - define fractions:
          t_gauche_minus: fraction of torsions < -pi/3
          t_gauche_plus : fraction between -pi/3 and +pi/3
          t_trans       : fraction >  pi/3
      - rotatable-bond fraction:
          #bonds with at least one neighbor on each side / total #bonds
    """
    coords = np.asarray(coords, float)
    natm = coords.shape[0]
    if natm == 0:
        return np.zeros(4, float)

    torsions = []
    bonds = []

    # build list of bonds (j,k) with j<k
    for j in range(natm):
        for k in neighbors[j]:
            if j < k:
                bonds.append((j, k))

    n_bonds = len(bonds)
    if n_bonds == 0:
        return np.zeros(4, float)

    n_rotatable = 0
    for (j, k) in bonds:
        neigh_j = [i for i in neighbors[j] if i != k]
        neigh_k = [l for l in neighbors[k] if l != j]
        if len(neigh_j) > 0 and len(neigh_k) > 0:
            n_rotatable += 1
            for i in neigh_j:
                for l in neigh_k:
                    ang = _dihedral_angle(coords[i], coords[j], coords[k], coords[l])
                    torsions.append(ang)

    torsions = np.array(torsions, float)
    if torsions.size == 0:
        t_gm = t_gp = t_tr = 0.0
    else:
        t_gm = float(np.mean(torsions < -np.pi / 3))
        t_gp = float(np.mean((torsions >= -np.pi / 3) & (torsions <= np.pi / 3)))
        t_tr = float(np.mean(torsions > np.pi / 3))

    rot_frac = float(n_rotatable / (n_bonds + 1e-12))

    return np.array([t_gm, t_gp, t_tr, rot_frac], float)


def features_for_molecule(
    mol,
    blocks=('composition', 'shape', 'two_body', 'three_body', 'torsion'),
    scale=1.25,
):
    """
    Geometry + composition descriptor (47 features when all blocks are used):

      1. 'composition'     -> 5 features
      2. 'shape'           -> 6 features
      3. 'two_body'        -> 24 features
      4. 'three_body'      -> 8 features
      5. 'torsion'         -> 4 features
    """
    blocks = tuple(blocks)

    coords, symbols = _get_coords_symbols(mol)

    # build bond graph once (used by three_body and torsion)
    neighbors = _build_bond_graph(coords, symbols, scale=scale)

    chunks = []

    for b in ('composition', 'shape', 'two_body', 'three_body', 'torsion'):
        if b not in blocks:
            continue

        if b == 'composition':
            xb = _composition_block(symbols)
        elif b == 'shape':
            xb = _shape_block(coords, symbols)
        elif b == 'two_body':
            xb = _two_body_block(coords, symbols)
        elif b == 'three_body':
            xb = _three_body_block(coords, symbols, neighbors)
        elif b == 'torsion':
            xb = _torsion_block(coords, neighbors)
        else:
            raise ValueError(f"Unknown block '{b}' requested in features_for_molecule")

        chunks.append(np.asarray(xb, float).ravel())

    if len(chunks) == 0:
        return np.zeros(0, float)

    return np.concatenate(chunks, axis=0)


# Geometry block sizes 
# Not general, only for CHONS. From 47-D feature vector 
GEOM_BLOCK_SIZES = {
    'composition': 5,
    'shape': 6,
    'two_body': 24,
    'three_body': 8,
    'torsion': 4,
}


# ------------------------------------------------------------------
# 2) Electronic blocks (guess & SCF) with LB guess
# ------------------------------------------------------------------

GUESS_ELEC_ELEM_ORDER = ['C', 'H', 'O', 'N', 'S']

import numpy as np
from pyscf import dft

def scf_eigs_and_charges(mol, xc_scf='PBE0', xc_lb='pbe'):
    """
    Build an electronic description from the *converged* SCF solution:

      - Choose RKS/UKS(mol) based on mol.spin
      - Build a spin-consistent LB guess density (same logic as lb_guess_eigs_and_charges)
      - Run SCF with that initial density
      - Extract alpha-channel (or restricted) MO energies and coeffs
      - Use the final density matrix for Mulliken-like atomic charges

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object. IMPORTANT: mol.spin = Nalpha - Nbeta.
    xc_scf : str, optional
        Exchange-correlation functional for the SCF calculation (default 'PBE0').
    xc_lb : str, optional
        Exchange-correlation functional used inside LB(mol, xc=xc_lb) for the guess.

    Returns
    -------
    eps : (nmo,) ndarray of float
        MO energies (alpha channel or restricted).
    C : (nao, nmo) ndarray of float
        MO coefficients (columns = MOs, alpha or restricted).
    n_occ : int
        Number of occupied alpha orbitals (Nalpha).
    q_atomic : (natm,) ndarray of float
        Mulliken-like atomic charges from the final SCF density.
    aoslice : (natm, 4) ndarray of int
        Slices mapping atoms -> AO index ranges.
    """

    # --- 0) Choose restricted vs unrestricted KS based on spin ---
    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    mf.xc = xc_scf
    mf.verbose = 0

    # --- 1) Build LB Fock and solve generalized eigenproblem ---
    # Use the same LB machinery as in lb_guess_eigs_and_charges
    fock_guess = LB(mol, xc_lb)
    eps_guess, C_guess = solveF(mol, fock_guess)

    eps_guess = np.asarray(eps_guess, float)    # (nao,)
    C_guess   = np.asarray(C_guess,   float)    # (nao, nao)
    nao = C_guess.shape[0]

    # --- 2) Total electrons and spin from the Mole object ---
    nelec = mol.nelectron
    if isinstance(nelec, tuple):
        nelec = sum(nelec)

    spin = int(mol.spin)   # PySCF: spin = Nalpha - Nbeta

    # Nalpha, Nbeta consistent with nelec and spin
    Nalpha = (nelec + spin) // 2
    Nbeta  = (nelec - spin) // 2

    # Clamp to [0, nao] for safety
    Nalpha = max(0, min(nao, Nalpha))
    Nbeta  = max(0, min(nao, Nbeta))

    # --- 3) Build spin-resolved LB density (same logic as LB guess) ---
    if Nalpha > 0:
        C_alpha_occ = C_guess[:, :Nalpha]
        dm_alpha = C_alpha_occ @ C_alpha_occ.T   # occupancy 1 per alpha
    else:
        dm_alpha = np.zeros((nao, nao), float)

    if Nbeta > 0:
        C_beta_occ = C_guess[:, :Nbeta]
        dm_beta = C_beta_occ @ C_beta_occ.T      # occupancy 1 per beta
    else:
        dm_beta = np.zeros((nao, nao), float)

    # Initial density for SCF:
    #   - RKS (spin 0): spin-summed density (nao, nao)
    #   - UKS (spin != 0): spin-resolved density (2, nao, nao)
    if mol.spin == 0:
        dm0 = dm_alpha + dm_beta
    else:
        dm0 = np.array([dm_alpha, dm_beta])

    # --- 4) Run SCF with LB-based spin-consistent initial density ---
    mf.kernel(dm0=dm0)

    # --- 5) Extract MO energies, coeffs, occupations (alpha or restricted) ---
    eps_raw = mf.mo_energy
    C_raw   = mf.mo_coeff
    occ_raw = mf.mo_occ

    # Helper to extract a single-spin (alpha) view from PySCF objects
    def _to_alpha_1d(arr):
        # energies / occupations: want 1D (nmo,)
        if isinstance(arr, (list, tuple)):
            # e.g. [eps_alpha, eps_beta]
            return np.asarray(arr[0], float)
        arr_np = np.asarray(arr)
        if arr_np.ndim == 1:
            # restricted or already alpha
            return arr_np.astype(float)
        elif arr_np.ndim == 2:
            # e.g. (2, nmo) -> take alpha
            return arr_np[0].astype(float)
        else:
            raise ValueError(f"Unexpected MO energy/occ shape {arr_np.shape}")

    def _to_alpha_2d(arr):
        # MO coeffs: want 2D (nao, nmo)
        if isinstance(arr, (list, tuple)):
            # [C_alpha, C_beta]
            return np.asarray(arr[0], float)
        arr_np = np.asarray(arr)
        if arr_np.ndim == 2:
            # restricted or already alpha: (nao, nmo)
            return arr_np.astype(float)
        elif arr_np.ndim == 3:
            # (2, nao, nmo) -> alpha
            return arr_np[0].astype(float)
        else:
            raise ValueError(f"Unexpected MO coeff shape {arr_np.shape}")

    eps = _to_alpha_1d(eps_raw)    # (nmo,)
    C   = _to_alpha_2d(C_raw)      # (nao, nmo)
    mo_occ_alpha = _to_alpha_1d(occ_raw)

    # Number of occupied alpha orbitals (Nalpha)
    n_occ = int(np.count_nonzero(mo_occ_alpha > 1e-6))

    # --- 6) Mulliken-like charges from final SCF density ---
    dm = mf.make_rdm1()            # RKS: (nao,nao); UKS: (2,nao,nao)
    S  = mf.get_ovlp()             # (nao,nao)

    if dm.ndim == 3:
        dm_tot = dm[0] + dm[1]     # alpha + beta
    else:
        dm_tot = dm

    PS = dm_tot @ S
    pop_ao = np.diag(PS)

    aoslice = mol.aoslice_by_atom()   # (natm, 4)
    Z = mol.atom_charges().astype(float)
    natm = mol.natm

    q_atomic = np.zeros(natm, float)
    for ia, (ibeg, iend, _, _) in enumerate(aoslice):
        q_atomic[ia] = Z[ia] - pop_ao[ibeg:iend].sum()

    return eps, C, n_occ, q_atomic, aoslice


#def scf_eigs_and_charges(mol):
#    """
#    Build an electronic description from the *converged* SCF solution (PBE0):
#
#      - Run RKS/UKS(mol).kernel() with LB density guess
#      - Use mf.mo_energy, mf.mo_coeff (alpha channel for open-shell)
#      - Use the final density matrix for Mulliken-like atomic charges
#
#    Returns:
#      eps      : (nmo,) array of MO energies (alpha channel or restricted)
#      C        : (nao, nmo) MO coefficients (columns = MOs, alpha or restricted)
#      n_occ    : number of occupied alpha orbitals (n_occ <= nmo)
#      q_atomic : (natm,) Mulliken-like charges from final dm
#      aoslice  : slices mapping atoms -> AOs
#    """
#
#    # --- 0) Choose restricted vs unrestricted KS ---
#    if mol.spin == 0:
#        mf = dft.RKS(mol)
#    else:
#        mf = dft.UKS(mol)
#
#    mf.xc = 'PBE0'
#    mf.verbose = 0
#
#    # --- 1) Build LB Fock and initial density (restricted) ---
#    fock_guess = LB(mol)                    # your LB guess
#    eps_guess, C_guess = solveF(mol, fock_guess)
#
#    eps_guess = np.asarray(eps_guess, float)
#    C_guess   = np.asarray(C_guess,   float)   # (nao, nao)
#
#    nelec = mol.nelectron
#    if isinstance(nelec, tuple):
#        nelec = sum(nelec)
#    nao = C_guess.shape[0]
#
#    n_occ_restricted = max(0, min(nao, nelec // 2))
#    if n_occ_restricted > 0:
#        C_occ = C_guess[:, :n_occ_restricted]
#        dm_restricted = 2.0 * (C_occ @ C_occ.T)   # closed-shell density
#    else:
#        dm_restricted = np.zeros((nao, nao), float)
#
#    # --- 2) Promote to spin-resolved density for open-shell ---
#    if mol.spin == 0:
#        dm0 = dm_restricted                      # (nao, nao)
#    else:
#        # crude but reasonable: split equally between alpha and beta
#        dm_alpha = 0.5 * dm_restricted
#        dm_beta  = 0.5 * dm_restricted
#        dm0 = np.array([dm_alpha, dm_beta])      # (2, nao, nao)
#
#    # --- 3) Run SCF once, starting from LB density ---
#    mf.kernel(dm0=dm0)
#
#    # --- 4) Extract MO energies, coeffs, occupations (alpha or restricted) ---
#    eps_raw = mf.mo_energy
#    C_raw   = mf.mo_coeff
#    occ_raw = mf.mo_occ
#
#    # Helper to extract a single-spin view from PySCF objects
#    def _to_alpha_1d(arr):
#        # energies / occupations: want 1D (nmo,)
#        if isinstance(arr, (list, tuple)):
#            # e.g. [eps_alpha, eps_beta]
#            return np.asarray(arr[0], float)
#        arr_np = np.asarray(arr)
#        if arr_np.ndim == 1:
#            # restricted or already alpha
#            return arr_np.astype(float)
#        elif arr_np.ndim == 2:
#            # e.g. (2, nmo) -> take alpha
#            return arr_np[0].astype(float)
#        else:
#            raise ValueError(f"Unexpected MO energy/occ shape {arr_np.shape}")
#
#    def _to_alpha_2d(arr):
#        # MO coeffs: want 2D (nao, nmo)
#        if isinstance(arr, (list, tuple)):
#            # [C_alpha, C_beta]
#            return np.asarray(arr[0], float)
#        arr_np = np.asarray(arr)
#        if arr_np.ndim == 2:
#            # restricted or already alpha: (nao, nmo)
#            return arr_np.astype(float)
#        elif arr_np.ndim == 3:
#            # (2, nao, nmo) -> alpha
#            return arr_np[0].astype(float)
#        else:
#            raise ValueError(f"Unexpected MO coeff shape {arr_np.shape}")
#
#    eps = _to_alpha_1d(eps_raw)      # (nmo,)
#    C   = _to_alpha_2d(C_raw)        # (nao, nmo)
#    mo_occ_alpha = _to_alpha_1d(occ_raw)
#
#    # Number of occupied alpha orbitals
#    n_occ = int(np.count_nonzero(mo_occ_alpha > 1e-6))
#
#    # --- 5) Mulliken-like charges from final density ---
#    dm = mf.make_rdm1()              # RKS: (nao,nao); UKS: (2,nao,nao)
#    S  = mf.get_ovlp()               # (nao,nao)
#
#    if dm.ndim == 3:
#        dm_tot = dm[0] + dm[1]       # alpha + beta
#    else:
#        dm_tot = dm
#
#    PS = dm_tot @ S
#    pop_ao = np.diag(PS)
#
#    aoslice = mol.aoslice_by_atom()  # (natm, 4)
#    Z = mol.atom_charges().astype(float)
#    natm = mol.natm
#
#    q_atomic = np.zeros(natm, float)
#    for ia, (ibeg, iend, _, _) in enumerate(aoslice):
#        q_atomic[ia] = Z[ia] - pop_ao[ibeg:iend].sum()
#
#    return eps, C, n_occ, q_atomic, aoslice


#def scf_eigs_and_charges(mol):
#    """
#    Build an electronic description from the *converged* SCF solution (PBE0):
#
#      - Run RKS(mol).kernel()
#      - Use mf.mo_energy, mf.mo_coeff
#      - Use the final density matrix for Mulliken-like atomic charges
#
#    Returns:
#      eps      : (nao,) array of MO energies (SCF)
#      C        : (nao, nao) MO coefficients (columns = MOs)
#      n_occ    : number of doubly-occupied orbitals (from mo_occ)
#      q_atomic : (natm,) Mulliken-like charges from final dm
#      aoslice  : slices mapping atoms -> AOs
#    """
#    #mf = scf.RHF(mol)
#
#    mf = dft.RKS(mol)
#    mf.xc = 'PBE0'
#    mf.verbose = 0
#
#    # --- 1) Build LB Hamiltonian ---
#    fock_guess = LB(mol)     # or pbe, hf, etc.
#    eps, C = solveF(mol, fock_guess)
#
#    # --- 2) Build initial density ---
#    nelec = mol.nelectron
#    n_occ = nelec // 2
#    C_occ = C[:, :n_occ]
#    dm0 = 2 * (C_occ @ C_occ.T)
#
#    # --- 3) Run SCF starting from LB density ---
#    if mol.spin == 0:
#        mf.kernel(dm0=dm0)
#    else:
#        dm_alpha = 0.5 * dm0
#        dm_beta = 0.5 * dm0
#        dm0 = np.array([dm_alpha, dm_beta])  # shape (2, nao, nao)
#        mf.kernel(dm0=dm0)
#    
#    #mf = dft.RKS(mol)
#    #mf.xc = 'PBE0'
#    
#    mf.verbose = 0
#    mf.kernel()  # fully converged SCF
#
#    # MO energies and coefficients from SCF
#    eps = np.asarray(mf.mo_energy, float)         # (nmo,)
#    C = np.asarray(mf.mo_coeff, float)           # (nao, nmo)
#
#    # Number of occupied orbitals from mo_occ (safer than nelec//2)
#    mo_occ = np.asarray(mf.mo_occ, float)        # (nmo,)
#    n_occ = int(np.count_nonzero(mo_occ > 1e-6))
#
#    # Mulliken-like atomic charges from final density
#    dm = mf.make_rdm1()                          # (nao, nao)
#    S = mf.get_ovlp()                            # (nao, nao)
#
#    if dm.ndim == 3:             # UHF/UKS: dm[0] = α, dm[1] = β
#        dm_tot = dm[0] + dm[1]   # total density
#    else:
#        dm_tot = dm
#
#    #PS = dm @ S
#    #pop_ao = np.diag(PS)
#
#    PS = dm_tot @ S
#    pop_ao = np.diag(PS)
#
#    aoslice = mol.aoslice_by_atom()              # (natm, 4)
#    Z = mol.atom_charges().astype(float)
#    natm = mol.natm
#
#    q_atomic = np.zeros(natm, float)
#    for ia, (ibeg, iend, _, _) in enumerate(aoslice):
#        q_atomic[ia] = Z[ia] - pop_ao[ibeg:iend].sum()
#
#    return eps, C, n_occ, q_atomic, aoslice


import numpy as np

def lb_guess_eigs_and_charges(mol, xc='pbe'):
    """
    Build a 'pre-SCF' electronic description using the LB2020 guess Hamiltonian:

      - H_LB = LB(mol, xc)
      - Solve H_LB C = S C eps
      - Build spin-consistent density from occupied LB orbitals
      - Mulliken-like atomic charges from that density

    Returns:
      eps      : (nao,) array of LB-guess MO energies
      C        : (nao, nao) MO coefficients (columns = MOs)
      n_occ    : number of *doubly-occupied* orbitals (min(Nα, Nβ))
      q_atomic : (natm,) Mulliken-like charges from LB density
      aoslice  : slices mapping atoms -> AOs
    """
    # --- 1) LB Hamiltonian (effective one-electron) ---
    fock = LB(mol, xc)

    # generalized eigenproblem H C = S C eps
    eps, C = solveF(mol, fock)   # (nao,), (nao, nao)
    eps = np.asarray(eps, float)
    C   = np.asarray(C, float)

    nao = C.shape[0]

    # --- 2) Total electrons and spin from the Mole object ---
    Ne = mol.nelectron            # total electrons (int)
    if isinstance(Ne, tuple):
        Ne = sum(Ne)

    spin = int(mol.spin)          # PySCF convention: spin = Nα - Nβ = 2S

    # Nα, Nβ consistent with Ne and spin
    # PySCF guarantees that (Ne + spin) and (Ne - spin) are even
    Nalpha = (Ne + spin) // 2
    Nbeta  = (Ne - spin) // 2

    # Clamp to [0, nao] for safety on weird edge cases
    Nalpha = max(0, min(nao, Nalpha))
    Nbeta  = max(0, min(nao, Nbeta))

    # Number of *doubly* occupied spatial orbitals
    #n_occ = min(Nalpha, Nbeta)
    n_occ = Nalpha

    # --- 3) Build spin-resolved density from LB orbitals ---
    # Use the same orbital ordering for α and β:
    # α fills first Nα spatial orbitals, β fills first Nβ.
    if Nalpha > 0:
        C_alpha_occ = C[:, :Nalpha]
        dm_alpha = C_alpha_occ @ C_alpha_occ.T   # occupancy 1 per α
    else:
        dm_alpha = np.zeros((nao, nao), float)

    if Nbeta > 0:
        C_beta_occ = C[:, :Nbeta]
        dm_beta = C_beta_occ @ C_beta_occ.T      # occupancy 1 per β
    else:
        dm_beta = np.zeros((nao, nao), float)

    # Total spin-summed density
    dm = dm_alpha + dm_beta                      # (nao, nao)

    # --- 4) Mulliken-like atomic populations / charges ---
    S = mol.intor_symmetric('int1e_ovlp')        # (nao, nao)
    PS = dm @ S
    pop_ao = np.diag(PS)

    aoslice = mol.aoslice_by_atom()
    Z = mol.atom_charges().astype(float)
    natm = mol.natm

    q_atomic = np.zeros(natm, float)
    for ia, (ibeg, iend, _, _) in enumerate(aoslice):
        q_atomic[ia] = Z[ia] - pop_ao[ibeg:iend].sum()

    return eps, C, n_occ, q_atomic, aoslice


#def lb_guess_eigs_and_charges(mol, xc='pbe'):
#    """
#    Build a 'pre-SCF' electronic description using the LB2020 guess Hamiltonian:
#
#      - H_LB = LB(mol, xc)
#      - Solve H_LB C = S C eps
#      - Build closed-shell density from occupied LB orbitals
#      - Mulliken-like atomic charges from that density
#
#    Returns:
#      eps      : (nao,) array of LB-guess MO energies
#      C        : (nao, nao) MO coefficients (columns = MOs)
#      n_occ    : number of doubly-occupied orbitals
#      q_atomic : (natm,) Mulliken-like charges from LB density
#      aoslice  : slices mapping atoms -> AOs
#    """
#    # LB Hamiltonian (effective one-electron)
#    fock = LB(mol, xc)
#
#    # generalized eigenproblem H C = S C eps
#    eps, C = solveF(mol, fock)   # (nao,), (nao, nao)
#    eps = np.asarray(eps, float)
#    C = np.asarray(C, float)
#
#    nao = C.shape[0]
#    nelec_total = mol.nelectron
#    if isinstance(nelec_total, tuple):
#        nelec_total = sum(nelec_total)
#    n_occ = max(0, min(nao, nelec_total // 2))
#
#    # Closed-shell density from occupied LB orbitals
#    if n_occ > 0:
#        C_occ = C[:, :n_occ]
#        dm = 2.0 * (C_occ @ C_occ.T)
#    else:
#        dm = np.zeros((nao, nao), float)
#
#    S = mol.intor_symmetric('int1e_ovlp')
#    PS = dm @ S
#    pop_ao = np.diag(PS)
#
#    aoslice = mol.aoslice_by_atom()
#    Z = mol.atom_charges().astype(float)
#    natm = mol.natm
#    q_atomic = np.zeros(natm, float)
#    for ia, (ibeg, iend, _, _) in enumerate(aoslice):
#        q_atomic[ia] = Z[ia] - pop_ao[ibeg:iend].sum()
#
#    return eps, C, n_occ, q_atomic, aoslice


def guess_frontier_energies_block(eps, n_occ):
    """
    5 features from orbital energies (guess or SCF):
      [HOMO, LUMO, gap, HOMO-1, LUMO+1]
    """
    if eps.size == 0:
        return np.zeros(5, float)

    norb = eps.size
    if n_occ == 0:
        homo_idx = 0
        lumo_idx = min(1, norb - 1)
    else:
        homo_idx = min(n_occ - 1, norb - 1)
        lumo_idx = min(n_occ, norb - 1)

    eH = float(eps[homo_idx])
    eL = float(eps[lumo_idx])
    gap = eL - eH

    homo_m1_idx = max(homo_idx - 1, 0)
    lumo_p1_idx = min(lumo_idx + 1, norb - 1)

    eH_m1 = float(eps[homo_m1_idx])
    eL_p1 = float(eps[lumo_p1_idx])

    return np.array([eH, eL, gap, eH_m1, eL_p1], float)


def guess_population_block(q_atomic):
    """
    [ mean(|q|), max(|q|), std(q) ]
    """
    if q_atomic.size == 0:
        return np.zeros(3, float)

    abs_q = np.abs(q_atomic)
    mean_absq = float(abs_q.mean())
    max_absq = float(abs_q.max())
    std_q = float(q_atomic.std()) if q_atomic.size > 1 else 0.0

    return np.array([mean_absq, max_absq, std_q], float)


def guess_frontier_element_fractions_block(mol, C, n_occ, aoslice):
    """
    CHONS HOMO/LUMO element fractions (10D):
      [ frac_C(HOMO), ..., frac_S(HOMO),
        frac_C(LUMO), ..., frac_S(LUMO) ]
    """
    nao, norb = C.shape
    natm = mol.natm
    if norb == 0 or natm == 0:
        return np.zeros(2 * len(GUESS_ELEC_ELEM_ORDER), float)

    if n_occ == 0:
        homo_idx = 0
    else:
        homo_idx = min(n_occ - 1, norb - 1)
    lumo_idx = min(n_occ, norb - 1)

    symbols = np.array([mol.atom_symbol(i) for i in range(natm)], dtype=object)
    feats = []

    for mo_idx in (homo_idx, lumo_idx):
        W_atom = np.zeros(natm, float)
        for ia, (ibeg, iend, _, _) in enumerate(aoslice):
            coeffs = C[ibeg:iend, mo_idx]
            W_atom[ia] = np.sum(np.abs(coeffs) ** 2)
        total = float(W_atom.sum() + 1e-12)

        for elem in GUESS_ELEC_ELEM_ORDER:
            mask = (symbols == elem)
            frac = float(W_atom[mask].sum() / total) if np.any(mask) else 0.0
            feats.append(frac)

    return np.array(feats, float)


def _atom_weights_from_C(mol, C, mo_idx, aoslice):
    """
    Aggregate |C_mu,mo|^2 per atom and return (W_atom, total),
    where W_atom[a] is the unnormalized weight on atom a.
    """
    natm = mol.natm
    W_atom = np.zeros(natm, float)

    for ia, (ibeg, iend, _, _) in enumerate(aoslice):
        coeffs = C[ibeg:iend, mo_idx]
        W_atom[ia] = np.sum(np.abs(coeffs) ** 2)

    total = float(W_atom.sum())
    return W_atom, total


def guess_ipr_entropy_block(mol, C, n_occ, aoslice):
    """
    IPR and Shannon entropy for HOMO and LUMO.

    4 features:
      [IPR_HOMO, IPR_LUMO, S_HOMO, S_LUMO]
    """
    nao, norb = C.shape
    natm = mol.natm
    if norb == 0 or natm == 0:
        return np.zeros(4, float)

    if n_occ == 0:
        homo_idx = 0
    else:
        homo_idx = min(n_occ - 1, norb - 1)
    lumo_idx = min(n_occ, norb - 1)

    feats = []
    for mo_idx in (homo_idx, lumo_idx):
        W_atom, total = _atom_weights_from_C(mol, C, mo_idx, aoslice)
        p = W_atom / (total + 1e-12)
        ipr = float(np.sum(p ** 2))
        S = float(-np.sum(p * np.log(p + 1e-12)))  # entropy in nats
        feats.append(ipr)
        feats.append(S)

    return np.array(feats, float)


def fukui_block(mol, C, n_occ, aoslice):
    """
    Fukui-like summary statistics based on HOMO / LUMO atom weights.

    f_plus(i)  ~ weight of atom i in LUMO
    f_minus(i) ~ weight of atom i in HOMO

    6 features:
      [ mean(f_plus), max(f_plus), std(f_plus),
        mean(f_minus), max(f_minus), std(f_minus) ]
    """
    nao, norb = C.shape
    natm = mol.natm
    if norb == 0 or natm == 0:
        return np.zeros(6, float)

    if n_occ == 0:
        homo_idx = 0
    else:
        homo_idx = min(n_occ - 1, norb - 1)
    lumo_idx = min(n_occ, norb - 1)

    W_H, tot_H = _atom_weights_from_C(mol, C, homo_idx, aoslice)
    W_L, tot_L = _atom_weights_from_C(mol, C, lumo_idx, aoslice)

    f_minus = W_H / (tot_H + 1e-12)
    f_plus = W_L / (tot_L + 1e-12)

    return np.array([
        float(f_plus.mean()), float(f_plus.max()), float(f_plus.std()),
        float(f_minus.mean()), float(f_minus.max()), float(f_minus.std()),
    ], float)


# Block sizes for electronic features
# Not general
GUESS_ELEC_BLOCK_SIZES = {
    'guess_frontier_energies': 5,
    'guess_population': 3,
    'guess_frontier_element_fractions': 10,
    'guess_ipr_entropy': 4,
    'fukui': 6,
}

GUESS_ELEC_BLOCKS_DEFAULT = (
    'guess_frontier_energies',
    'guess_population',
    'guess_frontier_element_fractions',
    'guess_ipr_entropy',
    'fukui',
)


def guess_elec_features_for_molecule(
    mol,
    elec_blocks=GUESS_ELEC_BLOCKS_DEFAULT,
    xc='pbe',
):
    """
    Build the 'initial-guess' electronic descriptor from LB2020 Hamiltonian.

    Supports blocks:
      - 'guess_frontier_energies'          (5)
      - 'guess_population'                 (3)
      - 'guess_frontier_element_fractions' (10)
      - 'guess_ipr_entropy'                (4)
      - 'fukui'                            (6)
    """
    elec_blocks = tuple(elec_blocks)
    if len(elec_blocks) == 0:
        return np.zeros(0, float)

    eps, C, n_occ, q_atomic, aoslice = lb_guess_eigs_and_charges(mol, xc=xc)

    chunks = []
    for b in elec_blocks:
        if b == 'guess_frontier_energies':
            chunks.append(guess_frontier_energies_block(eps, n_occ))
        elif b == 'guess_population':
            chunks.append(guess_population_block(q_atomic))
        elif b == 'guess_frontier_element_fractions':
            chunks.append(
                guess_frontier_element_fractions_block(mol, C, n_occ, aoslice)
            )
        elif b == 'guess_ipr_entropy':
            chunks.append(
                guess_ipr_entropy_block(mol, C, n_occ, aoslice)
            )
        elif b == 'fukui':
            chunks.append(
                fukui_block(mol, C, n_occ, aoslice)
            )
        else:
            raise ValueError(f"Unknown guess-electronic block '{b}'")

    if not chunks:
        return np.zeros(0, float)

    return np.concatenate(chunks, axis=0)


def scf_elec_features_for_molecule(
    mol,
    elec_blocks=GUESS_ELEC_BLOCKS_DEFAULT, 
    xc_scf='PBE0', xc_lb='pbe'
):
    """
    Build the electronic descriptor using the *converged SCF* solution.
    """
    elec_blocks = tuple(elec_blocks)
    if len(elec_blocks) == 0:
        return np.zeros(0, float)

    eps, C, n_occ, q_atomic, aoslice = scf_eigs_and_charges(mol,  xc_scf=xc_scf, xc_lb=xc_lb)

    chunks = []
    for b in elec_blocks:
        if b == 'guess_frontier_energies':
            chunks.append(guess_frontier_energies_block(eps, n_occ))
        elif b == 'guess_population':
            chunks.append(guess_population_block(q_atomic))
        elif b == 'guess_frontier_element_fractions':
            chunks.append(
                guess_frontier_element_fractions_block(mol, C, n_occ, aoslice)
            )
        elif b == 'guess_ipr_entropy':
            chunks.append(
                guess_ipr_entropy_block(mol, C, n_occ, aoslice)
            )
        elif b == 'fukui':
            chunks.append(
                fukui_block(mol, C, n_occ, aoslice)
            )
        else:
            raise ValueError(f"Unknown SCF-electronic block '{b}'")

    if not chunks:
        return np.zeros(0, float)

    return np.concatenate(chunks, axis=0)


# ------------------------------------------------------------------
# 3) Combined geom + electronic descriptors and helpers
# ------------------------------------------------------------------

def features_geom_comp_guess_elec(
    mol,
    geom_blocks=('composition', 'shape', 'two_body', 'three_body', 'torsion'),
    elec_blocks=GUESS_ELEC_BLOCKS_DEFAULT,
    xc='pbe',
):
    x_geom = np.asarray(features_for_molecule(mol, blocks=geom_blocks), float).ravel()
    x_elec = guess_elec_features_for_molecule(mol, elec_blocks=elec_blocks, xc=xc)
    return np.concatenate([x_geom, x_elec], axis=0)


def features_geom_comp_scf_elec(
    mol,
    geom_blocks=('composition', 'shape', 'two_body', 'three_body', 'torsion'),
    elec_blocks=GUESS_ELEC_BLOCKS_DEFAULT,
):
    x_geom = np.asarray(features_for_molecule(mol, blocks=geom_blocks), float).ravel()
    x_elec = scf_elec_features_for_molecule(mol, elec_blocks=elec_blocks)
    return np.concatenate([x_geom, x_elec], axis=0)


def make_feature_fn_geom_guess_elec(
    geom_blocks=('composition', 'shape', 'two_body', 'three_body', 'torsion'),
    elec_blocks=GUESS_ELEC_BLOCKS_DEFAULT,
    xc='pbe',
):
    geom_blocks = tuple(geom_blocks)
    elec_blocks = tuple(elec_blocks)

    def _fn(mol, blocks=None):
        if blocks is None:
            return features_geom_comp_guess_elec(
                mol,
                geom_blocks=geom_blocks,
                elec_blocks=elec_blocks,
                xc=xc,
            )

        blocks = tuple(blocks)
        chunks = []

        for b in blocks:
            if b in geom_blocks:
                xb = np.asarray(features_for_molecule(mol, blocks=(b,)), float).ravel()
                chunks.append(xb)
            elif b in elec_blocks:
                # compute full electronic vector once, slice out block
                x_all = guess_elec_features_for_molecule(
                    mol, elec_blocks=elec_blocks, xc=xc
                )
                offset = 0
                for eb in elec_blocks:
                    size = GUESS_ELEC_BLOCK_SIZES[eb]
                    if eb == b:
                        chunks.append(x_all[offset:offset + size])
                        break
                    offset += size
            else:
                raise ValueError(f"Unknown block '{b}'")

        return np.concatenate(chunks, axis=0) if chunks else np.zeros(0, float)

    return _fn


def make_feature_fn_geom_scf_elec(
    geom_blocks=('composition', 'shape', 'two_body', 'three_body', 'torsion'),
    elec_blocks=GUESS_ELEC_BLOCKS_DEFAULT,
):
    geom_blocks = tuple(geom_blocks)
    elec_blocks = tuple(elec_blocks)

    def _fn(mol, blocks=None):
        if blocks is None:
            return features_geom_comp_scf_elec(
                mol,
                geom_blocks=geom_blocks,
                elec_blocks=elec_blocks,
            )

        blocks = tuple(blocks)
        chunks = []

        for b in blocks:
            if b in geom_blocks:
                xb = np.asarray(features_for_molecule(mol, blocks=(b,)), float).ravel()
                chunks.append(xb)
            elif b in elec_blocks:
                # compute full electronic vector once, slice out block
                x_all = scf_elec_features_for_molecule(mol, elec_blocks=elec_blocks)
                offset = 0
                for eb in elec_blocks:
                    size = GUESS_ELEC_BLOCK_SIZES[eb]
                    if eb == b:
                        chunks.append(x_all[offset:offset + size])
                        break
                    offset += size
            else:
                raise ValueError(f"Unknown block '{b}'")

        return np.concatenate(chunks, axis=0) if chunks else np.zeros(0, float)

    return _fn


def precompute_features_with_slices(
    mol_list,
    feature_fn,
    geom_blocks,
    elec_blocks,
):
    """
    Run feature_fn ONCE for each mol, with all blocks, and build:
      - X_all: (N_samples, D_total)
      - block_slices: dict mapping block_name -> slice in columns of X_all
      - meta slices: 'geom_all', 'elec_all', 'all'

    Assumes feature_fn(mol) returns features in the order:
        [geom_blocks..., elec_blocks...]
    and that each block has the size given in GEOM_BLOCK_SIZES / GUESS_ELEC_BLOCK_SIZES.
    """
    mol_list = list(mol_list)

    # 1) build X_all using ALL blocks
    X_rows = []
    for mol in mol_list:
        xi = feature_fn(mol)      # full geom+elec descriptor
        xi = np.asarray(xi, float).ravel()
        X_rows.append(xi)
    X_all = np.vstack(X_rows)     # (N, D_total)

    # 2) compute column slices per block
    block_slices = {}
    offset = 0

    # geometry blocks first
    for b in geom_blocks:
        size = GEOM_BLOCK_SIZES[b]
        block_slices[b] = slice(offset, offset + size)
        offset += size

    geom_total_dim = offset

    # then electronic blocks
    for b in elec_blocks:
        size = GUESS_ELEC_BLOCK_SIZES[b]
        block_slices[b] = slice(offset, offset + size)
        offset += size

    elec_total_dim = offset - geom_total_dim
    total_dim = offset

    # meta slices
    block_slices['geom_all'] = slice(0, geom_total_dim)
    block_slices['elec_all'] = slice(geom_total_dim, geom_total_dim + elec_total_dim)
    block_slices['all'] = slice(0, total_dim)

    return X_all, block_slices

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



def precompute_features_with_slices_and_extra(
    mol_list,
    feature_fn,
    geom_blocks,
    elec_blocks,
    extra_block_mats=None,
):
    """
    Like precompute_features_with_slices, but you can append extra descriptor
    blocks (e.g. SLATM, Coulomb matrix) that live at the same level as
    'geom_all' and 'elec_all'.

    Inputs
    ------
    mol_list        : list of PySCF Mole objects, length N
    feature_fn      : function(mol) -> 1D array of geom+elec features
                      (same as before, usually make_feature_fn_geom_*_elec)
    geom_blocks     : tuple/list of geometry block names, in order
    elec_blocks     : tuple/list of electronic block names, in order
    extra_block_mats: dict name -> array, optional
                      each array has shape (N_samples, D_block),
                      where rows are in the SAME order as mol_list.

                      Example:
                        {
                          'slatm': slatm_matrix,      # (N, D_slatm)
                          'coulomb': cm_matrix,       # (N, D_cm)
                        }

    Returns
    -------
    X_all       : (N_samples, D_total) full feature matrix
                  [geom blocks | elec blocks | extra blocks]
    block_slices: dict mapping block_name -> slice in columns of X_all.
                  Includes:
                    - all individual geom blocks
                    - all individual elec blocks
                    - 'geom_all'
                    - 'elec_all'
                    - each extra block name
                    - 'extra_all' (all extra descriptors concatenated)
                    - 'core_all' (geom+elec only, no extras)
                    - 'all' (everything: geom + elec + extras)
    """
    mol_list = list(mol_list)
    N = len(mol_list)

    # 1) core geom+elec features, exactly as in your old function
    X_rows = []
    for mol in mol_list:
        xi = feature_fn(mol)          # full geom+elec descriptor
        xi = np.asarray(xi, float).ravel()
        X_rows.append(xi)
    X_core = np.vstack(X_rows)        # (N, D_core)

    block_slices = {}
    offset = 0

    # geometry blocks first
    for b in geom_blocks:
        size = GEOM_BLOCK_SIZES[b]
        block_slices[b] = slice(offset, offset + size)
        offset += size
    geom_total_dim = offset

    # then electronic blocks
    for b in elec_blocks:
        size = GUESS_ELEC_BLOCK_SIZES[b]
        block_slices[b] = slice(offset, offset + size)
        offset += size
    elec_total_dim = offset - geom_total_dim

    core_total_dim = offset

    # meta slices for core
    block_slices['geom_all'] = slice(0, geom_total_dim)
    block_slices['elec_all'] = slice(geom_total_dim,
                                     geom_total_dim + elec_total_dim)
    block_slices['core_all'] = slice(0, core_total_dim)

    # 2) append extra descriptor blocks (SLATM, CM, etc.)
    extra_cols = []
    extra_start = core_total_dim
    extra_total_dim = 0

    if extra_block_mats is not None:
        for name, mat in extra_block_mats.items():
            mat = np.asarray(mat, float)
            if mat.shape[0] != N:
                raise ValueError(
                    f"Extra block '{name}' has {mat.shape[0]} rows but "
                    f"mol_list has {N}"
                )
            D_block = mat.shape[1]

            # slice for this extra block in the final X_all
            block_slices[name] = slice(offset, offset + D_block)
            offset += D_block
            extra_total_dim += D_block

            extra_cols.append(mat)

    # build final X_all
    if extra_cols:
        X_all = np.hstack([X_core] + extra_cols)
        # meta slice for all extra blocks together
        block_slices['extra_all'] = slice(extra_start,
                                          extra_start + extra_total_dim)
    else:
        X_all = X_core

    # final meta slice: everything
    block_slices['all'] = slice(0, offset)

    return X_all, block_slices

def col_indices_for_blocks(block_slices, block_names):
    """Return a 1D array of column indices for a list of block names."""
    idx = []
    for b in block_names:
        sl = block_slices[b]
        idx.extend(range(sl.start, sl.stop))
    return np.array(idx, dtype=int)
