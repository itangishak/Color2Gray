import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from utils.color_conversion import rgb_to_lab
from scipy.ndimage import sobel, gaussian_filter, median_filter, uniform_filter
from utils.guided_filter import guided_filter


def color2gray_vanilla(rgb_img: np.ndarray, theta: float = 30.0, lab_override=None) -> np.ndarray:
    """
    Color2Gray (Gooch et al., SIGGRAPH 2005).

    For each pixel pair, pick either the luminance difference ΔL or a scaled
    signed chromatic distance ΔC depending on which dominates at angle theta.
    Solve the resulting Poisson system by conjugate gradient.
    """
    lab = rgb_to_lab(rgb_img) if lab_override is None else lab_override

    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    h, w = L.shape
    n = h * w

    # Build a flat list of horizontal then vertical neighbour pairs.
    m = (h * (w - 1)) + ((h - 1) * w)
    i_edges = np.empty(m, dtype=np.int32)
    j_edges = np.empty(m, dtype=np.int32)

    idx = np.arange(n, dtype=np.int32).reshape(h, w)
    h_size = h * (w - 1)
    i_edges[:h_size] = idx[:, :-1].ravel()
    j_edges[:h_size] = idx[:, 1:].ravel()
    i_edges[h_size:] = idx[:-1, :].ravel()
    j_edges[h_size:] = idx[1:, :].ravel()

    Lf = L.ravel()
    af = a.ravel()
    bf = b.ravel()

    dL = Lf[i_edges] - Lf[j_edges]

    # Signed chromatic distance: sign(da+db) captures the direction of the
    # colour shift in the a-b plane rather than just saturation magnitude.
    da = af[i_edges] - af[j_edges]
    db = bf[i_edges] - bf[j_edges]
    dC = np.sign(da + db) * np.sqrt(da ** 2 + db ** 2)

    k = np.tan(np.deg2rad(np.clip(theta, 0.0, 85.0)))
    use_L = np.abs(dL) >= k * np.abs(dC)
    d_ij = np.where(use_L, dL, k * dC)

    # Assemble Laplacian: -1 on each edge, degree on diagonal.
    # A small regulariser (1e-3) prevents a singular system when the image is uniform.
    row_off = np.concatenate([i_edges, j_edges])
    col_off = np.concatenate([j_edges, i_edges])
    data_off = -np.ones(row_off.shape[0], dtype=np.float32)
    A = sparse.csr_matrix((data_off, (row_off, col_off)), shape=(n, n))
    deg = np.bincount(row_off, minlength=n).astype(np.float32)
    A.setdiag(A.diagonal() + deg + 1e-3)

    b_vec = np.zeros(n, dtype=np.float32)
    np.add.at(b_vec, i_edges, d_ij)
    np.add.at(b_vec, j_edges, -d_ij)

    # Warm-start from L* — usually converges in well under 150 iterations.
    g_flat, _ = cg(A, b_vec, x0=Lf.copy(), tol=1e-4, maxiter=150)

    g = g_flat.reshape(h, w)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8) * 255
    return g.astype(np.uint8)


def color2gray_enhanced(
    rgb_img: np.ndarray,
    theta: float = 30.0,
    alpha: float = 4.0,
    guided_r_list: list = [3, 8, 16],
    isolum_threshold: float = 5.0,
    lab_override=None,
    force_ishihara_rescue: bool = False,
) -> np.ndarray:
    """
    Enhanced Color2Gray with saliency-aware chromatic boosting.

    Estimates how isoluminant the image is (rho), then scales up chroma
    boost strength, relaxes suppression gates, and blends in finer detail
    scales accordingly. Feeds the boosted Lab into the vanilla Poisson solver.
    """
    lab = rgb_to_lab(rgb_img) if lab_override is None else lab_override

    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # Initial light denoise — just enough to kill salt-and-pepper before gradients.
    a_clean = gaussian_filter(median_filter(a, size=3), sigma=0.5)
    b_clean = gaussian_filter(median_filter(b, size=3), sigma=0.5)

    # Measure isoluminance: high chroma relative to low L variance → Ishihara-like.
    chroma = np.sqrt(a_clean ** 2 + b_clean ** 2)
    isolum_score = float(np.mean(chroma)) / (float(np.std(L)) + 1.0)
    S_low, S_high, k_sigmoid = 5.0, 13.0, 1.5
    x = (isolum_score - S_low) / max(1e-6, (S_high - S_low))
    rho = 1.0 / (1.0 + np.exp(-k_sigmoid * x))
    if bool(force_ishihara_rescue):
        rho = 1.0
    print(f"Isoluminance score: {isolum_score:.2f} → rho = {rho:.3f}")

    # Re-denoise with strength adapted to rho: preserve more detail for isoluminant scenes.
    median_size = int(5 - 2.0 * rho)
    gauss_sigma = 0.7 - 0.3 * rho
    a_clean = gaussian_filter(median_filter(a, size=median_size), sigma=gauss_sigma)
    b_clean = gaussian_filter(median_filter(b, size=median_size), sigma=gauss_sigma)

    # Luminance gradients — computed once, reused for all gates below.
    gx_L, gy_L = sobel(L, axis=0), sobel(L, axis=1)
    mag_L = np.hypot(gx_L, gy_L).astype(np.float32) + 1e-5

    gx_a, gy_a = sobel(a_clean, axis=0), sobel(a_clean, axis=1)
    gx_b, gy_b = sobel(b_clean, axis=0), sobel(b_clean, axis=1)
    mag_C = np.hypot(gx_a + gx_b, gy_a + gy_b).astype(np.float32)

    # Phase-inhibition gate: suppress boosting where chroma edges run parallel to
    # luminance edges (the structure is already captured in L, no need to boost).
    dot_LC = (gx_L * (gx_a + gx_b) + gy_L * (gy_a + gy_b)).astype(np.float32)
    alignment = np.clip(np.abs(dot_LC) / (mag_L * mag_C + 1e-6), 0.0, 1.0)
    M_inhibit = np.sin(np.arccos(alignment)) ** 2

    # Competitive saliency gate: boost where chroma contrast dominates locally over L.
    mag_L_smooth = gaussian_filter(mag_L, sigma=5.0)
    ratio_local = mag_C / (mag_L_smooth + 1.0)
    W_comp = 1.0 / (1.0 + np.exp(-4.0 * (ratio_local - 1.2)))

    # Haze/noise gate: don't boost in flat, low-chroma areas.
    chroma = np.sqrt(a_clean ** 2 + b_clean ** 2)
    M_haze = 1.0 / (1.0 + np.exp(-0.8 * (chroma - 25.0)))

    # Multi-scale chroma details via guided filter.
    detail_a_list, detail_b_list = [], []
    for r in guided_r_list:
        eps = 0.08 + 0.01 * float(r)
        sm_a = guided_filter(L, a_clean, r=int(r), eps=float(eps))
        sm_b = guided_filter(L, b_clean, r=int(r), eps=float(eps))
        detail_a_list.append(a_clean - sm_a)
        detail_b_list.append(b_clean - sm_b)
    D_a = np.maximum.reduce(detail_a_list)
    D_b = np.maximum.reduce(detail_b_list)

    # Blend in finer scales for isoluminant content (rho > 0.3).
    if rho > 0.3:
        fine_a, fine_b = [], []
        for r in [2, 4, 6]:
            eps = 0.05 + 0.005 * float(r)
            sm_a = guided_filter(L, a_clean, r=int(r), eps=float(eps))
            sm_b = guided_filter(L, b_clean, r=int(r), eps=float(eps))
            fine_a.append(a_clean - sm_a)
            fine_b.append(b_clean - sm_b)
        D_a = (1.0 - rho) * D_a + rho * np.maximum.reduce(fine_a)
        D_b = (1.0 - rho) * D_b + rho * np.maximum.reduce(fine_b)

    # Scale effective boost and relax gates with rho.
    alpha_eff = float(alpha) * (1.0 + 3.8 * rho)
    gate_power = 1.0 - 0.88 * rho
    M_haze_eff = np.power(M_haze + 1e-6, gate_power)
    M_inhibit_eff = np.power(M_inhibit + 1e-6, gate_power)
    W_comp_eff = np.clip(np.power(W_comp, (1.0 - 0.7 * rho)), 0.0, 4.0)

    Omega = alpha_eff * M_inhibit_eff * M_haze_eff * W_comp_eff

    # Minimum boost floor for very low-L-gradient regions so pseudo-isochromatic
    # numbers don't disappear even when all gates are near zero.
    floor_boost = 0.7 * float(alpha) * rho
    Omega[mag_L < 3.0] = np.maximum(Omega[mag_L < 3.0], floor_boost)

    # Smooth Omega spatially — tighter radius at high rho to preserve fine dot edges.
    smooth_r = 4 if rho < 0.5 else 3
    smooth_eps = 0.05 if rho < 0.5 else 0.12
    Omega = guided_filter(L, Omega.astype(np.float32), r=smooth_r, eps=smooth_eps)

    print(f"alpha_eff={alpha_eff:.2f} | mean Omega={np.mean(Omega):.3f} | "
          f"max Omega={np.max(Omega):.3f} | pixels>1.5: {100.0 * np.mean(Omega > 1.5):.1f}%")

    a_boosted = np.clip(a_clean + Omega * D_a, -128.0, 127.0)
    b_boosted = np.clip(b_clean + Omega * D_b, -128.0, 127.0)

    lab_boosted = lab.copy()
    lab_boosted[:, :, 1] = a_boosted
    lab_boosted[:, :, 2] = b_boosted

    return color2gray_vanilla(rgb_img, theta=theta, lab_override=lab_boosted)
