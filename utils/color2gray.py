# utils/color2gray.py
import numpy as np
from scipy.sparse.linalg import cg
from scipy import sparse
from utils.color_conversion import rgb_to_lab

def color2gray_vanilla(rgb_img, theta=30.0, lab_override=None):
    if lab_override is None:
        lab = rgb_to_lab(rgb_img)
    else:
        lab = lab_override

    L = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    chroma = np.sqrt(a**2 + b**2)

    h, w = L.shape
    n = h * w

    # Build neighbor pairs (8-connectivity + radius)
    radius = 5
    rows, cols, data = [], [], []
    targets = []

    idx = np.arange(n).reshape(h, w)

    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dy == 0 and dx == 0: continue
            if abs(dy) + abs(dx) > radius: continue

            shift_idx = np.roll(np.roll(idx, dy, axis=0), dx, axis=1).flatten()
            mask = (shift_idx >= 0)

            i = idx.flatten()[mask]
            j = shift_idx[mask]

            delta_L = L.flatten()[mask] - L.flatten()[mask]
            delta_C = chroma.flatten()[mask] - chroma.flatten()[mask]
            delta = np.abs(delta_L) + theta * np.abs(delta_C)
            sign = np.sign(delta_L) if theta == 0 else np.sign(delta_L + theta * delta_C)

            rows.extend(i)
            cols.extend(j)
            data.extend(sign * delta)
            rows.extend(j)
            cols.extend(i)
            data.extend(-sign * delta)
            targets.extend(sign * delta)

    # Add small regularization to diagonal
    diag = np.arange(n)
    rows.extend(diag)
    cols.extend(diag)
    data.extend([0.001] * n)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    b = np.zeros(n)
    for t, i in zip(targets, rows[:len(targets)]):
        b[i] += t

    g_flat, _ = cg(A, b, tol=1e-5)
    g = g_flat.reshape(h, w)
    g = (g - g.min()) / (g.max() - g.min()) * 255
    return g.astype(np.uint8)