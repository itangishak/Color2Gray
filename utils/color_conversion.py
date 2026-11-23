import numpy as np

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB image (uint8, H×W×3) to CIE L*a*b* (D65 illuminant)."""
    rgb = rgb.astype(np.float64) / 255.0

    # Linearise sRGB — undo the gamma encoding before the matrix multiply.
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92
    rgb *= 100.0

    # sRGB (linear) → XYZ using the standard IEC 61966-2-1 matrix.
    mat = np.array([[0.4124, 0.3576, 0.1805],
                    [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]])
    xyz = rgb.reshape(-1, 3) @ mat.T
    xyz = xyz.reshape(rgb.shape)

    # XYZ → L*a*b*, normalised against the D65 white point.
    xyz_ref = np.array([95.047, 100.0, 108.883])
    xyz /= xyz_ref
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1 / 3.0)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116.0)

    L = 116 * xyz[:, :, 1] - 16
    a = 500 * (xyz[:, :, 0] - xyz[:, :, 1])
    b = 200 * (xyz[:, :, 1] - xyz[:, :, 2])
    return np.stack([L, a, b], axis=-1)
