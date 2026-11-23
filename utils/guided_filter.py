import numpy as np
from scipy.ndimage import uniform_filter


def guided_filter(I: np.ndarray, p: np.ndarray, r: int = 4, eps: float = 0.001) -> np.ndarray:
    """
    Guided image filter (He et al., 2010).

    Smooths p using I as a structural guide, preserving edges present in I.
    r controls the filter radius; eps controls how much edge detail bleeds through.
    """
    size = 2 * r + 1

    mean_I  = uniform_filter(I,     size, mode='reflect')
    mean_p  = uniform_filter(p,     size, mode='reflect')
    mean_Ip = uniform_filter(I * p, size, mode='reflect')
    mean_II = uniform_filter(I * I, size, mode='reflect')

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    return uniform_filter(a, size, mode='reflect') * I + uniform_filter(b, size, mode='reflect')
