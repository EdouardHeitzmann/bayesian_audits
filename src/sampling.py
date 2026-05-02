import random
import numpy as np

def pivotal_sampling(pi):
    pi = pi[:]  # copy
    n = len(pi)

    i = 0
    while i < n - 1:
        if pi[i] in (0, 1):
            i += 1
            continue

        j = i + 1
        while j < n and pi[j] in (0, 1):
            j += 1
        if j == n:
            break

        s = pi[i] + pi[j]
        u = random.random()

        if s <= 1:
            if u < pi[j] / s:
                pi[j] = s
                pi[i] = 0
            else:
                pi[i] = s
                pi[j] = 0
        else:
            if u < (1 - pi[j]) / (2 - s):
                pi[j] = 1
                pi[i] = s - 1
            else:
                pi[i] = 1
                pi[j] = s - 1

    # return selected indices
    return [i for i, p in enumerate(pi) if p == 1]

def implicit_sampler(numpy_arr, wt_vec, sample_size):
    total_weight = wt_vec.sum()
    if sample_size > total_weight:
        raise ValueError("Sample size cannot be greater than total weight.")
    rng = np.random.default_rng()
    positions_to_transfer = rng.choice(total_weight, size=sample_size, replace=False)
    positions_to_transfer.sort()
    bins = np.cumsum(wt_vec)
    owners = np.searchsorted(bins, positions_to_transfer, side="right")
    counts_local = np.bincount(owners, minlength=len(wt_vec))
    return counts_local

def rowwise_dirichlet(parameters, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.gamma(shape=parameters, scale=1.0)
    row_sums = x.sum(axis=1, keepdims=True)
    # For very small concentration parameters, gamma draws can underflow to all zeros,
    # leading to NaNs on normalization. Resample only those rows with a stable fallback.
    bad_rows = (~np.isfinite(row_sums[:, 0])) | (row_sums[:, 0] <= 0.0)
    if np.any(bad_rows):
        tiny = np.finfo(np.float64).tiny
        bad_idx = np.flatnonzero(bad_rows)
        for i in bad_idx:
            alpha = np.maximum(parameters[i], tiny)
            x[i] = rng.dirichlet(alpha)
        row_sums = x.sum(axis=1, keepdims=True)
    return x / row_sums
