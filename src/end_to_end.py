from .conversion import build_section_list, _convert_profile_to_numpy_arrays, profile_to_numpy_ballot_matrix, numpy_ballot_matrix_to_t_type, t_type_to_numpy_ballot_matrix, numpy_ballot_matrix_to_profile, _permutation_matrix_constructor
from .sampling import rowwise_dirichlet, pivotal_sampling
import numpy as np
from votekit.elections import FastSTV

MAX_PERMUTATION_MATRIX_BYTES = 512 * 1024 * 1024

def _is_upset(blank_elec, sampled_t_vec, canonical_winners):
    blank_elec._data.wt_vec = sampled_t_vec
    _, play_by_play, _ = blank_elec._run_election(blank_elec._data)
    winners = tuple(
        frozenset([blank_elec.candidates[c] for c in play["winners"]])
        for play in play_by_play
        if play["round_type"] in {"election", "default"}
    )
    return winners != canonical_winners

def _run_posterior_updates_for_batch(
    posterior_samples_per_update,
    pi,
    prior_row,
    discrepancy_mat,
    wt_vec,
    posterior_sample_mat,
    blank_elec,
    canonical_winners,
    rng,
):
    num_upsets = 0
    for _ in range(posterior_samples_per_update):
        sampled_rows = pivotal_sampling(pi)
        posterior_sample_mat = resample_posterior_rows(
            prior_row,
            discrepancy_mat,
            wt_vec,
            posterior_sample_mat,
            sampled_rows,
            rng,
        )
        sampled_t_vec = posterior_sample_mat.sum(axis=0)
        if _is_upset(blank_elec, sampled_t_vec, canonical_winners):
            num_upsets += 1
    upset_probability = num_upsets / posterior_samples_per_update
    return upset_probability, posterior_sample_mat

def _run_batches(
    max_batches,
    batch_size,
    t_type_BAL,
    t_type_CVR,
    dedup_t_type_CVR,
    CVR_lut,
    prior_row,
    discrepancy_mat,
    wt_vec,
    posterior_sample_mat,
    posterior_samples_per_update,
    pi,
    blank_elec,
    canonical_winners,
    rng,
):
    upset_probs = []
    discrepancies = []
    num_discrepancies = 0
    for i in range(max_batches):
        batch_t_type_BAL = t_type_BAL[i * batch_size:(i + 1) * batch_size]
        batch_t_type_CVR = CVR_lut[t_type_CVR[i * batch_size:(i + 1) * batch_size]]
        if np.any(batch_t_type_CVR == -1):
            raise ValueError("CVR lookup failed for some ballots. Check the CVR LUT and the t_type_CVR.")
        np.add.at(discrepancy_mat, (batch_t_type_CVR, batch_t_type_BAL), 1)
        wt_vec -= np.bincount(batch_t_type_CVR, minlength=len(wt_vec))
        num_discrepancies += int(np.count_nonzero(batch_t_type_BAL != dedup_t_type_CVR[batch_t_type_CVR]))
        discrepancies.append(num_discrepancies / ((i + 1) * batch_size))
        uniq_sampled_rows = np.unique(batch_t_type_CVR)
        posterior_sample_mat = resample_posterior_rows(
            prior_row,
            discrepancy_mat,
            wt_vec,
            posterior_sample_mat,
            uniq_sampled_rows,
            rng,
        )
        upset_probability, posterior_sample_mat = _run_posterior_updates_for_batch(
            posterior_samples_per_update,
            pi,
            prior_row,
            discrepancy_mat,
            wt_vec,
            posterior_sample_mat,
            blank_elec,
            canonical_winners,
            rng,
        )
        upset_probs.append(upset_probability)
    return upset_probs, discrepancies, posterior_sample_mat

def e2e_synthetic_audit(pf, m, eta = 0.05, kappa =2, L = None, frac_ASN_max = 0.5, batch_size = 10, posterior_samples_per_update = 1000, rows_per_markov_step = 100, prior_type = "pessimistic", seed = None):
    cands = pf.candidates
    M = len(cands)
    canonical_elec = FastSTV(pf, m, tiebreak="borda")
    canonical_quota = canonical_elec.threshold
    canonical_winners = canonical_elec.get_elected()

    numpy_matrix, wt_vec = _convert_profile_to_numpy_arrays(pf)
    wt_vec = wt_vec.astype(int)
    N = wt_vec.sum()
    pi = (rows_per_markov_step*wt_vec/N).copy()
    max_sample_size = int(N*frac_ASN_max)
    max_batches = max_sample_size//batch_size
    number_of_noised_ballots = int(N*eta)
    if L is None:
        L = M-1
        numpy_matrix = numpy_matrix[:, [i for i in range(numpy_matrix.shape[1]) if i != numpy_matrix.shape[1]-2]]
    sections = build_section_list(M, L)
    T = sections[0]
    perm_matrix_dtype = np.dtype(np.int16 if M > np.iinfo(np.int8).max else np.int8)
    perm_matrix_bytes = T * L * perm_matrix_dtype.itemsize
    if perm_matrix_bytes > MAX_PERMUTATION_MATRIX_BYTES:
        gb = perm_matrix_bytes / (1024 ** 3)
        raise MemoryError(
            f"Permutation matrix would require approximately {gb:.2f} GiB. "
            "Reduce candidates/ranking depth (M or L) before running this audit."
        )
    all_permutations_matrix = _permutation_matrix_constructor(M, L, sections=sections)
    blank_profile = numpy_ballot_matrix_to_profile(all_permutations_matrix, M)
    blank_elec = FastSTV(blank_profile, n_seats=m, tiebreak="random")
    blank_elec.candidates = cands
    blank_elec.threshold = canonical_quota

    dedup_t_type_CVR = numpy_ballot_matrix_to_t_type(numpy_matrix, M, L)

    CVR_lut = np.full(T, -1, dtype=np.int64)
    CVR_lut[dedup_t_type_CVR] = np.arange(len(dedup_t_type_CVR))
    t_type_CVR = np.repeat(dedup_t_type_CVR, wt_vec)

    rng = np.random.default_rng(seed)

    rng.shuffle(t_type_CVR)
    noised_positions = rng.choice(N, size=number_of_noised_ballots, replace=False)
    t_type_BAL = t_type_CVR.copy()
    t_type_BAL[noised_positions] = rng.choice(T, size=number_of_noised_ballots, replace=True)
    if prior_type == "pessimistic":
        epsilon = 1/(T-1)
    elif prior_type == "uniform":
        epsilon = 1/T
    else:
        raise ValueError("Invalid prior type. Must be 'pessimistic' or 'uniform'.")
    prior_row = np.zeros(T, dtype=np.float64) + epsilon
    discrepancy_mat = np.zeros((len(wt_vec),T), dtype=np.int64)
    posterior_sample_mat = wt_vec[:, None] * prior_row[None, :]
    prior_row *= kappa

    if prior_type == "pessimistic":
        for i, t_type in enumerate(dedup_t_type_CVR):
            posterior_sample_mat[i, t_type] = 0
    upset_probs, discrepancies, _ = _run_batches(
        max_batches,
        batch_size,
        t_type_BAL,
        t_type_CVR,
        dedup_t_type_CVR,
        CVR_lut,
        prior_row,
        discrepancy_mat,
        wt_vec,
        posterior_sample_mat,
        posterior_samples_per_update,
        pi,
        blank_elec,
        canonical_winners,
        rng,
    )
    return upset_probs, discrepancies

def _run_posterior_updates_for_batch_multi_kappa(
    posterior_samples_per_update,
    pi,
    prior_rows,
    discrepancy_mat,
    wt_vec,
    posterior_sample_mats,
    blank_elec,
    canonical_winners,
    rng,
):
    num_upsets = np.zeros(len(prior_rows), dtype=np.int64)
    for _ in range(posterior_samples_per_update):
        sampled_rows = pivotal_sampling(pi)
        for i, prior_row in enumerate(prior_rows):
            posterior_sample_mats[i] = resample_posterior_rows(
                prior_row,
                discrepancy_mat,
                wt_vec,
                posterior_sample_mats[i],
                sampled_rows,
                rng,
            )
            sampled_t_vec = posterior_sample_mats[i].sum(axis=0)
            if _is_upset(blank_elec, sampled_t_vec, canonical_winners):
                num_upsets[i] += 1
    upset_probabilities = (num_upsets / posterior_samples_per_update).tolist()
    return upset_probabilities, posterior_sample_mats

def _run_batches_multi_kappa(
    max_batches,
    batch_size,
    t_type_BAL,
    t_type_CVR,
    dedup_t_type_CVR,
    CVR_lut,
    prior_rows,
    discrepancy_mat,
    wt_vec,
    posterior_sample_mats,
    posterior_samples_per_update,
    pi,
    blank_elec,
    canonical_winners,
    rng,
):
    upset_probs_by_kappa = [[] for _ in prior_rows]
    discrepancies = []
    num_discrepancies = 0
    for i in range(max_batches):
        batch_t_type_BAL = t_type_BAL[i * batch_size:(i + 1) * batch_size]
        batch_t_type_CVR = CVR_lut[t_type_CVR[i * batch_size:(i + 1) * batch_size]]
        if np.any(batch_t_type_CVR == -1):
            raise ValueError("CVR lookup failed for some ballots. Check the CVR LUT and the t_type_CVR.")
        np.add.at(discrepancy_mat, (batch_t_type_CVR, batch_t_type_BAL), 1)
        wt_vec -= np.bincount(batch_t_type_CVR, minlength=len(wt_vec))
        num_discrepancies += int(np.count_nonzero(batch_t_type_BAL != dedup_t_type_CVR[batch_t_type_CVR]))
        discrepancies.append(num_discrepancies / ((i + 1) * batch_size))
        uniq_sampled_rows = np.unique(batch_t_type_CVR)
        for j, prior_row in enumerate(prior_rows):
            posterior_sample_mats[j] = resample_posterior_rows(
                prior_row,
                discrepancy_mat,
                wt_vec,
                posterior_sample_mats[j],
                uniq_sampled_rows,
                rng,
            )
        upset_probabilities, posterior_sample_mats = _run_posterior_updates_for_batch_multi_kappa(
            posterior_samples_per_update,
            pi,
            prior_rows,
            discrepancy_mat,
            wt_vec,
            posterior_sample_mats,
            blank_elec,
            canonical_winners,
            rng,
        )
        for j, upset_probability in enumerate(upset_probabilities):
            upset_probs_by_kappa[j].append(upset_probability)
    return upset_probs_by_kappa, discrepancies, posterior_sample_mats

def e2e_synthetic_audit_multi_kappa(pf, m, kappa_list, eta = 0.05, L = None, frac_ASN_max = 0.5, batch_size = 10, posterior_samples_per_update = 1000, rows_per_markov_step = 100, prior_type = "pessimistic", seed = None):
    kappa_values = np.asarray(kappa_list, dtype=np.float64)
    if kappa_values.ndim != 1 or len(kappa_values) == 0:
        raise ValueError("kappa_list must be a non-empty 1D list/array of kappa values.")

    cands = pf.candidates
    M = len(cands)
    canonical_elec = FastSTV(pf, m, tiebreak="borda")
    canonical_quota = canonical_elec.threshold
    canonical_winners = canonical_elec.get_elected()

    numpy_matrix, wt_vec = _convert_profile_to_numpy_arrays(pf)
    wt_vec = wt_vec.astype(int)
    N = wt_vec.sum()
    pi = (rows_per_markov_step*wt_vec/N).copy()
    max_sample_size = int(N*frac_ASN_max)
    max_batches = max_sample_size//batch_size
    number_of_noised_ballots = int(N*eta)
    if L is None:
        L = M-1
        numpy_matrix = numpy_matrix[:, [i for i in range(numpy_matrix.shape[1]) if i != numpy_matrix.shape[1]-2]]
    sections = build_section_list(M, L)
    T = sections[0]
    perm_matrix_dtype = np.dtype(np.int16 if M > np.iinfo(np.int8).max else np.int8)
    perm_matrix_bytes = T * L * perm_matrix_dtype.itemsize
    if perm_matrix_bytes > MAX_PERMUTATION_MATRIX_BYTES:
        gb = perm_matrix_bytes / (1024 ** 3)
        raise MemoryError(
            f"Permutation matrix would require approximately {gb:.2f} GiB. "
            "Reduce candidates/ranking depth (M or L) before running this audit."
        )
    all_permutations_matrix = _permutation_matrix_constructor(M, L, sections=sections)
    blank_profile = numpy_ballot_matrix_to_profile(all_permutations_matrix, M)
    blank_elec = FastSTV(blank_profile, n_seats=m)
    blank_elec.candidates = cands
    blank_elec.threshold = canonical_quota

    dedup_t_type_CVR = numpy_ballot_matrix_to_t_type(numpy_matrix, M, L)

    CVR_lut = np.full(T, -1, dtype=np.int64)
    CVR_lut[dedup_t_type_CVR] = np.arange(len(dedup_t_type_CVR))
    t_type_CVR = np.repeat(dedup_t_type_CVR, wt_vec)

    rng = np.random.default_rng(seed)

    rng.shuffle(t_type_CVR)
    noised_positions = rng.choice(N, size=number_of_noised_ballots, replace=False)
    t_type_BAL = t_type_CVR.copy()
    t_type_BAL[noised_positions] = rng.choice(T, size=number_of_noised_ballots, replace=True)
    if prior_type == "pessimistic":
        epsilon = 1/(T-1)
    elif prior_type == "uniform":
        epsilon = 1/T
    else:
        raise ValueError("Invalid prior type. Must be 'pessimistic' or 'uniform'.")

    base_prior_row = np.zeros(T, dtype=np.float64) + epsilon
    discrepancy_mat = np.zeros((len(wt_vec),T), dtype=np.int64)
    prior_rows = [base_prior_row * kappa for kappa in kappa_values]
    posterior_sample_mats = [wt_vec[:, None] * prior_row[None, :] for prior_row in prior_rows]

    if prior_type == "pessimistic":
        for i, t_type in enumerate(dedup_t_type_CVR):
            for posterior_sample_mat in posterior_sample_mats:
                posterior_sample_mat[i, t_type] = 0
    upset_probs_by_kappa, discrepancies, _ = _run_batches_multi_kappa(
        max_batches,
        batch_size,
        t_type_BAL,
        t_type_CVR,
        dedup_t_type_CVR,
        CVR_lut,
        prior_rows,
        discrepancy_mat,
        wt_vec,
        posterior_sample_mats,
        posterior_samples_per_update,
        pi,
        blank_elec,
        canonical_winners,
        rng,
    )
    return upset_probs_by_kappa, discrepancies

def resample_posterior_rows(prior_row, discrepancy_mat, wt_vec, mutable_posterior_sample_mat, rows_to_resample, rng=None):
    if np.ndim(prior_row) == 2:
        parameter_matrix = discrepancy_mat[rows_to_resample] + prior_row[np.arange(len(rows_to_resample))]
    else:
        parameter_matrix = discrepancy_mat[rows_to_resample] + prior_row
    dirichlet_samples = rowwise_dirichlet(parameter_matrix, rng)*wt_vec[rows_to_resample][:, None]
    mutable_posterior_sample_mat[rows_to_resample] = dirichlet_samples + discrepancy_mat[rows_to_resample]
    return mutable_posterior_sample_mat
