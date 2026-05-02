from votekit import RankProfile
import numpy as np
from numpy.typing import NDArray
from functools import lru_cache
import pandas as pd

def _convert_profile_to_numpy_arrays(pf: RankProfile) -> tuple[NDArray, NDArray]:
    """
    Taken from code I wrote for numpy's inner STV logic. 
    wt_vecs are too clunky to work with directly, so I have the conversion function below repeat rows with w>1.
    """
    df = pf.df.copy()
    candidate_to_index = {frozenset([name]): i for i, name in enumerate(pf.candidates)}
    candidate_to_index[frozenset(["~"])] = int(-127)  

    ranking_columns = [c for c in df.columns if c.startswith("Ranking")]
    num_rows = len(df)
    num_cols = len(ranking_columns)
    if num_cols > len(pf.candidates):
        ranking_columns = ranking_columns[: len(pf.candidates)]
        num_cols = len(ranking_columns)
    cells = df[ranking_columns].to_numpy()

    def map_cell(cell):
        try:
            return candidate_to_index[cell]
        except KeyError:
            raise TypeError(f"Found invalid entry: {cell}")

    mapped = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8)
    ballot_matrix: NDArray = np.full(
        (num_rows, num_cols + 1),
        -127,
        dtype=np.int8,
    )
    ballot_matrix[:, :num_cols] = mapped

    wt_vec: NDArray = df["Weight"].astype(np.float64).to_numpy()

    return ballot_matrix, wt_vec

def profile_to_numpy_ballot_matrix(pf: RankProfile) -> NDArray:
    ballot_matrix, wt_vec = _convert_profile_to_numpy_arrays(pf)
    repeated_ballot_matrix = np.repeat(ballot_matrix, wt_vec.astype(int), axis=0)
    return repeated_ballot_matrix

@lru_cache(maxsize=None)
def build_section_list(m: int, L: int) -> list[int]:
    """
    Helper function listing the number of ballots with prescribed lengths and number of candidates.

    Similar to the _child_block_size Peter is adding to votekit.utils.

    Args:
        m: (int) number of candidates that the permutation is picking from.
        L: (int) maximum length of the permutation.

    Returns:
        section_list (list[int]): list where entry i corresponds to the number of permutations of m
            items with length at most L where the first i positions have been prescribed.
            Includes the empty permutation as an option (at every depth).
    """
    section = 1
    section_list = [1]
    for k in range(L - 1, -1, -1):
        section = section * (m - k) + 1
        section_list.append(section)
    section_list.reverse()
    return section_list

def t_type_to_numpy_ballot_matrix(t_type_vect, m, L):
    copied_winner_comb_vec = t_type_vect.copy()
    sections = build_section_list(m, L)
    permutation_elements = np.tile(np.arange(m, dtype=np.int8), (len(t_type_vect), 1))
    mask = np.ones_like(permutation_elements, dtype=bool)
    permutation_matrix = 0-np.ones((t_type_vect.shape[0], L), dtype=np.int8)
    for i, section in enumerate(sections[1:]):
        keep_going = copied_winner_comb_vec !=0
        q_vec, copied_winner_comb_vec[keep_going] = np.divmod(copied_winner_comb_vec[keep_going]-1, section)
        permutation_matrix[keep_going, i] = permutation_elements[keep_going, q_vec]
        mask[keep_going, q_vec] = False
        mask[~keep_going, 0] = False
        permutation_elements = permutation_elements[mask].reshape(permutation_elements.shape[0], -1)
        mask = np.ones_like(permutation_elements, dtype=bool)
    return permutation_matrix

def _vectorized_perm_updater(
    winner_comb_vec: NDArray, m: int, L: int, winner_bitsring_vec: NDArray, winner_vec: NDArray
):
    """
    Re-used from the code I wrote for votekit's meek STV
    """
    winner_mask_array = np.left_shift(1, winner_vec.astype(np.int64)) - 1
    truncated_winner_mask_array = np.bitwise_and(winner_mask_array, winner_bitsring_vec)
    no_update_needed = np.bitwise_and(winner_mask_array + 1, winner_bitsring_vec) != 0
    update_needed = ~no_update_needed
    if np.any(no_update_needed):
        raise ValueError(
            "_vectorized_perm_updater was called to add winners that are "
            "already present in some winner combinations."
        )
    sections = build_section_list(m, L)
    L_vec = np.bitwise_count(winner_bitsring_vec[update_needed])
    section_vec = np.array(sections)[L_vec + 1]
    shift_vec = np.bitwise_count(truncated_winner_mask_array[update_needed])
    return winner_comb_vec + section_vec * (winner_vec - shift_vec) + 1, np.bitwise_or(
        winner_bitsring_vec, winner_mask_array + 1
    )

def numpy_ballot_matrix_to_t_type(
    ballot_matrix: NDArray, m: int, L: int
) -> NDArray:
    winner_bitsring_vec = np.zeros(ballot_matrix.shape[0], dtype=np.int64)
    pos_vec = np.zeros(ballot_matrix.shape[0], dtype=int)
    comb_vec = np.zeros(ballot_matrix.shape[0], dtype=int)
    fpv_vec = np.copy(ballot_matrix[:, 0])
    needs_update = fpv_vec > -1
    #comb_vec[needs_update] +=1
    while np.any(needs_update):
        comb_vec[needs_update], winner_bitsring_vec[needs_update] = _vectorized_perm_updater(
            comb_vec[needs_update], m, L, winner_bitsring_vec[needs_update], fpv_vec[needs_update]
        )
        pos_vec[needs_update] += 1
        fpv_vec[needs_update] = ballot_matrix[needs_update, pos_vec[needs_update]]
        needs_update = fpv_vec > -1
    return comb_vec

def numpy_ballot_matrix_to_profile(numpy_matrix, M, wt_vec=None) -> RankProfile:
    """
    Re-used from the get_profile method I wrote for votekit's STV suite.
    """

    if wt_vec is None:
        wt_vec = np.ones(numpy_matrix.shape[0], dtype=np.float64)  # default weight vector of all 1s

    # make candidates an alphanumeric list of length M: ['A', 'B', 'C', ...]
    candidates = [chr(ord('A') + i) for i in range(M)]

    idx_to_fset = {c: frozenset([candidates[c]]) for c in range(M)}

    # --- 1) drop last column by view (sentinel column) ---
    #A = numpy_matrix[:, :-1]

    n_rows, n_cols = numpy_matrix.shape

    # --- 2) keep only entries in `remaining` ---
    remaining_arr = np.array(range(M), dtype=np.int64)  # all candidates are remaining in this context
    keep_mask = np.isin(numpy_matrix, remaining_arr)

    out = numpy_matrix.copy()  # make a copy to avoid modifying the original matrix

    # --- 3.5) drop rows that are empty after filtering AND rows with weight 0 ---
    # keep rows that have at least one remaining candidate and nonzero weight
    row_keep_mask = ~(out == -127).all(axis=1) & (wt_vec != 0)
    out = out[row_keep_mask]
    wt_vec = wt_vec[row_keep_mask]
    n_rows = out.shape[0]

    # --- 4) int8 -> frozenset mapping via 256-entry LUT ---
    # default for anything missing (including -127): frozenset("~")
    lut: np.ndarray = np.empty(256, dtype=object)
    lut[:] = frozenset(["~"])
    for k, v in idx_to_fset.items():
        lut[int(np.int16(k)) + 128] = v
    # index into LUT (shift by +128 to map [-128,127] -> [0,255])
    obj = lut[out.astype(np.int16, copy=False) + 128]  # dtype=object, frozensets

    # --- 5) to DataFrame with Ranking_i columns ---
    data = {f"Ranking_{i+1}": obj[:, i] for i in range(n_cols)}
    df = pd.DataFrame(data)

    # --- 6) Ballot Index column & set as index ---
    df.insert(0, "Ballot Index", np.arange(n_rows, dtype=int))
    df.set_index("Ballot Index", inplace=True)

    # --- 7) Voter Set: empty set per row (distinct objects) ---
    df["Voter Set"] = pd.Series([set() for _ in range(n_rows)], dtype=object, index=df.index)

    # --- 8) Weight column ---
    df["Weight"] = wt_vec.astype(np.float64, copy=False)

    return RankProfile(
        max_ranking_length=n_cols,
        candidates=tuple(candidates),
        df=df,
    )

def _permutation_matrix_constructor(
    m: int,
    L: int,
    sections: list[int] | None = None,
    dtype = None
):
    """
    Return a dense matrix where each row is a permutation of m items with length at most L.

    Row i of this matrix should correspond to
        `index_to_lexicographic_ballot(index: i-1, n_candidates: m, max_length: L)`
        from votekit.utils.
    Building this matrix is a bad idea when m, L are greater than 10.

    Args:
        m: (int) number of candidates that each permutation is picking from.
        L: (int) the maximum ranking length of each permutation.
        sections: (list[int] | None) pre-computed section list for the given m, L.
            If None, this will be computed by the function.
        dtype: (np.dtype | None) the dtype of the output array. If None, this will be
            int8 if possible and int16 otherwise.
    """
    if sections is None:
        sections = build_section_list(m, L)

    if len(sections) != L + 1:
        raise ValueError("sections must have length L+1")

    if dtype is None:
        out_dtype = np.dtype(np.int16 if m > np.iinfo(np.int8).max else np.int8)
    else:
        out_dtype = np.dtype(dtype)

    A = np.full((sections[0], L), -1, dtype=out_dtype)
    used = np.zeros(m, dtype=bool)

    def fill(start, depth):
        if depth == L:
            return

        section = sections[depth + 1]
        row = start + 1

        for x in range(m):
            if used[x]:
                continue

            A[row : row + section, depth] = x
            used[x] = True
            fill(row, depth + 1)
            used[x] = False

            row += section

    fill(0, 0)
    return A
