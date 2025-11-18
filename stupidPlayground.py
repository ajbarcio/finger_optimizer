import numpy as np
import itertools
import time
from utils import generate_matrices_from_pattern, canonical_form_general


# Build group elements (unchanged)
def build_group_elements(m, n, include_global_flip_options=True):
    elems = []
    for sign_pattern in itertools.product((-1, 1), repeat=m):
        row_signs = np.array(sign_pattern, dtype=np.int8)
        for perm in itertools.permutations(range(n)):
            perm_arr = np.array(perm, dtype=np.int64)
            if include_global_flip_options:
                elems.append((row_signs, perm_arr, 1))
                elems.append((row_signs, perm_arr, -1))
            else:
                elems.append((row_signs, perm_arr, 1))
    return elems

# ---- CORRECTED precompute_maps ----
def precompute_maps(positions, D, group_elems):
    """
    positions: list of (r,c) variable positions in generation order, length K
    D: base matrix (2D numpy)
    group_elems: list of (row_signs, perm, global_flip)

    Returns a list of dicts, one per group element:
        {
          "src_for_target": int32[K]  # the source variable index that maps to target j, or -1 if maps from fixed slot
          "fixed_value_for_target": int8[K]  # meaningful when src_for_target[j] == -1
          "row_sign_for_src": int8[K]  # row sign for each source index (indexable by source index)
          "global_flip": int (+1 or -1)
        }
    """
    pos_to_index = {pos: idx for idx, pos in enumerate(positions)}
    K = len(positions)
    m, n = D.shape
    pre = []

    for (row_signs, perm, global_flip) in group_elems:
        # inverse permutation (perm_inv[c_tgt] = c_src)
        perm_inv = np.empty_like(perm)
        perm_inv[perm] = np.arange(len(perm), dtype=np.int64)

        src_for_target = np.full(K, -1, dtype=np.int32)
        fixed_value_for_target = np.zeros(K, dtype=np.int8)

        # For each target index (tgt position) determine where its value comes from under g^-1:
        # The value at target position (r_tgt, c_tgt) after applying g comes from source position (r_src, c_src)
        # with r_src = r_tgt and c_src = perm_inv[c_tgt]
        for tgt_idx, (r_tgt, c_tgt) in enumerate(positions):
            r_src = int(r_tgt)
            c_src = int(perm_inv[int(c_tgt)])
            src_pos = (r_src, c_src)
            if src_pos in pos_to_index:
                src_idx = pos_to_index[src_pos]
                src_for_target[tgt_idx] = src_idx
            else:
                # maps from a fixed slot in base matrix; record that fixed value (apply row_sign not yet)
                fixed_value_for_target[tgt_idx] = int(D[r_src, c_src])

        # row_sign for each source variable index: row_sign[row_of_src]
        row_sign_for_src = np.array([int(row_signs[int(r)]) for (r, c) in positions], dtype=np.int8)

        pre.append({
            "src_for_target": src_for_target,
            "fixed_value_for_target": fixed_value_for_target,
            "row_sign_for_src": row_sign_for_src,
            "global_flip": int(global_flip)
        })

    return pre

# ---- corrected group-aware orderly generator ----
def isomorph_free_generator_groupaware(D, value_set=(-1,0,1), positions_order=None,
                                       use_final_verification=True, debug=False, debug_limit=20):
    """
    Corrected group-aware isomorph-free generator using orderly generation pruning.
    """
    D = np.array(D)
    m, n = D.shape

    # positions in generation order
    if positions_order is None:
        positions = [(i, j) for i in range(m) for j in range(n) if D[i, j] != 0]
    else:
        positions = list(positions_order)
    K = len(positions)
    base = D.copy().astype(np.int8)

    # build group elements and precompute maps
    group_elems = build_group_elements(m, n, include_global_flip_options=True)
    pre = precompute_maps(positions, D, group_elems)

    assigned = [0] * K
    yielded = 0
    pruned = 0
    checks = 0

    def prefix_prune_check(t):
        nonlocal checks, pruned
        checks += 1
        if t == 0:
            return False
        curr_prefix = tuple(int(x) for x in assigned[:t])

        for g in pre:
            src_for_target = g["src_for_target"]
            fixed_vals = g["fixed_value_for_target"]
            row_sign_for_src = g["row_sign_for_src"]
            global_flip = g["global_flip"]

            # Build mapped prefix only if fully determined
            mapped = [0] * t
            incomplete = False
            for k in range(t):
                src_idx = int(src_for_target[k])
                if src_idx == -1:
                    # determined by fixed value: apply global_flip
                    mapped_val = int(global_flip * int(fixed_vals[k]))
                else:
                    if src_idx >= t:
                        incomplete = True
                        break
                    # determined by assigned[src_idx], apply row sign and global flip
                    mapped_val = int(global_flip * int(row_sign_for_src[src_idx]) * int(assigned[src_idx]))
                mapped[k] = mapped_val

            if incomplete:
                continue

            # now mapped is fully determined; prune only if mapped < curr_prefix
            if tuple(mapped) < curr_prefix:
                pruned += 1
                return True

        return False

    def rec(pos_idx):
        nonlocal yielded, pruned
        if pos_idx == K:
            out = base.copy()
            for idx, (r,c) in enumerate(positions):
                out[r, c] = int(assigned[idx])
            out = out.astype(np.int8)
            if use_final_verification:
                canon = canonical_form_general(out)  # user's function
                out_tuple = tuple(map(tuple, out.tolist()))
                if canon == out_tuple:
                    yielded += 1
                    if debug and yielded <= debug_limit:
                        print(f"[DEBUG] yield #{yielded}: prefix checks={checks}, pruned={pruned}")
                    yield out
                else:
                    # shouldn't happen; skip
                    if debug:
                        print("[DEBUG] final verification FAILED for candidate; skipping")
                    return
            else:
                yielded += 1
                if debug and yielded <= debug_limit:
                    print(f"[DEBUG] yield #{yielded}: prefix checks={checks}, pruned={pruned}")
                yield out
            return

        for v in value_set:
            assigned[pos_idx] = int(v)
            if prefix_prune_check(pos_idx + 1):
                # pruned here; continue with next value
                continue
            yield from rec(pos_idx + 1)

    yield from rec(0)

# ---- helper collector ----
def collect_generator_to_array(gen, chunk_size=2000):
    blocks = []
    buf = []
    last_shape = None
    for i, mat in enumerate(gen, 1):
        buf.append(mat)
        last_shape = mat.shape
        if len(buf) >= chunk_size:
            blocks.append(np.stack(buf))
            buf = []
    if buf:
        blocks.append(np.stack(buf))
    if not blocks:
        return np.empty((0,)+last_shape if last_shape is not None else (0,0,0), dtype=np.int8)
    return np.concatenate(blocks, axis=0)

# ---- quick test harness for 3x4 ----
if __name__ == "__main__":
    import itertools, time

    D = np.ones((3,4), dtype=int)
    vals = (-1, 0, 1)

    t0 = time.time()
    gen = isomorph_free_generator_groupaware(D, value_set=vals, use_final_verification=True, debug=True)
    uniq = []
    for i, mat in enumerate(gen, 1):
        uniq.append(mat)
        if i % 500 == 0:
            print(f"yielded {i:,}", end="\r")
    t1 = time.time()
    uniq_array = np.stack(uniq) if uniq else np.empty((0,)+D.shape, dtype=np.int8)
    print("\nDone. unique count:", uniq_array.shape[0], "time(s):", t1 - t0)
