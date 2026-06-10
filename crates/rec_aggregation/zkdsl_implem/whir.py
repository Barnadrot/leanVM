from snark_lib import *
from fiat_shamir import *

WHIR_INITIAL_FOLDING_FACTOR = WHIR_INITIAL_FOLDING_FACTOR_PLACEHOLDER
WHIR_SUBSEQUENT_FOLDING_FACTOR = WHIR_SUBSEQUENT_FOLDING_FACTOR_PLACEHOLDER
WHIR_FIRST_RS_REDUCTION_FACTOR = WHIR_FIRST_RS_REDUCTION_FACTOR_PLACEHOLDER
MIN_WHIR_LOG_INV_RATE = MIN_WHIR_LOG_INV_RATE_PLACEHOLDER
MAX_WHIR_LOG_INV_RATE = MAX_WHIR_LOG_INV_RATE_PLACEHOLDER
MAX_NUM_VARIABLES_TO_SEND_COEFFS = MAX_NUM_VARIABLES_TO_SEND_COEFFS_PLACEHOLDER

WHIR_ALL_POTENTIAL_NUM_QUERIES = WHIR_ALL_POTENTIAL_NUM_QUERIES_PLACEHOLDER
WHIR_ALL_POTENTIAL_QUERY_GRINDING = WHIR_ALL_POTENTIAL_QUERY_GRINDING_PLACEHOLDER
WHIR_ALL_POTENTIAL_NUM_OODS = WHIR_ALL_POTENTIAL_NUM_OODS_PLACEHOLDER
WHIR_ALL_POTENTIAL_FOLDING_GRINDING = WHIR_ALL_POTENTIAL_FOLDING_GRINDING_PLACEHOLDER
MIN_STACKED_N_VARS = MIN_STACKED_N_VARS_PLACEHOLDER

# WHIR uniskip (pw13 h8): the first WHIR_SKIP_K variables of the INITIAL folding sumcheck are
# bound by ONE univariate challenge r0 over the integer window {0..2^K−1} (same window, K and
# Lagrange-denominator constants as the AIR skip). Mirrors python-verifier
# verify_whir_initial_sumcheck_with_skip and sumcheck::verify_product_sumcheck_with_skip.
WHIR_SKIP_K = SKIP_K_PLACEHOLDER
WHIR_SKIP_WINDOW = 2**WHIR_SKIP_K
WHIR_SKIP_N_COEFFS = 2 * (WHIR_SKIP_WINDOW - 1) + 1
WHIR_SKIP_POWER_SUMS = WHIR_SKIP_POWER_SUMS_PLACEHOLDER  # [S_m = Σ_{j<2^K} j^m; canonical]
WHIR_SKIP_LAGRANGE_C = SKIP_LAGRANGE_C_PLACEHOLDER  # [(Π_{y≠x}(x−y))^{-1}; 2^K] (shared with AIR skip)
WHIR_SKIP_TAIL_VARS = WHIR_INITIAL_FOLDING_FACTOR - WHIR_SKIP_K
WHIR_INITIAL_SUMCHECK_CHALLENGES = WHIR_SKIP_TAIL_VARS + 1


def whir_open(
    prev_fs,
    n_vars,
    initial_log_inv_rate,
    prev_root,
    ood_points_commit,
    combination_randomness_powers_0,
    prev_claimed_sum,
):
    n_rounds, n_final_vars, num_queries, num_oods, query_grinding_bits, folding_grinding = get_whir_params(
        n_vars, initial_log_inv_rate
    )
    folding_factors = Array(n_rounds + 1)
    folding_factors[0] = WHIR_INITIAL_FOLDING_FACTOR
    for i in range(1, n_rounds + 1):
        folding_factors[i] = WHIR_SUBSEQUENT_FOLDING_FACTOR

    all_folding_randomness = Array(n_rounds + 2)
    all_ood_points = Array(n_rounds)
    all_stir_points = Array(n_rounds + 1)
    all_combination_randomness_powers = Array(n_rounds)

    # h8: per-round CHALLENGE counts (round 0 binds WHIR_INITIAL_FOLDING_FACTOR variables with
    # only WHIR_INITIAL_SUMCHECK_CHALLENGES challenges: [r0, then the 3 linear ones]).
    challenge_counts = Array(n_rounds + 1)
    challenge_counts[0] = WHIR_INITIAL_SUMCHECK_CHALLENGES
    for i in range(1, n_rounds + 1):
        challenge_counts[i] = WHIR_SUBSEQUENT_FOLDING_FACTOR

    carry = Array((n_rounds + 1) * 4)
    # Round 0 hoisted out of the loop (uniskip variant); its outputs seed carry[4..8] exactly as
    # the legacy r=0 iteration would have.
    fs0: Mut = prev_fs
    (
        fs0,
        all_folding_randomness[0],
        all_ood_points[0],
        root0,
        all_stir_points[0],
        all_combination_randomness_powers[0],
        claimed_sum_0_out,
        whir_skip_l16,
    ) = whir_round_initial(
        fs0,
        prev_root,
        num_queries[0],
        n_vars + initial_log_inv_rate,
        prev_claimed_sum,
        query_grinding_bits[0],
        num_oods[1],
        folding_grinding[0],
    )
    carry[4] = fs0
    carry[5] = root0
    carry[6] = claimed_sum_0_out
    carry[7] = n_vars + initial_log_inv_rate - WHIR_FIRST_RS_REDUCTION_FACTOR
    for r in range(1, n_rounds):
        base = r * 4
        fs: Mut = carry[base]
        root: Mut = carry[base + 1]
        claimed_sum: Mut = carry[base + 2]
        domain_sz: Mut = carry[base + 3]
        (
            fs,
            all_folding_randomness[r],
            all_ood_points[r],
            root,
            all_stir_points[r],
            all_combination_randomness_powers[r],
            claimed_sum,
        ) = whir_round(
            fs,
            root,
            folding_factors[r],
            two_exp(folding_factors[r]),
            0,
            num_queries[r],
            domain_sz,
            claimed_sum,
            query_grinding_bits[r],
            num_oods[r + 1],
            folding_grinding[r],
        )
        domain_sz -= 1
        carry[base + 4] = fs
        carry[base + 5] = root
        carry[base + 6] = claimed_sum
        carry[base + 7] = domain_sz
    fs: Mut = carry[n_rounds * 4]
    root = carry[n_rounds * 4 + 1]
    claimed_sum: Mut = carry[n_rounds * 4 + 2]
    domain_sz = carry[n_rounds * 4 + 3]

    fs, all_folding_randomness[n_rounds], claimed_sum = sumcheck_verify_with_grinding(
        fs, WHIR_SUBSEQUENT_FOLDING_FACTOR, claimed_sum, 2, folding_grinding[n_rounds]
    )

    fs, final_coefficients = fs_receive_ef_by_log_dynamic(
        fs,
        n_final_vars,
        MAX_NUM_VARIABLES_TO_SEND_COEFFS - WHIR_SUBSEQUENT_FOLDING_FACTOR,
        MAX_NUM_VARIABLES_TO_SEND_COEFFS + 1,
    )

    fs, all_stir_points[n_rounds], final_folds = sample_stir_indexes_and_fold(
        fs,
        num_queries[n_rounds],
        0,
        WHIR_SUBSEQUENT_FOLDING_FACTOR,
        2**WHIR_SUBSEQUENT_FOLDING_FACTOR,
        domain_sz,
        root,
        all_folding_randomness[n_rounds],
        query_grinding_bits[n_rounds],
    )

    final_stir_points = all_stir_points[n_rounds]
    for i in range(0, num_queries[n_rounds]):
        alpha = final_stir_points[i]
        final_pol_eval_on_stir_point = match_range(
            n_final_vars,
            range(
                MAX_NUM_VARIABLES_TO_SEND_COEFFS - WHIR_SUBSEQUENT_FOLDING_FACTOR, MAX_NUM_VARIABLES_TO_SEND_COEFFS + 1
            ),
            lambda n: univariate_eval_on_base(final_coefficients, alpha, n),
        )
        copy_ef(final_pol_eval_on_stir_point, final_folds + i * DIM)

    fs, all_folding_randomness[n_rounds + 1], end_sum = sumcheck_verify(fs, n_final_vars, claimed_sum, 2)

    # h8: the global challenge vector has n_vars − WHIR_SKIP_K + 1 entries: slot 0 = r0 (binding
    # the first WHIR_SKIP_K variables), slot s ≥ 1 ↔ variable s + WHIR_SKIP_K − 1.
    folding_randomness_global = Array((n_vars - WHIR_SKIP_K + 1) * DIM)

    start_buf = Array(n_rounds + 2)
    start_buf[0] = folding_randomness_global
    for i in range(0, n_rounds + 1):
        start: Mut = start_buf[i]
        for j in range(0, challenge_counts[i]):
            copy_ef(all_folding_randomness[i] + j * DIM, start + j * DIM)
        start += challenge_counts[i] * DIM
        start_buf[i + 1] = start
    start = start_buf[n_rounds + 1]
    for j in range(0, n_final_vars):
        copy_ef(all_folding_randomness[n_rounds + 1] + j * DIM, start + j * DIM)

    # h8: commitment OODs are round-0 statements with dense points — under the skip their
    # weight is Σ_j L16[j]·eq(p, bits4(j) ∥ chal[1:]), which factors as
    # MLE(L16)(p[0..K]) · eq(p[K..], chal[1:]) (the head sum of an eq weight is the MLE of the
    # Lagrange table at the point's head coords). Mirrors the python spec's round_weights at
    # round 0 applied to dense points.
    all_ood_recovered_evals = Array(num_oods[0] * DIM)
    for i in range(0, num_oods[0]):
        expanded_from_univariate = expand_from_univariate_ext(ood_points_commit + i * DIM, n_vars)
        ood_head_tbl = compute_eq_mle_extension(expanded_from_univariate, WHIR_SKIP_K)
        ood_head = dot_product_ee_ret(whir_skip_l16, ood_head_tbl, WHIR_SKIP_WINDOW)
        ood_tail = poly_eq_extension_dynamic_ret(
            expanded_from_univariate + WHIR_SKIP_K * DIM,
            folding_randomness_global + DIM,
            n_vars - WHIR_SKIP_K,
        )
        mul_extension(ood_head, ood_tail, all_ood_recovered_evals + i * DIM)
    ood_eval_sum = Array(DIM)
    dot_product_ee_dynamic(
        all_ood_recovered_evals,
        combination_randomness_powers_0,
        ood_eval_sum,
        num_oods[0],
    )

    eval_carry = Array((n_rounds + 1) * 3)
    eval_carry[0] = n_vars
    eval_carry[1] = folding_randomness_global
    eval_carry[2] = ood_eval_sum
    for i in range(0, n_rounds):
        base = i * 3
        n_vars_remaining: Mut = eval_carry[base]
        my_folding_randomness: Mut = eval_carry[base + 1]
        eval_weights: Mut = eval_carry[base + 2]
        n_vars_remaining -= folding_factors[i]
        my_ood_recovered_evals = Array(num_oods[i + 1] * DIM)
        combination_randomness_powers = all_combination_randomness_powers[i]
        # h8: advance by the CHALLENGE count (4 for round 0, not 7); after round 0 the remaining
        # suffix is one-challenge-per-variable again, so every later pairing is pure tensor.
        my_folding_randomness += challenge_counts[i] * DIM
        for j in range(0, num_oods[i + 1]):
            expanded_from_univariate = expand_from_univariate_ext(all_ood_points[i] + j * DIM, n_vars_remaining)
            poly_eq_extension_dynamic_to(
                expanded_from_univariate, my_folding_randomness, my_ood_recovered_evals + j * DIM, n_vars_remaining
            )
        summed_ood = Array(DIM)
        dot_product_ee_dynamic(
            my_ood_recovered_evals,
            combination_randomness_powers,
            summed_ood,
            num_oods[i + 1],
        )

        query_recovered_evals = Array((num_queries[i]) * DIM)
        stir_points_i = all_stir_points[i]
        for j in range(0, num_queries[i]):  # unroll ?
            expanded_from_univariate = expand_from_univariate_base(stir_points_i[j], n_vars_remaining)
            poly_eq_base_extension_to(expanded_from_univariate, my_folding_randomness, query_recovered_evals + j * DIM, n_vars_remaining)
        query_eval_sum = Array(DIM)
        dot_product_ee_dynamic(
            query_recovered_evals,
            combination_randomness_powers + num_oods[i + 1] * DIM,
            query_eval_sum,
            num_queries[i],
        )
        eval_weights = add_extension_ret(eval_weights, query_eval_sum)
        eval_weights = add_extension_ret(summed_ood, eval_weights)
        eval_carry[base + 3] = n_vars_remaining
        eval_carry[base + 4] = my_folding_randomness
        eval_carry[base + 5] = eval_weights
    eval_weights = eval_carry[n_rounds * 3 + 2]
    final_value = match_range(
        n_final_vars,
        range(MAX_NUM_VARIABLES_TO_SEND_COEFFS - WHIR_SUBSEQUENT_FOLDING_FACTOR, MAX_NUM_VARIABLES_TO_SEND_COEFFS + 1),
        lambda n: eval_multilinear_coeffs_rev(final_coefficients, all_folding_randomness[n_rounds + 1], n),
    )
    # copy_ef(mul_extension_ret(eval_weights, final_value), end_sum);

    return fs, folding_randomness_global, eval_weights, final_value, end_sum, whir_skip_l16


def sumcheck_verify(fs, n_steps, claimed_sum, degree: Const):
    challenges = Array(n_steps * DIM)
    new_fs, new_claimed_sum = sumcheck_verify_helper(fs, n_steps, claimed_sum, degree, challenges)
    return new_fs, challenges, new_claimed_sum


def sumcheck_verify_helper(prev_fs, n_steps, prev_claimed_sum, degree: Const, challenges):
    carry = Array((n_steps + 1) * 2)
    carry[0] = prev_fs
    carry[1] = prev_claimed_sum
    for sc_round in range(0, n_steps):
        base = sc_round * 2
        fs: Mut = carry[base]
        claimed_sum: Mut = carry[base + 1]
        fs, poly = fs_receive_ef_inlined(fs, degree + 1)
        polynomial_sum_at_0_and_1(poly, degree, claimed_sum)
        fs, rand = fs_sample_ef(fs)
        claimed_sum = univariate_polynomial_eval(poly, rand, degree)
        copy_ef(rand, challenges + sc_round * DIM)
        carry[base + 2] = fs
        carry[base + 3] = claimed_sum

    final_fs = carry[n_steps * 2]
    final_claimed_sum = carry[n_steps * 2 + 1]
    return final_fs, final_claimed_sum


def sumcheck_verify_reversed(fs, n_steps, claimed_sum, degree: Const):
    challenges = Array(n_steps * DIM)
    new_fs, final_claimed_sum = sumcheck_verify_reversed_helper(fs, n_steps, claimed_sum, degree, challenges)
    return new_fs, challenges, final_claimed_sum


def sumcheck_verify_reversed_helper(fs, n_steps, claimed_sum, degree: Const, challenges):
    debug_assert(n_steps < 32)
    new_fd, final_sum = match_range(
        n_steps,
        range(0, 32),
        lambda s: sumcheck_verify_reversed_helper_const(fs, s, claimed_sum, degree, challenges),
    )
    return new_fd, final_sum


def sumcheck_verify_reversed_helper_const(prev_fs, n_steps: Const, prev_claimed_sum, degree: Const, challenges):
    fs: Mut = prev_fs
    claimed_sum: Mut = prev_claimed_sum
    for sc_round in unroll(0, n_steps):
        fs, poly = fs_receive_ef_inlined(fs, degree + 1)
        polynomial_sum_at_0_and_1(poly, degree, claimed_sum)
        fs, rand = fs_sample_ef(fs)
        claimed_sum = univariate_polynomial_eval(poly, rand, degree)
        copy_ef(rand, challenges + (n_steps - 1 - sc_round) * DIM)

    return fs, claimed_sum


def sumcheck_verify_with_grinding(prev_fs, n_steps, prev_claimed_sum, degree: Const, folding_grinding_bits):
    challenges = Array(n_steps * DIM)
    carry = Array((n_steps + 1) * 2)
    carry[0] = prev_fs
    carry[1] = prev_claimed_sum
    for sc_round in range(0, n_steps):
        base = sc_round * 2
        fs: Mut = carry[base]
        claimed_sum: Mut = carry[base + 1]
        fs, poly = fs_receive_ef_inlined(fs, degree + 1)
        polynomial_sum_at_0_and_1(poly, degree, claimed_sum)
        fs = fs_grinding(fs, folding_grinding_bits)
        fs, rand = fs_sample_ef(fs)
        claimed_sum = univariate_polynomial_eval(poly, rand, degree)
        copy_ef(rand, challenges + sc_round * DIM)
        carry[base + 2] = fs
        carry[base + 3] = claimed_sum

    final_fs = carry[n_steps * 2]
    final_claimed_sum = carry[n_steps * 2 + 1]
    return final_fs, challenges, final_claimed_sum


@inline
def decompose_and_verify_merkle_batch(num_queries, sampled, root, height, num_chunks, stir_points, answers):
    debug_assert(height < 25)
    match_range(
        height,
        range(5, 25),
        lambda h: decompose_and_verify_merkle_batch_with_height(
            num_queries, sampled, root, h, num_chunks, stir_points, answers
        ),
    )
    return


def decompose_and_verify_merkle_batch_with_height(
    num_queries, sampled, root, height: Const, num_chunks, stir_points, answers
):
    if num_chunks == DIM * 2:
        decompose_and_verify_merkle_batch_const(num_queries, sampled, root, height, DIM * 2, stir_points, answers)
        return
    if num_chunks == 16:
        decompose_and_verify_merkle_batch_const(num_queries, sampled, root, height, 16, stir_points, answers)
        return
    if num_chunks == 8:
        decompose_and_verify_merkle_batch_const(num_queries, sampled, root, height, 8, stir_points, answers)
        return
    if num_chunks == 20:
        decompose_and_verify_merkle_batch_const(num_queries, sampled, root, height, 20, stir_points, answers)
        return
    if num_chunks == 1:
        decompose_and_verify_merkle_batch_const(num_queries, sampled, root, height, 1, stir_points, answers)
        return
    if num_chunks == 4:
        decompose_and_verify_merkle_batch_const(num_queries, sampled, root, height, 4, stir_points, answers)
        return
    if num_chunks == 5:
        decompose_and_verify_merkle_batch_const(num_queries, sampled, root, height, 5, stir_points, answers)
        return
    print(num_chunks)
    assert False, "decompose_and_verify_merkle_batch called with unsupported num_chunks"


def decompose_and_verify_merkle_batch_const(
    num_queries, sampled, root, height: Const, num_chunks: Const, stir_points, merkle_leaves
):
    leaf_iv = build_iv(num_chunks * DIGEST_LEN)
    for i in range(0, num_queries):
        merkle_leaves[i], stir_points[i] = decompose_and_verify_merkle_query(
            sampled[i], height, root, num_chunks, leaf_iv
        )
    return


def sample_stir_indexes_and_fold(
    prev_fs,
    num_queries,
    merkle_leaves_in_basefield,
    folding_factor,
    two_pow_folding_factor,
    domain_size,
    prev_root,
    folding_randomness,
    query_grinding_bits,
):
    fs: Mut = prev_fs
    folded_domain_size = domain_size - folding_factor

    fs = fs_grinding(fs, query_grinding_bits)
    sampled, fs = fs_sample_queries(fs, num_queries)

    merkle_leaves = Array(num_queries)
    stir_points = Array(num_queries)

    n_chunks_per_answer: Imm
    # the number of chunk of 8 field elements per merkle leaf opened
    if merkle_leaves_in_basefield == 1:
        n_chunks_per_answer = two_pow_folding_factor
    else:
        n_chunks_per_answer = two_pow_folding_factor * DIM

    decompose_and_verify_merkle_batch(
        num_queries,
        sampled,
        prev_root,
        folded_domain_size,
        n_chunks_per_answer / DIGEST_LEN,
        stir_points,
        merkle_leaves,
    )

    folds = Array(num_queries * DIM)

    poly_eq = compute_eq_mle_extension_dynamic(folding_randomness, folding_factor)

    if merkle_leaves_in_basefield == 1:
        for i in range(0, num_queries):
            dot_product_be_dynamic(merkle_leaves[i], poly_eq, folds + i * DIM, two_pow_folding_factor)
    else:
        for i in range(0, num_queries):
            dot_product_ee_dynamic(merkle_leaves[i], poly_eq, folds + i * DIM, two_pow_folding_factor)

    return fs, stir_points, folds


def whir_skip_lagrange_weights(r0):
    # L_x(r0) = c_x · Π_{y≠x}(r0 − y) over the window {0..2^K−1}; c_x = WHIR_SKIP_LAGRANGE_C are
    # build-time-inverted constant denominators — no in-circuit inversion (same pattern as the
    # AIR skip in recursion.py).
    diffs = Array(WHIR_SKIP_WINDOW)
    for x in unroll(0, WHIR_SKIP_WINDOW):
        diffs[x] = sub_extension_base_ret(r0, x)
    pre = Array(WHIR_SKIP_WINDOW)
    suf = Array(WHIR_SKIP_WINDOW)
    pre[0] = ONE_EF_PTR
    suf[WHIR_SKIP_WINDOW - 1] = ONE_EF_PTR
    for x in unroll(1, WHIR_SKIP_WINDOW):
        pre[x] = mul_extension_ret(pre[x - 1], diffs[x - 1])
        rev = WHIR_SKIP_WINDOW - 1 - x
        suf[rev] = mul_extension_ret(suf[rev + 1], diffs[rev + 1])
    weights = Array(WHIR_SKIP_WINDOW * DIM)
    for x in unroll(0, WHIR_SKIP_WINDOW):
        mul_extension(mul_base_extension_ret(WHIR_SKIP_LAGRANGE_C[x], pre[x]), suf[x], weights + x * DIM)
    return weights


def whir_initial_sumcheck_with_skip(prev_fs, claimed_sum, folding_grinding_bits):
    # h8: rounds 0..WHIR_SKIP_K−1 of the initial folding sumcheck become ONE univariate round.
    # FS order (mirrors sumcheck::verify_product_sumcheck_with_skip): absorb the FULL 31-coeff
    # vector v'(X) → grind → sample r0. Round-0 identity: dot(coeffs, S_m) == claimed_sum
    # (plain window sum, no ê). New target = v'(r0). Then WHIR_SKIP_TAIL_VARS standard rounds.
    fs: Mut = prev_fs
    fs, skip_coeffs = fs_receive_ef_inlined(fs, WHIR_SKIP_N_COEFFS)
    s_consts = Array(WHIR_SKIP_N_COEFFS)
    for j in unroll(0, WHIR_SKIP_N_COEFFS):
        s_consts[j] = WHIR_SKIP_POWER_SUMS[j]
    window_dot = Array(DIM)
    dot_product_be(s_consts, skip_coeffs, window_dot, WHIR_SKIP_N_COEFFS)
    copy_ef(window_dot, claimed_sum)  # write-once equality: the round-0 window identity
    fs = fs_grinding(fs, folding_grinding_bits)
    fs, r0 = fs_sample_ef(fs)
    r0_powers = powers_const(r0, WHIR_SKIP_N_COEFFS)
    target = dot_product_ee_ret(skip_coeffs, r0_powers, WHIR_SKIP_N_COEFFS)
    fs, rest, final_sum = sumcheck_verify_with_grinding(fs, WHIR_SKIP_TAIL_VARS, target, 2, folding_grinding_bits)
    challenges = Array(WHIR_INITIAL_SUMCHECK_CHALLENGES * DIM)
    copy_ef(r0, challenges)
    for j in unroll(0, WHIR_SKIP_TAIL_VARS):
        copy_ef(rest + j * DIM, challenges + (j + 1) * DIM)
    return fs, challenges, final_sum


def sample_stir_indexes_and_fold_initial(prev_fs, num_queries, domain_size, prev_root, folding_randomness, query_grinding_bits):
    # h8 round-0 leaf fold: leaf slot m holds the global TOP-7 index bits big-endian. The top
    # WHIR_SKIP_K bits (block j = m >> WHIR_SKIP_TAIL_VARS) are bound by r0 via Lagrange weights,
    # the lower WHIR_SKIP_TAIL_VARS bits by the linear challenges (eval_eq big-endian):
    # poly_eq[m] = L16[m >> t]·eq_tail[m & (2^t−1)], j-major — mirrors whir::uniskip::skip_leaf_weights.
    fs: Mut = prev_fs
    folded_domain_size = domain_size - WHIR_INITIAL_FOLDING_FACTOR

    fs = fs_grinding(fs, query_grinding_bits)
    sampled, fs = fs_sample_queries(fs, num_queries)

    merkle_leaves = Array(num_queries)
    stir_points = Array(num_queries)

    decompose_and_verify_merkle_batch(
        num_queries,
        sampled,
        prev_root,
        folded_domain_size,
        2**WHIR_INITIAL_FOLDING_FACTOR / DIGEST_LEN,
        stir_points,
        merkle_leaves,
    )

    lagrange16 = whir_skip_lagrange_weights(folding_randomness)
    eq_tail = compute_eq_mle_extension(folding_randomness + DIM, WHIR_SKIP_TAIL_VARS)
    poly_eq = Array(2**WHIR_INITIAL_FOLDING_FACTOR * DIM)
    for j in unroll(0, WHIR_SKIP_WINDOW):
        for u in unroll(0, 2**WHIR_SKIP_TAIL_VARS):
            mul_extension(
                lagrange16 + j * DIM,
                eq_tail + u * DIM,
                poly_eq + (j * 2**WHIR_SKIP_TAIL_VARS + u) * DIM,
            )

    folds = Array(num_queries * DIM)
    for i in range(0, num_queries):
        dot_product_be_dynamic(merkle_leaves[i], poly_eq, folds + i * DIM, 2**WHIR_INITIAL_FOLDING_FACTOR)

    return fs, stir_points, folds, lagrange16


def whir_round_initial(
    prev_fs,
    prev_root,
    num_queries,
    domain_size,
    claimed_sum,
    query_grinding_bits,
    num_ood,
    folding_grinding_bits,
):
    # Round 0 of whir_open under the uniskip: identical to whir_round except (a) the initial
    # sumcheck is the skip variant (4 challenges for 7 variables) and (b) the base-field leaf
    # fold uses the L16 ⊗ eq_tail weight table. Also returns L16 (reused for every round-0
    # statement-weight head sum downstream).
    fs: Mut = prev_fs
    fs, folding_randomness, new_claimed_sum_a = whir_initial_sumcheck_with_skip(
        fs, claimed_sum, folding_grinding_bits
    )

    fs, root, ood_points, ood_evals = parse_commitment(fs, num_ood)

    fs, stir_points, folds, lagrange16 = sample_stir_indexes_and_fold_initial(
        fs,
        num_queries,
        domain_size,
        prev_root,
        folding_randomness,
        query_grinding_bits,
    )

    fs = fs_duplex(fs)
    fs, combination_randomness_gen = fs_sample_ef(fs)

    combination_randomness_powers = powers(combination_randomness_gen, num_queries + num_ood)

    claimed_sum_0 = Array(DIM)
    dot_product_ee_dynamic(ood_evals, combination_randomness_powers, claimed_sum_0, num_ood)

    claimed_sum_1 = Array(DIM)
    dot_product_ee_dynamic(folds, combination_randomness_powers + num_ood * DIM, claimed_sum_1, num_queries)

    new_claimed_sum_b = add_extension_ret(claimed_sum_0, claimed_sum_1)

    final_sum = add_extension_ret(new_claimed_sum_a, new_claimed_sum_b)

    return (
        fs,
        folding_randomness,
        ood_points,
        root,
        stir_points,
        combination_randomness_powers,
        final_sum,
        lagrange16,
    )


def whir_round(
    prev_fs,
    prev_root,
    folding_factor,
    two_pow_folding_factor,
    merkle_leaves_in_basefield,
    num_queries,
    domain_size,
    claimed_sum,
    query_grinding_bits,
    num_ood,
    folding_grinding_bits,
):
    fs: Mut = prev_fs
    fs, folding_randomness, new_claimed_sum_a = sumcheck_verify_with_grinding(
        fs, folding_factor, claimed_sum, 2, folding_grinding_bits
    )

    fs, root, ood_points, ood_evals = parse_commitment(fs, num_ood)

    fs, stir_points, folds = sample_stir_indexes_and_fold(
        fs,
        num_queries,
        merkle_leaves_in_basefield,
        folding_factor,
        two_pow_folding_factor,
        domain_size,
        prev_root,
        folding_randomness,
        query_grinding_bits,
    )

    fs = fs_duplex(fs)
    fs, combination_randomness_gen = fs_sample_ef(fs)

    combination_randomness_powers = powers(combination_randomness_gen, num_queries + num_ood)

    claimed_sum_0 = Array(DIM)
    dot_product_ee_dynamic(ood_evals, combination_randomness_powers, claimed_sum_0, num_ood)

    claimed_sum_1 = Array(DIM)
    dot_product_ee_dynamic(folds, combination_randomness_powers + num_ood * DIM, claimed_sum_1, num_queries)

    new_claimed_sum_b = add_extension_ret(claimed_sum_0, claimed_sum_1)

    final_sum = add_extension_ret(new_claimed_sum_a, new_claimed_sum_b)

    return (
        fs,
        folding_randomness,
        ood_points,
        root,
        stir_points,
        combination_randomness_powers,
        final_sum,
    )


@inline
def polynomial_sum_at_0_and_1(coeffs, degree, dst):
    debug_assert(1 < degree)
    add_ee(sum_continuous_ef(coeffs, degree + 1), coeffs, dst)
    return


def parse_commitment(fs, num_ood):
    root: Imm
    ood_points: Imm
    ood_evals: Imm
    debug_assert(num_ood < 5)
    debug_assert(num_ood != 0)
    new_fs, root, ood_points, ood_evals = match_range(
        num_ood, range(1, 5), lambda n: parse_whir_commitment_const(fs, n)
    )
    return new_fs, root, ood_points, ood_evals


def parse_whir_commitment_const(fs, num_ood: Const):
    new_fs: Mut
    new_fs, root = fs_receive_chunks(fs, 1)
    new_fs, ood_points = fs_sample_many_ef(new_fs, num_ood)
    new_fs, ood_evals = fs_receive_ef_inlined(new_fs, num_ood)
    return new_fs, root, ood_points, ood_evals


@inline
def get_whir_params(n_vars, log_inv_rate):
    debug_assert(WHIR_INITIAL_FOLDING_FACTOR < n_vars)
    nv_except_first_round = n_vars - WHIR_INITIAL_FOLDING_FACTOR
    debug_assert(MAX_NUM_VARIABLES_TO_SEND_COEFFS < nv_except_first_round)
    n_rounds = div_ceil_dynamic(
        nv_except_first_round - MAX_NUM_VARIABLES_TO_SEND_COEFFS, WHIR_SUBSEQUENT_FOLDING_FACTOR
    )
    final_vars = nv_except_first_round - n_rounds * WHIR_SUBSEQUENT_FOLDING_FACTOR

    debug_assert(MIN_WHIR_LOG_INV_RATE <= log_inv_rate)
    debug_assert(log_inv_rate <= MAX_WHIR_LOG_INV_RATE)
    num_queries: Imm
    num_queries = get_num_queries(log_inv_rate, n_vars)

    query_grinding_bits: Imm
    query_grinding_bits = get_query_grinding_bits(log_inv_rate, n_vars)

    num_oods = get_num_oods(log_inv_rate, n_vars)

    folding_grinding: Imm
    folding_grinding = get_folding_grinding(log_inv_rate, n_vars)

    return n_rounds, final_vars, num_queries, num_oods, query_grinding_bits, folding_grinding


@inline
def get_num_queries(log_inv_rate, n_vars):
    res = match_range(
        log_inv_rate,
        range(MIN_WHIR_LOG_INV_RATE, MAX_WHIR_LOG_INV_RATE + 1),
        lambda r: get_num_queries_const_rate(r, n_vars),
    )
    return res


def get_num_queries_const_rate(log_inv_rate: Const, n_vars):
    res = match_range(
        n_vars,
        range(MIN_STACKED_N_VARS, TWO_ADICITY + WHIR_INITIAL_FOLDING_FACTOR - log_inv_rate + 1),
        lambda nv: get_num_queries_const(log_inv_rate, nv),
    )
    return res


def get_num_queries_const(log_inv_rate: Const, n_vars: Const):
    max = len(WHIR_ALL_POTENTIAL_NUM_QUERIES[log_inv_rate - MIN_WHIR_LOG_INV_RATE][n_vars - MIN_STACKED_N_VARS])
    num_queries = Array(max)
    for i in unroll(0, max):
        num_queries[i] = WHIR_ALL_POTENTIAL_NUM_QUERIES[log_inv_rate - MIN_WHIR_LOG_INV_RATE][
            n_vars - MIN_STACKED_N_VARS
        ][i]
    return num_queries


@inline
def get_query_grinding_bits(log_inv_rate, n_vars):
    res = match_range(
        log_inv_rate,
        range(MIN_WHIR_LOG_INV_RATE, MAX_WHIR_LOG_INV_RATE + 1),
        lambda r: get_query_grinding_bits_const_rate(r, n_vars),
    )
    return res


def get_query_grinding_bits_const_rate(log_inv_rate: Const, n_vars):
    res = match_range(
        n_vars,
        range(MIN_STACKED_N_VARS, TWO_ADICITY + WHIR_INITIAL_FOLDING_FACTOR - log_inv_rate + 1),
        lambda nv: get_query_grinding_bits_const(log_inv_rate, nv),
    )
    return res


def get_query_grinding_bits_const(log_inv_rate: Const, n_vars: Const):
    max = len(WHIR_ALL_POTENTIAL_QUERY_GRINDING[log_inv_rate - MIN_WHIR_LOG_INV_RATE][n_vars - MIN_STACKED_N_VARS])
    query_grinding_bits = Array(max)
    for i in unroll(0, max):
        query_grinding_bits[i] = WHIR_ALL_POTENTIAL_QUERY_GRINDING[log_inv_rate - MIN_WHIR_LOG_INV_RATE][
            n_vars - MIN_STACKED_N_VARS
        ][i]
    return query_grinding_bits


@inline
def get_folding_grinding(log_inv_rate, n_vars):
    res = match_range(
        log_inv_rate,
        range(MIN_WHIR_LOG_INV_RATE, MAX_WHIR_LOG_INV_RATE + 1),
        lambda r: get_folding_grinding_const_rate(r, n_vars),
    )
    return res


def get_folding_grinding_const_rate(log_inv_rate: Const, n_vars):
    res = match_range(
        n_vars,
        range(MIN_STACKED_N_VARS, TWO_ADICITY + WHIR_INITIAL_FOLDING_FACTOR - log_inv_rate + 1),
        lambda nv: get_folding_grinding_const(log_inv_rate, nv),
    )
    return res


def get_folding_grinding_const(log_inv_rate: Const, n_vars: Const):
    max = len(WHIR_ALL_POTENTIAL_FOLDING_GRINDING[log_inv_rate - MIN_WHIR_LOG_INV_RATE][n_vars - MIN_STACKED_N_VARS])
    folding_grinding = Array(max)
    for i in unroll(0, max):
        folding_grinding[i] = WHIR_ALL_POTENTIAL_FOLDING_GRINDING[log_inv_rate - MIN_WHIR_LOG_INV_RATE][
            n_vars - MIN_STACKED_N_VARS
        ][i]
    return folding_grinding


def get_num_oods(log_inv_rate, n_vars):
    res = match_range(
        log_inv_rate,
        range(MIN_WHIR_LOG_INV_RATE, MAX_WHIR_LOG_INV_RATE + 1),
        lambda r: get_num_oods_const_rate(r, n_vars),
    )
    return res


def get_num_oods_const_rate(log_inv_rate: Const, n_vars):
    res = match_range(
        n_vars,
        range(MIN_STACKED_N_VARS, TWO_ADICITY + WHIR_INITIAL_FOLDING_FACTOR - log_inv_rate + 1),
        lambda nv: get_num_oods_const(log_inv_rate, nv),
    )
    return res


def get_num_oods_const(log_inv_rate: Const, n_vars: Const):
    max = len(WHIR_ALL_POTENTIAL_NUM_OODS[log_inv_rate - MIN_WHIR_LOG_INV_RATE][n_vars - MIN_STACKED_N_VARS])
    num_oods = Array(max)
    for i in unroll(0, max):
        num_oods[i] = WHIR_ALL_POTENTIAL_NUM_OODS[log_inv_rate - MIN_WHIR_LOG_INV_RATE][n_vars - MIN_STACKED_N_VARS][i]
    return num_oods
