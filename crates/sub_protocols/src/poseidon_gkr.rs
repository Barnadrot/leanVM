//! Layered-circuit GKR for Poseidon1 (KoalaBear, t=16, alpha=3).
//!
//! Proves that the 68 intermediate column evaluations at r_air are the correct
//! MLEs by chaining checkpoint-to-checkpoint transitions through the Poseidon
//! round function.
//!
//! Transition structure (matching the trace generator):
//! - 2 full-round-pair transitions (inputs→cp1, cp1→cp2): degree 9
//! - 1 linear transition (cp2→partial-ready): degree 1, folded into first partial
//! - 20 partial round transitions: degree 3 each
//! - 1 full-round-pair transition (after-partials→cp3): degree 9
//! Total: 24 checkpoint transitions.
//!
//! The final 2 rounds (cp3→output) are NOT covered by the GKR — they're
//! verified by the AIR constraints using GKR-verified cp3 + memory-bound I/O.

use backend::*;
use lean_vm::{EF, F};

const WIDTH: usize = 16;

/// Per-row full state at each checkpoint boundary.
/// checkpoint_states[c][i] = 16-element state at row i after checkpoint c.
/// c=0: inputs, c=1: after beginning pair 0, c=2: after beginning pair 1,
/// c=3: after transition (frc+M_i), c=4..23: after each partial round,
/// c=24: after ending pair.
pub fn compute_checkpoint_states(
    input_cols: &[&[F]],
    n_rows: usize,
) -> Vec<Vec<[EF; WIDTH]>> {
    let initial_rc = poseidon1_initial_constants();
    let final_rc = poseidon1_final_constants();
    let frc = poseidon1_sparse_first_round_constants();
    let m_i = poseidon1_sparse_m_i();
    let first_rows = poseidon1_sparse_first_row();
    let v_vecs = poseidon1_sparse_v();
    let scalar_rc = poseidon1_sparse_scalar_round_constants();

    let mut checkpoints: Vec<Vec<[EF; WIDTH]>> = Vec::new();

    // Checkpoint 0: inputs
    let mut current: Vec<[EF; WIDTH]> = (0..n_rows)
        .map(|i| std::array::from_fn(|k| EF::from(input_cols[k][i])))
        .collect();
    checkpoints.push(current.clone());

    // Checkpoint 1: after beginning full round pair 0 (initial_rc[0], initial_rc[1])
    apply_2_full_rounds(&mut current, &initial_rc[0], &initial_rc[1]);
    checkpoints.push(current.clone());

    // Checkpoint 2: after beginning full round pair 1 (initial_rc[2], initial_rc[3])
    apply_2_full_rounds(&mut current, &initial_rc[2], &initial_rc[3]);
    checkpoints.push(current.clone());

    // Checkpoint 3: after transition (add frc, multiply by M_i) — LINEAR
    for row in current.iter_mut() {
        for (s, &c) in row.iter_mut().zip(frc.iter()) {
            *s += c;
        }
        let input = *row;
        for k in 0..WIDTH {
            let coeffs: [EF; WIDTH] = m_i[k].map(EF::from);
            row[k] = EF::ZERO;
            for j in 0..WIDTH {
                row[k] += input[j] * coeffs[j];
            }
        }
    }
    checkpoints.push(current.clone());

    // Checkpoints 4..23: after each partial round (20 rounds)
    for r in 0..20 {
        for row in current.iter_mut() {
            row[0] = row[0].cube();
            if r < 19 {
                row[0] += scalar_rc[r];
            }
            let old_s0 = row[0];
            let fr: [EF; WIDTH] = first_rows[r].map(EF::from);
            let mut new_s0 = EF::ZERO;
            for j in 0..WIDTH {
                new_s0 += row[j] * fr[j];
            }
            row[0] = new_s0;
            for j in 1..WIDTH {
                row[j] += old_s0 * v_vecs[r][j - 1];
            }
        }
        checkpoints.push(current.clone());
    }

    // Checkpoint 24: after ending full round pair (final_rc[0], final_rc[1])
    apply_2_full_rounds(&mut current, &final_rc[0], &final_rc[1]);
    checkpoints.push(current.clone());

    // Note: the LAST 2 ending rounds (final_rc[2], final_rc[3]) + compression logic
    // are NOT checkpointed — they're verified by the AIR constraints.

    checkpoints // 25 checkpoints: 0..24
}

fn apply_2_full_rounds(rows: &mut [[EF; WIDTH]], rc1: &[F; WIDTH], rc2: &[F; WIDTH]) {
    for row in rows.iter_mut() {
        for k in 0..WIDTH {
            row[k] += rc1[k];
            row[k] = row[k].cube();
        }
        mds_circ_16(row);
        for k in 0..WIDTH {
            row[k] += rc2[k];
            row[k] = row[k].cube();
        }
        mds_circ_16(row);
    }
}

/// Apply a checkpoint transition to a set of element evaluations.
/// Used by the verifier to check the sumcheck endpoint.
fn apply_transition_to_evals(transition: usize, input: &[EF]) -> [EF; WIDTH] {
    let initial_rc = poseidon1_initial_constants();
    let final_rc = poseidon1_final_constants();
    let frc = poseidon1_sparse_first_round_constants();
    let m_i = poseidon1_sparse_m_i();
    let first_rows = poseidon1_sparse_first_row();
    let v_vecs = poseidon1_sparse_v();
    let scalar_rc = poseidon1_sparse_scalar_round_constants();

    let mut state: [EF; WIDTH] = std::array::from_fn(|k| input[k]);

    match transition {
        0 => {
            // 2 beginning full rounds (initial_rc[0], initial_rc[1])
            apply_2_full_rounds_single(&mut state, &initial_rc[0], &initial_rc[1]);
        }
        1 => {
            // 2 beginning full rounds (initial_rc[2], initial_rc[3])
            apply_2_full_rounds_single(&mut state, &initial_rc[2], &initial_rc[3]);
        }
        2 => {
            // Linear transition: add frc, multiply by M_i
            for (s, &c) in state.iter_mut().zip(frc.iter()) {
                *s += c;
            }
            let input_copy = state;
            for k in 0..WIDTH {
                let coeffs: [EF; WIDTH] = m_i[k].map(EF::from);
                state[k] = EF::ZERO;
                for j in 0..WIDTH {
                    state[k] += input_copy[j] * coeffs[j];
                }
            }
        }
        t if (3..=22).contains(&t) => {
            // Partial round (t-3 = round index 0..19)
            let r = t - 3;
            state[0] = state[0].cube();
            if r < 19 {
                state[0] += scalar_rc[r];
            }
            let old_s0 = state[0];
            let fr: [EF; WIDTH] = first_rows[r].map(EF::from);
            let mut new_s0 = EF::ZERO;
            for j in 0..WIDTH {
                new_s0 += state[j] * fr[j];
            }
            state[0] = new_s0;
            for j in 1..WIDTH {
                state[j] += old_s0 * v_vecs[r][j - 1];
            }
        }
        23 => {
            // 2 ending full rounds (final_rc[0], final_rc[1])
            apply_2_full_rounds_single(&mut state, &final_rc[0], &final_rc[1]);
        }
        _ => unreachable!("invalid transition index"),
    }

    state
}

fn apply_2_full_rounds_single(state: &mut [EF; WIDTH], rc1: &[F; WIDTH], rc2: &[F; WIDTH]) {
    for k in 0..WIDTH {
        state[k] += rc1[k];
        state[k] = state[k].cube();
    }
    mds_circ_16(state);
    for k in 0..WIDTH {
        state[k] += rc2[k];
        state[k] = state[k].cube();
    }
    mds_circ_16(state);
}

/// Degree of the sumcheck polynomial per variable.
/// = 1 (from eq factor) + degree of the transition function.
fn sumcheck_degree(transition: usize) -> usize {
    match transition {
        0 | 1 | 23 => 10,   // eq(1) * 2 full rounds(3^2=9) = 10
        2 => 2,              // eq(1) * linear(1) = 2
        3..=22 => 4,         // eq(1) * partial round cube(3) = 4
        _ => unreachable!(),
    }
}

/// Prove the Poseidon GKR.
/// Returns (final_point, final_input_evals) — the input claim that needs
/// combined-GKR verification at the returned point.
pub fn prove_poseidon_gkr(
    prover_state: &mut impl FSProver<EF>,
    input_cols: &[&[F]],
    n_rows: usize,
    log_n_rows: usize,
) -> (MultilinearPoint<EF>, Vec<EF>) {
    let checkpoints = compute_checkpoint_states(input_cols, n_rows);
    // 25 checkpoints: 0=inputs, 1=cp1, 2=cp2, 3=transition, 4..23=partials, 24=cp3

    // Sample element-index challenge
    prover_state.duplex();
    let p_el: Vec<EF> = prover_state.sample_vec(4);
    let mut eq_p_el = eval_eq(&p_el);

    // Sample initial row-index challenge
    prover_state.duplex();
    let mut p_row: Vec<EF> = prover_state.sample_vec(log_n_rows);

    // Initial claim: the LAST checkpoint (24) evaluated at (p_row, p_el)
    let eq_p_row = eval_eq(&p_row);
    let mut current_claim: EF = {
        let mut v = EF::ZERO;
        for i in 0..n_rows {
            let mut row_val = EF::ZERO;
            for k in 0..WIDTH {
                row_val += eq_p_el[k] * checkpoints[24][i][k];
            }
            v += eq_p_row[i] * row_val;
        }
        v
    };
    prover_state.add_extension_scalar(current_claim);

    // Process transitions in REVERSE: 23, 22, ..., 0
    for t in (0..24).rev() {
        let degree = sumcheck_degree(t);
        let prev = &checkpoints[t];
        let n = 1usize << log_n_rows;

        // Send the claimed value for this transition (for recursion circuit FS)
        if t < 23 {
            prover_state.add_extension_scalar(current_claim);
        }

        // Fold prev alongside eq_table during the sumcheck
        let mut eq_table = eval_eq(&p_row);
        let mut folded_prev: Vec<[EF; WIDTH]> = prev.clone();
        let mut challenges = Vec::with_capacity(log_n_rows);

        for v in 0..log_n_rows {
            let half = n >> (v + 1);
            let n_evals = degree + 1;
            let mut evals = vec![EF::ZERO; n_evals];

            for point in 0..n_evals {
                let pt = EF::from_usize(point);
                let mut sum = EF::ZERO;
                for h in 0..half {
                    let eq_val = eq_table[2 * h] * (EF::ONE - pt) + eq_table[2 * h + 1] * pt;
                    let prev_interp: [EF; WIDTH] = std::array::from_fn(|k| {
                        folded_prev[2 * h][k] * (EF::ONE - pt) + folded_prev[2 * h + 1][k] * pt
                    });
                    let round_out = apply_transition_to_evals(t, &prev_interp);
                    let mut weighted = EF::ZERO;
                    for k in 0..WIDTH {
                        weighted += eq_p_el[k] * round_out[k];
                    }
                    sum += eq_val * weighted;
                }
                evals[point] = sum;
            }

            let coeffs = evals_to_coeffs(&evals);
            prover_state.add_extension_scalars(&coeffs);
            let r_v: EF = prover_state.sample();
            challenges.push(r_v);

            // Fold eq_table and prev with challenge r_v
            let mut new_eq = Vec::with_capacity(half);
            let mut new_prev = Vec::with_capacity(half);
            for h in 0..half {
                new_eq.push(eq_table[2 * h] * (EF::ONE - r_v) + eq_table[2 * h + 1] * r_v);
                new_prev.push(std::array::from_fn(|k| {
                    folded_prev[2 * h][k] * (EF::ONE - r_v) + folded_prev[2 * h + 1][k] * r_v
                }));
            }
            eq_table = new_eq;
            folded_prev = new_prev;

            // Update claim via coefficient evaluation at challenge
            current_claim = coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r_v + c);
        }

        // After all rounds: folded_prev[0] = prev evaluated at the endpoint
        debug_assert_eq!(folded_prev.len(), 1);
        debug_assert_eq!(eq_table.len(), 1);
        let inner_evals: Vec<EF> = folded_prev[0].to_vec();

        // Prover-side endpoint sanity check
        #[cfg(debug_assertions)]
        {
            let trans_out = apply_transition_to_evals(t, &inner_evals);
            let mut h_val = EF::ZERO;
            for k in 0..WIDTH { h_val += eq_p_el[k] * trans_out[k]; }
            debug_assert_eq!(eq_table[0] * h_val, current_claim, "prover endpoint mismatch at t={t}");
        }

        prover_state.add_extension_scalars(&inner_evals);

        // Sample alpha for element-index batching
        prover_state.duplex();
        let alpha: Vec<EF> = prover_state.sample_vec(4);
        let eq_alpha = eval_eq(&alpha);

        // Next claim: prev_state(endpoint, alpha)
        current_claim = EF::ZERO;
        for k in 0..WIDTH {
            current_claim += eq_alpha[k] * inner_evals[k];
        }

        challenges.reverse();
        p_row = challenges;
        eq_p_el = eq_alpha;
    }

    // Final claim: input state at the endpoint p_row
    let final_input_evals: Vec<EF> = {
        let eq_final = eval_eq(&p_row);
        (0..WIDTH)
            .map(|k| {
                let mut v = EF::ZERO;
                for i in 0..n_rows {
                    v += eq_final[i] * EF::from(input_cols[k][i]);
                }
                v
            })
            .collect()
    };

    (MultilinearPoint(p_row), final_input_evals)
}

fn interpolate_uni(evals: &[EF], x: EF) -> EF {
    let n = evals.len();
    let mut result = EF::ZERO;
    for i in 0..n {
        let mut basis = EF::ONE;
        for j in 0..n {
            if j != i {
                let i_f = EF::from_usize(i);
                let j_f = EF::from_usize(j);
                basis *= (x - j_f) / (i_f - j_f);
            }
        }
        result += evals[i] * basis;
    }
    result
}

/// Convert evaluations at 0, 1, ..., d to coefficient form.
fn evals_to_coeffs(evals: &[EF]) -> Vec<EF> {
    let n = evals.len();
    let mut coeffs = vec![EF::ZERO; n];
    for i in 0..n {
        let mut basis_coeffs = vec![EF::ZERO; n];
        basis_coeffs[0] = EF::ONE;
        let mut denom = EF::ONE;
        for j in 0..n {
            if j == i { continue; }
            denom *= EF::from_usize(i) - EF::from_usize(j);
            let j_f = EF::from_usize(j);
            for k in (1..n).rev() {
                basis_coeffs[k] = basis_coeffs[k - 1] - j_f * basis_coeffs[k];
            }
            basis_coeffs[0] = -j_f * basis_coeffs[0];
        }
        let inv_denom = EF::ONE / denom;
        for k in 0..n {
            coeffs[k] += evals[i] * basis_coeffs[k] * inv_denom;
        }
    }
    coeffs
}

/// Verify the Poseidon GKR.
pub fn verify_poseidon_gkr(
    verifier_state: &mut impl FSVerifier<EF>,
    log_n_rows: usize,
) -> Result<(MultilinearPoint<EF>, EF, Vec<EF>), ProofError> {
    verifier_state.duplex();
    let p_el: Vec<EF> = verifier_state.sample_vec(4);
    let mut eq_p_el = eval_eq(&p_el);

    verifier_state.duplex();
    let mut p_row: Vec<EF> = verifier_state.sample_vec(log_n_rows);

    let mut claimed = verifier_state.next_extension_scalar()?;

    let mut final_inner_evals = Vec::new();

    for t in (0..24).rev() {
        let degree = sumcheck_degree(t);
        let mut challenges = Vec::with_capacity(log_n_rows);

        if t < 23 {
            claimed = verifier_state.next_extension_scalar()?;
        }

        for _v in 0..log_n_rows {
            let n_coeffs = degree + 1;
            let coeffs = verifier_state.next_extension_scalars_vec(n_coeffs)?;

            // p(0) + p(1) == claimed
            let p0 = coeffs[0];
            let p1: EF = coeffs.iter().copied().sum();
            if p0 + p1 != claimed {
                return Err(ProofError::InvalidProof);
            }

            let r_v: EF = verifier_state.sample();
            challenges.push(r_v);
            // Evaluate polynomial at r_v using Horner's method
            claimed = coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r_v + c);
        }

        let inner_evals = verifier_state.next_extension_scalars_vec(WIDTH)?;

        // Verify endpoint: claimed == eq(p_row, endpoint) * Σ_k eq_p_el[k] * transition(inner_evals)[k]
        let trans_out = apply_transition_to_evals(t, &inner_evals);
        let mut h_val = EF::ZERO;
        for k in 0..WIDTH {
            h_val += eq_p_el[k] * trans_out[k];
        }
        // eq(p_row, endpoint): the folding pairs p_row[v] with challenges[n-1-v]
        // because eval_eq uses MSB-first ordering (p_row[0] = MSB of index)
        let mut challenges_rev = challenges.clone();
        challenges_rev.reverse();
        let eq_at_endpoint = MultilinearPoint(p_row.clone())
            .eq_poly_outside(&MultilinearPoint(challenges_rev));
        let check = eq_at_endpoint * h_val;

        if claimed != check {
            for k in 0..4.min(WIDTH) {
            }

            // Also try without eq factor
            return Err(ProofError::InvalidProof);
        }

        // Sample alpha for element-index batching
        verifier_state.duplex();
        let alpha: Vec<EF> = verifier_state.sample_vec(4);
        eq_p_el = eval_eq(&alpha);

        claimed = EF::ZERO;
        for k in 0..WIDTH {
            claimed += eq_p_el[k] * inner_evals[k];
        }

        // The folding pairs p_row[v] with challenges[n-1-v], so reverse for next round
        challenges.reverse();
        p_row = challenges;

        if t == 0 {
            final_inner_evals = inner_evals;
        }
    }

    Ok((MultilinearPoint(p_row), claimed, final_inner_evals))
}

use backend::ProofError;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_evals_to_coeffs() {
        // Polynomial p(x) = 3 + 2x + 5x^2 (degree 2, 3 evaluation points)
        // p(0) = 3, p(1) = 10, p(2) = 27
        let evals = vec![EF::from_usize(3), EF::from_usize(10), EF::from_usize(27)];
        let coeffs = super::evals_to_coeffs(&evals);
        assert_eq!(coeffs[0], EF::from_usize(3), "c0");
        assert_eq!(coeffs[1], EF::from_usize(2), "c1");
        assert_eq!(coeffs[2], EF::from_usize(5), "c2");

        // Verify p(0)+p(1) = c0 + sum(c_i)
        let p0 = coeffs[0];
        let p1: EF = coeffs.iter().copied().sum();
        assert_eq!(p0 + p1, evals[0] + evals[1]);
    }

    #[test]
    fn test_eq_folding() {
        let p = vec![EF::from_usize(3), EF::from_usize(7)];
        let eq_table = eval_eq::<EF>(&p);

        let r0 = EF::from_usize(5);
        let r1 = EF::from_usize(11);


        // Fold round 0 (pairs 0,1 and 2,3) with r0
        let new0 = eq_table[0] * (EF::ONE - r0) + eq_table[1] * r0;
        let new1 = eq_table[2] * (EF::ONE - r0) + eq_table[3] * r0;

        // Fold round 1 with r1
        let folded = new0 * (EF::ONE - r1) + new1 * r1;

        let direct = MultilinearPoint(p.clone()).eq_poly_outside(&MultilinearPoint(vec![r0, r1]));
        let reversed = MultilinearPoint(p).eq_poly_outside(&MultilinearPoint(vec![r1, r0]));

        assert!(folded == direct || folded == reversed, "eq folding must match one ordering");
    }

    #[test]
    fn test_mds_dense_vs_circ() {
        // Compute dense MDS
        let mds: [[F; 16]; 16] = {
            let mut mat = [[F::ZERO; 16]; 16];
            for j in 0..16 {
                let mut e = [F::ZERO; 16];
                e[j] = F::ONE;
                mds_circ_16(&mut e);
                for i in 0..16 { mat[i][j] = e[i]; }
            }
            mat
        };

        // Test with random EF values
        let state: [EF; 16] = std::array::from_fn(|i| EF::from_usize(100 + i * 37));

        // mds_circ_16 result
        let mut circ_result = state;
        mds_circ_16(&mut circ_result);

        // Dense matrix result
        let mut dense_result = [EF::ZERO; 16];
        for k in 0..16 {
            for j in 0..16 {
                dense_result[k] += EF::from(mds[k][j]) * state[j];
            }
        }

        for k in 0..16 {
            assert_eq!(circ_result[k], dense_result[k], "MDS mismatch at element {k}");
        }
    }
}
