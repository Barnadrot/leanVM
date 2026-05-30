//! Poseidon GKR for Poseidon1 (KoalaBear, t=16, alpha=3).
//!
//! Uses precomputed h_t tables + backend product sumcheck for each transition.
//! h_t(x) = <eq_p_el, T_t(cp_t(x))> precomputed in mixed F/EF arithmetic,
//! then degree-2 product sumcheck: Σ_x eq(r, x) * h_t(x) = claimed.

use backend::*;
use lean_vm::{EF, F};
use rayon::prelude::*;

const WIDTH: usize = 16;
const N_TRANSITIONS: usize = 29;

fn apply_transition_to_evals(t: usize, input: &[EF], c: &PoseidonConstants) -> [EF; WIDTH] {
    let mut state: [EF; WIDTH] = std::array::from_fn(|k| input[k]);
    match t {
        0..=3 => { let rc = &c.initial_rc[t]; for k in 0..WIDTH { state[k] += rc[k]; state[k] = state[k].cube(); } mds_circ_16(&mut state); }
        4 => { for (s, &rc) in state.iter_mut().zip(c.frc.iter()) { *s += rc; } let inp = state; for k in 0..WIDTH { state[k] = EF::ZERO; for j in 0..WIDTH { state[k] += inp[j] * c.m_i[k][j]; } } }
        5..=24 => { let r = t - 5; state[0] = state[0].cube(); if r < 19 { state[0] += c.scalar_rc[r]; } let old_s0 = state[0]; let mut new_s0 = EF::ZERO; for j in 0..WIDTH { new_s0 += state[j] * c.first_rows[r][j]; } state[0] = new_s0; for j in 1..WIDTH { state[j] += old_s0 * c.v_vecs[r][j - 1]; } }
        25..=28 => { let rc = &c.final_rc[t - 25]; for k in 0..WIDTH { state[k] += rc[k]; state[k] = state[k].cube(); } mds_circ_16(&mut state); }
        _ => unreachable!(),
    }
    state
}

struct PoseidonConstants {
    initial_rc: &'static [[F; WIDTH]],
    final_rc: &'static [[F; WIDTH]],
    frc: &'static [F; WIDTH],
    m_i: &'static [[F; WIDTH]; WIDTH],
    first_rows: &'static Vec<[F; WIDTH]>,
    v_vecs: &'static Vec<[F; WIDTH]>,
    scalar_rc: &'static Vec<F>,
}
unsafe impl Send for PoseidonConstants {}
unsafe impl Sync for PoseidonConstants {}

fn poseidon_constants() -> PoseidonConstants {
    PoseidonConstants {
        initial_rc: poseidon1_initial_constants(),
        final_rc: poseidon1_final_constants(),
        frc: poseidon1_sparse_first_round_constants(),
        m_i: poseidon1_sparse_m_i(),
        first_rows: poseidon1_sparse_first_row(),
        v_vecs: poseidon1_sparse_v(),
        scalar_rc: poseidon1_sparse_scalar_round_constants(),
    }
}

fn compute_checkpoint_states_base(input_cols: &[&[F]], n_rows: usize) -> Vec<Vec<[F; WIDTH]>> {
    let c = poseidon_constants();
    let mut checkpoints: Vec<Vec<[F; WIDTH]>> = Vec::with_capacity(N_TRANSITIONS + 1);
    let mut current: Vec<[F; WIDTH]> = (0..n_rows).map(|i| std::array::from_fn(|k| input_cols[k][i])).collect();
    checkpoints.push(current.clone());
    for r in 0..4 {
        current.par_iter_mut().for_each(|row| {
            for k in 0..WIDTH { row[k] += c.initial_rc[r][k]; row[k] = row[k].cube(); }
            mds_circ_16(row);
        });
        checkpoints.push(current.clone());
    }
    current.par_iter_mut().for_each(|row| {
        for (s, &rc) in row.iter_mut().zip(c.frc.iter()) { *s += rc; }
        let inp = *row;
        for k in 0..WIDTH { row[k] = F::ZERO; for j in 0..WIDTH { row[k] += inp[j] * c.m_i[k][j]; } }
    });
    checkpoints.push(current.clone());
    for r in 0..20 {
        current.par_iter_mut().for_each(|row| {
            row[0] = row[0].cube();
            if r < 19 { row[0] += c.scalar_rc[r]; }
            let old_s0 = row[0];
            let mut new_s0 = F::ZERO;
            for j in 0..WIDTH { new_s0 += row[j] * c.first_rows[r][j]; }
            row[0] = new_s0;
            for j in 1..WIDTH { row[j] += old_s0 * c.v_vecs[r][j - 1]; }
        });
        checkpoints.push(current.clone());
    }
    for r in 0..4 {
        current.par_iter_mut().for_each(|row| {
            for k in 0..WIDTH { row[k] += c.final_rc[r][k]; row[k] = row[k].cube(); }
            mds_circ_16(row);
        });
        checkpoints.push(current.clone());
    }
    checkpoints
}

/// Precompute h_t(x) = <eq_p_el, T_t(cp_t(x))> for all x.
/// Uses base-field checkpoint + EF eq_p_el dot product.
fn precompute_h_table(checkpoints: &[Vec<[F; WIDTH]>], t: usize, eq_p_el: &[EF; WIDTH], n_rows: usize) -> Vec<EF> {
    let c = poseidon_constants();
    let cp = &checkpoints[t + 1]; // T_t(cp_t) = cp_{t+1}
    cp.par_iter().map(|row| {
        let mut acc = EF::ZERO;
        for k in 0..WIDTH { acc += eq_p_el[k] * row[k]; }
        acc
    }).collect()
}

pub fn prove_poseidon_gkr(prover_state: &mut impl FSProver<EF>, input_cols: &[&[F]], n_rows: usize, log_n_rows: usize) -> (MultilinearPoint<EF>, Vec<EF>) {
    let c = poseidon_constants();
    let checkpoints = compute_checkpoint_states_base(input_cols, n_rows);

    prover_state.duplex();
    let p_el: Vec<EF> = prover_state.sample_vec(4);
    let mut eq_p_el_vec = eval_eq(&p_el);
    let mut eq_p_el: [EF; WIDTH] = std::array::from_fn(|k| eq_p_el_vec[k]);
    prover_state.duplex();
    let mut p_row: Vec<EF> = prover_state.sample_vec(log_n_rows);

    let eq_p_row = eval_eq(&p_row);
    let last_cp = N_TRANSITIONS;
    let mut current_claim: EF = (0..n_rows).into_par_iter()
        .map(|i| { let mut rv = EF::ZERO; for k in 0..WIDTH { rv += eq_p_el[k] * checkpoints[last_cp][i][k]; } eq_p_row[i] * rv })
        .sum();
    prover_state.add_extension_scalar(current_claim);

    for t in (0..N_TRANSITIONS).rev() {
        if t < N_TRANSITIONS - 1 { prover_state.add_extension_scalar(current_claim); }

        // Precompute h_t = <eq_p_el, cp_{t+1}> for the degree-2 product sumcheck
        let h_table = precompute_h_table(&checkpoints, t, &eq_p_el, n_rows);
        let eq_table = eval_eq(&p_row);

        // Verify claim
        let actual_sum: EF = h_table.iter().zip(eq_table.iter()).map(|(&h, &e)| h * e).sum();
        debug_assert_eq!(actual_sum, current_claim, "claim mismatch at t={t}");

        // Product sumcheck: Σ_x eq_table(x) * h_table(x) = current_claim
        let h_packed: Vec<EFPacking<EF>> = pack_extension(&h_table);
        let eq_packed: Vec<EFPacking<EF>> = pack_extension(&eq_table);

        let (point, _sum, _folded_h, _folded_eq) = run_product_sumcheck(
            &MleRef::ExtensionPacked(&h_packed),
            &MleRef::ExtensionPacked(&eq_packed),
            prover_state,
            current_claim,
            log_n_rows,
            0,
        );

        // Verify endpoint: h(r) * eq(r) should equal _sum from product sumcheck
        let eq_at_r = MultilinearPoint(p_row.clone()).eq_poly_outside(&point);
        let h_at_r: EF = h_table.evaluate(&point);
        debug_assert_eq!(_sum, h_at_r * eq_at_r, "product sumcheck endpoint mismatch");

        let cp_t_at_r: Vec<EF> = {
            let eq_r = eval_eq(&point.0);
            (0..WIDTH).map(|k| (0..n_rows).into_par_iter().map(|i| eq_r[i] * checkpoints[t][i][k]).sum()).collect()
        };
        prover_state.add_extension_scalars(&cp_t_at_r);
        prover_state.duplex();

        let alpha: Vec<EF> = prover_state.sample_vec(4);
        let eq_alpha = eval_eq(&alpha);
        current_claim = EF::ZERO;
        for k in 0..WIDTH { current_claim += eq_alpha[k] * cp_t_at_r[k]; }

        // MSB-first point from product sumcheck — need to pass to next transition
        p_row = point.0;
        eq_p_el = std::array::from_fn(|k| eq_alpha[k]);
        eq_p_el_vec = eq_alpha;
    }

    let final_input_evals: Vec<EF> = {
        let eq_final = eval_eq(&p_row);
        (0..WIDTH).map(|k| (0..n_rows).into_par_iter().map(|i| eq_final[i] * input_cols[k][i]).sum()).collect()
    };
    (MultilinearPoint(p_row), final_input_evals)
}

pub fn verify_poseidon_gkr(verifier_state: &mut impl FSVerifier<EF>, log_n_rows: usize) -> Result<(MultilinearPoint<EF>, EF, Vec<EF>), ProofError> {
    let c = poseidon_constants();
    verifier_state.duplex();
    let p_el: Vec<EF> = verifier_state.sample_vec(4);
    let mut eq_p_el = eval_eq(&p_el);
    verifier_state.duplex();
    let mut p_row: Vec<EF> = verifier_state.sample_vec(log_n_rows);
    let mut claimed = verifier_state.next_extension_scalar()?;
    let mut final_inner_evals = Vec::new();

    for t in (0..N_TRANSITIONS).rev() {
        if t < N_TRANSITIONS - 1 { claimed = verifier_state.next_extension_scalar()?; }

        // Verify product sumcheck: Σ eq(r, x) * h(x) = claimed (degree 2)
        let sc_result = sumcheck_verify(verifier_state, log_n_rows, 2, claimed, None)?;
        let challenges = sc_result.point.0;
        let final_sum = sc_result.value;

        // Read cp_t(r) from prover
        let cp_t_at_r = verifier_state.next_extension_scalars_vec(WIDTH)?;

        // Endpoint check WITHOUT division:
        let eq_at = MultilinearPoint(p_row.clone()).eq_poly_outside(&MultilinearPoint(challenges.clone()));
        let trans_out = apply_transition_to_evals(t, &cp_t_at_r, &c);
        let mut expected_h = EF::ZERO;
        for k in 0..WIDTH { expected_h += eq_p_el[k] * trans_out[k]; }
        if final_sum != eq_at * expected_h {
            eprintln!("  VERIFY FAIL t={t}: final_sum={final_sum:?}");
            eprintln!("    eq_at={eq_at:?}");
            eprintln!("    expected_h={expected_h:?}");
            eprintln!("    eq_at*h={:?}", eq_at * expected_h);
            eprintln!("    p_row len={} challenges len={}", p_row.len(), challenges.len());
            return Err(ProofError::InvalidProof);
        }

        verifier_state.duplex();
        let alpha: Vec<EF> = verifier_state.sample_vec(4);
        eq_p_el = eval_eq(&alpha);
        claimed = EF::ZERO;
        for k in 0..WIDTH { claimed += eq_p_el[k] * cp_t_at_r[k]; }
        p_row = challenges;
        if t == 0 { final_inner_evals = cp_t_at_r; }
    }
    Ok((MultilinearPoint(p_row), claimed, final_inner_evals))
}

use backend::ProofError;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_poseidon_gkr_round_trip() {
        use rand::{SeedableRng, rngs::StdRng, RngExt};
        let n_rows = 1 << 10;
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<Vec<F>> = (0..16).map(|_| (0..n_rows).map(|_| rng.random()).collect()).collect();
        let input_cols: Vec<&[F]> = input_data.iter().map(|v| v.as_slice()).collect();
        let mut prover = ProverState::new(utils::get_poseidon16().clone(), Default::default());
        let (point, evals) = super::prove_poseidon_gkr(&mut prover, &input_cols, n_rows, 10);
        let mut verifier = VerifierState::<EF, _>::new(prover.into_proof(), utils::get_poseidon16().clone(), Default::default()).unwrap();
        let (v_point, v_claim, v_evals) = super::verify_poseidon_gkr(&mut verifier, 10).unwrap();
        assert_eq!(point, v_point);
        assert_eq!(evals, v_evals);
    }
}
