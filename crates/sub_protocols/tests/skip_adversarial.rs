//! T4' adversarial tests for the univariate-skip wire protocol (plan_spec v2
//! §2): each tampered prover message must be rejected by the verifier chain,
//! at the check the soundness argument (§6) attributes it to.
//!
//! Tampering is done through the prover FS API itself (substituting one
//! message and continuing honestly from the post-tamper Fiat–Shamir state),
//! which produces a REAL self-consistent-except-for-the-lie transcript — the
//! strongest cheap adversary for each message slot:
//!
//! * `v0` coefficient at a multiple-of-2^k index → coset-sum check rejects
//!   (`verify_univariate_skip_round`, §6.1).
//! * `v0` coefficient at a non-multiple index → coset-sum passes, the AIR
//!   final check rejects (Horner target vs honest continuation, §6.2–6.3).
//! * conversion round message (wrong weights) → terminal `w(s)·ĝ` check
//!   rejects (§6.5–6.6).
//! * one `ĝ` entry → terminal `w(s)·ĝ` check rejects (§6.6).

use backend::*;
use lean_vm::{EF, ExtraDataForBuses, F, Poseidon8Precompile};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sub_protocols::{
    AirSumcheckSession, OuterSumcheckSession, SkipAir, SkipTableInput, compute_shifted_columns,
    fold_columns_at_x_point, natural_ordering_point_for_session, prove_air_univariate_skip,
    prove_batched_air_sumcheck_with_factors,
};

const K: usize = 4;
const N: usize = 9; // single table, b = K (p = 0)
const MAX_AIR_DEGREE: usize = 8;

fn n_uni() -> usize {
    (MAX_AIR_DEGREE + 1) * ((1 << K) - 1) + 1
}

fn bit_reverse(x: usize, bits: usize) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS as usize - bits)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tamper {
    None,
    V0CosetSlot,   // index 2^K (multiple of 2^K, != 0)
    V0FreeSlot,    // index 1 (not a multiple of 2^K)
    ConversionMsg, // wrong weight vector inside the conversion sumcheck
    GhatEntry,     // ĝ[0] += 1
}

/// Runs the full §2 wire protocol (prover + verifier) on one random
/// poseidon8-AIR table with the chosen tamper, returns the verifier outcome.
fn run_protocol(tamper: Tamper) -> Result<(), ProofError> {
    let mut rng = StdRng::seed_from_u64(7);
    let air = Poseidon8Precompile::<false>;
    let alpha: EF = rng.random();
    let extra = ExtraDataForBuses::new(&[], alpha.powers().collect_n(air.n_constraints()));

    let n_rows = 1usize << N;
    let non_padded = n_rows - 5;
    let flat: Vec<ArenaVec<F>> = (0..air.n_columns())
        .map(|_| {
            let mut col: Vec<F> = (0..n_rows).map(|_| rng.random()).collect();
            for r in non_padded..n_rows {
                col[r] = col[non_padded - 1];
            }
            ArenaVec::from_slice(&col)
        })
        .collect();
    let refs: Vec<&[F]> = flat.iter().map(|c| c.as_slice()).collect();
    let shifted = compute_shifted_columns(air.n_shift_columns(), &refs);
    let columns: Vec<&[F]> = refs
        .iter()
        .copied()
        .chain(shifted.iter().map(|c| c.as_slice()))
        .collect();
    let beta: Vec<EF> = (0..N).map(|_| rng.random()).collect();

    // Honest table sum Σ_row eq(β,row)·C(row).
    let initial_sum: EF = (0..n_rows)
        .map(|row| {
            let point: Vec<EF> = columns.iter().map(|c| EF::from(c[row])).collect();
            let mut w = EF::ONE;
            for (m, &b) in beta.iter().enumerate() {
                let bit = (row >> (N - 1 - m)) & 1;
                w *= if bit == 1 { b } else { EF::ONE - b };
            }
            w * <Poseidon8Precompile<false> as SumcheckComputation<EF>>::eval_extension(&air, &point, &extra)
        })
        .sum();

    // Honest v0 from the engine on a scratch transcript (the engine writes v0
    // itself; we only harvest the coefficients).
    let v0_honest = {
        let mut scratch = ProverState::<EF, _>::new(*get_poseidon8(), Default::default());
        let skip_air = SkipAir::<EF, _>::new(&air, &extra);
        let inputs = [SkipTableInput {
            columns: columns.clone(),
            eq_factor: beta.clone(),
            computation: &skip_air,
            sum: initial_sum,
            non_padded_n_rows: non_padded,
        }];
        prove_air_univariate_skip(&mut scratch, &inputs, K, n_uni()).v0_coeffs
    };

    let mut v0 = v0_honest;
    match tamper {
        Tamper::V0CosetSlot => v0[1 << K] += EF::ONE,
        Tamper::V0FreeSlot => v0[1] += EF::ONE,
        _ => {}
    }

    // ---------------- Prover (manual §2 chain, honest after the tamper) ----
    let b = K; // p = 0 for the single table
    let mut ps = ProverState::<EF, _>::new(*get_poseidon8(), Default::default());
    ps.add_extension_scalars(&v0);
    let r0: EF = ps.sample();

    let lag = lagrange_evals_on_subgroup::<F, EF>(r0, K);
    let sub = sub_block_lagrange_from_global(&lag, b);
    let ell: Vec<EF> = (0..1usize << b).map(|j| sub[bit_reverse(j, b)]).collect();

    // Naive Lagrange block-fold.
    let n_x = n_rows >> b;
    let folded: Vec<ArenaVec<EF>> = columns
        .iter()
        .map(|col| {
            ArenaVec::from_iter(
                (0..n_x).map(|x| ell.iter().enumerate().map(|(j, &l)| l * col[(x << b) | j]).sum::<EF>()),
            )
        })
        .collect();

    let beta_x = beta[..N - b].to_vec();
    let eq_blk: ArenaVec<EF> = eval_eq(&beta[N - b..]);
    let kappa: EF = eq_blk
        .iter()
        .enumerate()
        .map(|(j, &w)| w * lag[(1 << K) - (1 << b) + bit_reverse(j, b)])
        .sum();
    // Honest post-skip claim Q'(ρ) = Σ_x eq(β^x,x)·C(folded(x)).
    let eq_x: ArenaVec<EF> = eval_eq(&beta_x);
    let sum_post: EF = (0..n_x)
        .map(|x| {
            let point: Vec<EF> = folded.iter().map(|c| c[x]).collect();
            eq_x[x] * <Poseidon8Precompile<false> as SumcheckComputation<EF>>::eval_extension(&air, &point, &extra)
        })
        .sum();

    let session_extra = ExtraDataForBuses::new(&[], alpha.powers().collect_n(air.n_constraints()));
    let mut sessions: Vec<Box<dyn OuterSumcheckSession<EF> + '_>> = vec![Box::new(AirSumcheckSession::new_post_skip(
        MleGroupOwned::Extension(folded),
        beta_x.clone(),
        sum_post,
        air,
        session_extra,
        non_padded.div_ceil(1 << b),
    ))];
    let c_post = prove_batched_air_sumcheck_with_factors(&mut ps, &mut sessions, vec![kappa]);

    let fold_evals = sessions[0].final_column_evals();
    ps.add_extension_scalars(&fold_evals);

    let gamma: EF = ps.sample();
    let m_t = fold_evals.len();
    let gamma_pows: Vec<EF> = gamma.powers().collect_n(m_t);
    let x_nat = natural_ordering_point_for_session(&c_post.0, N - b);
    let g = fold_columns_at_x_point::<EF>(&columns, &x_nat, b);
    let mut g_gamma = vec![EF::ZERO; 1 << b];
    for (g_c, &gp) in g.iter().zip(&gamma_pows) {
        for (slot, &gj) in g_gamma.iter_mut().zip(g_c) {
            *slot += gj * gp;
        }
    }
    let weights_used = if tamper == Tamper::ConversionMsg {
        let mut w = ell.clone();
        w[0] += EF::ONE;
        w
    } else {
        ell.clone()
    };
    let (s_chals, _) = prove_weighted_block_sumcheck(&mut ps, weights_used, g_gamma);
    let s_rev: Vec<EF> = s_chals.iter().rev().copied().collect();
    let mut ghat: Vec<EF> = g
        .iter()
        .map(|g_c| g_c.evaluate(&MultilinearPoint(s_rev.clone())))
        .collect();
    if tamper == Tamper::GhatEntry {
        ghat[0] += EF::ONE;
    }
    ps.add_extension_scalars(&ghat);

    // ---------------- Verifier (the §2.1 table, mirrored) ------------------
    let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), *get_poseidon8(), Default::default())?;
    let skip = verify_univariate_skip_round(&mut vs, K, n_uni(), initial_sum)?;
    let Evaluation {
        point: c_post_v,
        value: t_final,
    } = sumcheck_verify(&mut vs, N - K, MAX_AIR_DEGREE + 1, skip.target, None)?;

    let col_evals = vs.next_extension_scalars_vec(m_t)?;
    let constraint_eval =
        <Poseidon8Precompile<false> as SumcheckComputation<EF>>::eval_extension(&air, &col_evals, &extra);
    let x_nat_v = natural_ordering_point_for_session(&c_post_v.0, N - b);
    let eq_val = MultilinearPoint(beta_x.clone()).eq_poly_outside(&MultilinearPoint(x_nat_v));
    let eq_blk_v: ArenaVec<EF> = eval_eq(&beta[N - b..]);
    let kappa_v: EF = eq_blk_v
        .iter()
        .enumerate()
        .map(|(j, &w)| w * skip.lagrange_on_d[(1 << K) - (1 << b) + bit_reverse(j, b)])
        .sum();
    if kappa_v * eq_val * constraint_eval != t_final {
        return Err(ProofError::InvalidProof);
    }

    let gamma_v: EF = vs.sample();
    let gamma_pows_v: Vec<EF> = gamma_v.powers().collect_n(m_t);
    let v_t: EF = col_evals.iter().zip(&gamma_pows_v).map(|(&e, &g)| e * g).sum();
    let Evaluation {
        point: s_v,
        value: v_final,
    } = sumcheck_verify(&mut vs, b, 2, v_t, None)?;
    let ghat_v = vs.next_extension_scalars_vec(m_t)?;
    let sub_v = sub_block_lagrange_from_global(&skip.lagrange_on_d, b);
    let s_rev_v: Vec<EF> = s_v.0.iter().rev().copied().collect();
    let eq_s: ArenaVec<EF> = eval_eq(&s_rev_v);
    let w_s: EF = (0..1usize << b).map(|j| sub_v[bit_reverse(j, b)] * eq_s[j]).sum();
    let ghat_gamma: EF = ghat_v.iter().zip(&gamma_pows_v).map(|(&gh, &p)| gh * p).sum();
    if w_s * ghat_gamma != v_final {
        return Err(ProofError::InvalidProof);
    }
    Ok(())
}

#[test]
fn honest_transcript_accepts() {
    run_protocol(Tamper::None).unwrap();
}

#[test]
fn tampered_v0_coset_slot_rejected_by_coset_sum() {
    assert!(run_protocol(Tamper::V0CosetSlot).is_err());
}

#[test]
fn tampered_v0_free_slot_rejected_by_final_check() {
    assert!(run_protocol(Tamper::V0FreeSlot).is_err());
}

#[test]
fn tampered_conversion_message_rejected() {
    assert!(run_protocol(Tamper::ConversionMsg).is_err());
}

#[test]
fn tampered_ghat_rejected() {
    assert!(run_protocol(Tamper::GhatEntry).is_err());
}
