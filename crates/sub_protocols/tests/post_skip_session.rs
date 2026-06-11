//! T2' isolation tests (plan_spec v2 §8):
//! 1. golden: `prove_batched_air_sumcheck_with_factors(.., ones)` is
//!    byte-identical to `prove_batched_air_sumcheck`;
//! 2. conversion sumcheck round-trips against the existing `sumcheck_verify`;
//! 3. `new_post_skip` sessions prove the correct claim through the factors
//!    driver (brute-force ground truth).

use backend::*;
use lean_vm::{
    EF, ExtraDataForBuses, F, HALF_DIGEST_LEN, POSEIDON_8_COL_ADDR_LEFT_HI, POSEIDON_8_COL_ADDR_LEFT_LO,
    POSEIDON_8_COL_FLAG_OUT4, POSEIDON_8_COL_INPUT_START, POSEIDON_8_COL_MULTIPLICITY, POSEIDON_8_COL_OUT_LO,
    POSEIDON_8_COL_ROUND_START, Poseidon8Precompile, compute_poseidon8_witness, fill_trace_poseidon_8,
    num_cols_poseidon_8,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sub_protocols::{
    AirSumcheckSession, OuterSumcheckSession, natural_ordering_point_for_session, prove_batched_air_sumcheck,
    prove_batched_air_sumcheck_with_factors,
};

/// A valid Poseidon8 trace (all AIR constraints satisfied), as in
/// tests/prove_poseidon.rs.
fn build_poseidon_trace(log_n_rows: usize, seed: u64) -> Vec<ArenaVec<F>> {
    let n_rows = 1 << log_n_rows;
    let mut rng = StdRng::seed_from_u64(seed);
    let n_cols = num_cols_poseidon_8();
    let mut trace: Vec<ArenaVec<F>> = (0..n_cols).map(|_| ArenaVec::filled(F::ZERO, n_rows)).collect();
    for t in trace.iter_mut().skip(POSEIDON_8_COL_INPUT_START).take(WIDTH) {
        *t = ArenaVec::from_iter((0..n_rows).map(|_| rng.random()));
    }
    trace[POSEIDON_8_COL_MULTIPLICITY] = ArenaVec::filled(F::ONE, n_rows);
    trace[POSEIDON_8_COL_FLAG_OUT4] = ArenaVec::filled(F::ONE, n_rows);
    trace[POSEIDON_8_COL_ADDR_LEFT_LO] = ArenaVec::filled(F::ZERO, n_rows);
    trace[POSEIDON_8_COL_ADDR_LEFT_HI] = ArenaVec::filled(F::from_usize(HALF_DIGEST_LEN), n_rows);
    #[allow(clippy::needless_range_loop)]
    for row in 0..n_rows {
        let input: [F; WIDTH] = std::array::from_fn(|i| trace[POSEIDON_8_COL_INPUT_START + i][row]);
        let (aux, perm_state) = compute_poseidon8_witness(input);
        for i in 0..WIDTH / 2 {
            trace[POSEIDON_8_COL_OUT_LO + i][row] = perm_state[i] + input[i];
        }
        for (i, v) in aux.iter().enumerate() {
            trace[POSEIDON_8_COL_ROUND_START + i][row] = *v;
        }
    }
    fill_trace_poseidon_8(&mut trace);
    trace
}

/// Runs the two-table batched AIR sumcheck (heights 2^10 and 2^9, exercising
/// the pre-join back-loading) with the given driver and returns the proof's
/// debug rendering (covers the full transcript).
fn run_two_table_driver(traces: &[Vec<ArenaVec<F>>; 2], with_factors: bool) -> String {
    let n_constraints = Poseidon8Precompile::<false>.n_constraints();
    let mut ps = ProverState::<EF, _>::new(*get_poseidon8(), Default::default());
    ps.duplex();
    let alpha = ps.sample();
    ps.duplex();

    let mut sessions: Vec<Box<dyn OuterSumcheckSession<EF> + '_>> = Vec::new();
    for trace in traces {
        let log_n_rows = log2_strict_usize(trace[0].len());
        let air_alpha_powers: Vec<EF> = alpha.powers().collect_n(n_constraints);
        let extra_data = ExtraDataForBuses::new(&[], air_alpha_powers);
        ps.duplex();
        let eq_factor: Vec<EF> = ps.sample_vec(log_n_rows);
        let column_refs: Vec<&[F]> = trace.iter().map(|c| c.as_slice()).collect();
        let packed = MleGroupRef::<EF>::Base(column_refs).pack();
        sessions.push(Box::new(AirSumcheckSession::new(
            packed,
            eq_factor,
            EF::ZERO,
            Poseidon8Precompile::<false>,
            extra_data,
            1 << log_n_rows,
        )));
    }

    let point = if with_factors {
        let ones = vec![EF::ONE; sessions.len()];
        prove_batched_air_sumcheck_with_factors(&mut ps, &mut sessions, ones)
    } else {
        prove_batched_air_sumcheck(&mut ps, &mut sessions)
    };
    for session in &sessions {
        ps.add_extension_scalars(&session.final_column_evals());
    }
    format!("{:?}|{:?}", point, ps.into_proof())
}

/// Plan_spec v2 §9.7: with `initial_k = ones`, the factors driver must be
/// bit-exact with the original.
#[test]
fn with_factors_ones_is_byte_identical() {
    let traces = [build_poseidon_trace(10, 1), build_poseidon_trace(9, 2)];
    let old = run_two_table_driver(&traces, false);
    let new = run_two_table_driver(&traces, true);
    assert_eq!(old, new);
}

/// Plan_spec v2 §2.3: the conversion prover round-trips against the EXISTING
/// `sumcheck_verify(state, b, 2, claim, None)`, and the final value equals
/// w-MLE(s)·g-MLE(s) (MLE point = reversed round challenges).
#[test]
fn conversion_sumcheck_round_trip() {
    let mut rng = StdRng::seed_from_u64(3);
    for b in 1..=5usize {
        let m = 1usize << b;
        let weights: Vec<EF> = (0..m).map(|_| rng.random()).collect();
        let g: Vec<EF> = (0..m).map(|_| rng.random()).collect();
        let claim: EF = weights.iter().zip(&g).map(|(&w, &v)| w * v).sum();

        let mut ps = ProverState::<EF, _>::new(*get_poseidon8(), Default::default());
        let (s_prover, final_prover) = prove_weighted_block_sumcheck(&mut ps, weights.clone(), g.clone());

        let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), *get_poseidon8(), Default::default()).unwrap();
        let Evaluation { point, value } = sumcheck_verify(&mut vs, b, 2, claim, None).unwrap();

        assert_eq!(point.0, s_prover, "b={b}: challenge transcripts diverged");
        assert_eq!(value, final_prover, "b={b}: final values diverged");

        let mle_point: Vec<EF> = s_prover.iter().rev().copied().collect();
        let w_at_s = weights.evaluate(&MultilinearPoint(mle_point.clone()));
        let g_at_s = g.evaluate(&MultilinearPoint(mle_point));
        assert_eq!(value, w_at_s * g_at_s, "b={b}: w(s)·g(s) mismatch");
    }
}

/// Plan_spec v2 §8 T2' isolation: a `new_post_skip` session (extension-folded
/// columns, initial factor κ via the driver) proves exactly the brute-force
/// claim, and the verifier-side final check (mirroring verify_execution.rs)
/// passes.
#[test]
fn post_skip_session_proves_folded_claim() {
    let log_n_rows = 10usize;
    let b = 2usize;
    let n = log_n_rows - b; // folded table size 2^8
    let n_cols = num_cols_poseidon_8();
    let air = Poseidon8Precompile::<false>;
    let air_degree = air.degree_air();
    let n_constraints = air.n_constraints();

    let trace = build_poseidon_trace(log_n_rows, 4);
    let mut rng = StdRng::seed_from_u64(5);

    // Lagrange weights for a random r0 (any Σ=1 weights would do — use the
    // real kernel to also exercise it on Goldilocks).
    let r0: EF = rng.random();
    let weights = lagrange_evals_on_subgroup::<F, EF>(r0, b);
    let folded: Vec<Vec<EF>> = (0..n_cols)
        .map(|c| {
            (0..1usize << n)
                .map(|x| {
                    (0..1usize << b)
                        .map(|j| weights[j] * trace[c][(x << b) | j])
                        .sum::<EF>()
                })
                .collect()
        })
        .collect();

    // FS: sample alpha (constraint batching), eq_factor, and the public κ.
    let mut ps = ProverState::<EF, _>::new(*get_poseidon8(), Default::default());
    ps.duplex();
    let alpha = ps.sample();
    ps.duplex();
    let eq_factor: Vec<EF> = ps.sample_vec(n);
    ps.duplex();
    let kappa: EF = ps.sample();
    let extra_data = ExtraDataForBuses::new(&[], alpha.powers().collect_n(n_constraints));

    // Brute-force ground truth: sum = Σ_x eq(ef, x)·C(folded(x)), with
    // eq_factor coordinate m pairing with row bit n−1−m (the verifier's
    // natural-ordering convention).
    let mut sum = EF::ZERO;
    for x in 0..1usize << n {
        let mut w = EF::ONE;
        for (m, &ef) in eq_factor.iter().enumerate() {
            let bit = (x >> (n - 1 - m)) & 1;
            w *= if bit == 1 { ef } else { EF::ONE - ef };
        }
        let point: Vec<EF> = folded.iter().map(|c| c[x]).collect();
        let c_eval = <Poseidon8Precompile<false> as SumcheckComputation<EF>>::eval_extension(&air, &point, &extra_data);
        sum += w * c_eval;
    }

    // Session + driver with initial factor κ.
    let packed_cols: Vec<ArenaVec<EFPacking<EF>>> = folded.iter().map(|c| pack_extension(c)).collect();
    let group = MleGroupOwned::ExtensionPacked(packed_cols);
    let mut sessions: Vec<Box<dyn OuterSumcheckSession<EF> + '_>> = vec![Box::new(AirSumcheckSession::new_post_skip(
        group,
        eq_factor.clone(),
        sum,
        air,
        extra_data,
        1 << n,
    ))];
    let air_point = prove_batched_air_sumcheck_with_factors(&mut ps, &mut sessions, vec![kappa]);
    let col_evals = sessions[0].final_column_evals();
    ps.add_extension_scalars(&col_evals);

    // Verifier mirror (cf. verify_execution.rs / tests/prove_poseidon.rs).
    let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), *get_poseidon8(), Default::default()).unwrap();
    vs.duplex();
    let alpha_v = vs.sample();
    vs.duplex();
    let eq_factor_v: Vec<EF> = vs.sample_vec(n);
    vs.duplex();
    let kappa_v: EF = vs.sample();
    assert_eq!((alpha_v, &eq_factor_v, kappa_v), (alpha, &eq_factor, kappa));
    let extra_data_v = ExtraDataForBuses::new(&[], alpha_v.powers().collect_n(n_constraints));

    let Evaluation { point, value } = sumcheck_verify(&mut vs, n, air_degree + 1, kappa * sum, None).unwrap();
    assert_eq!(point.0, air_point.0);

    let col_evals_v: Vec<EF> = vs.next_extension_scalars_vec(n_cols).unwrap();
    assert_eq!(col_evals_v, col_evals);
    let constraint_eval = <Poseidon8Precompile<false> as SumcheckComputation<EF>>::eval_extension(
        &air,
        &col_evals_v,
        &extra_data_v,
    );
    let natural_point = natural_ordering_point_for_session(&point.0, n);
    let eq_val = MultilinearPoint(eq_factor_v).eq_poly_outside(&MultilinearPoint(natural_point.clone()));
    assert_eq!(value, kappa_v * eq_val * constraint_eval);

    // The transcript col_evals are the folded columns' MLEs at the natural
    // point (review invariant 2b/2c seen from the session side).
    for (c, &ev) in folded.iter().zip(&col_evals_v) {
        assert_eq!(ev, c.evaluate(&MultilinearPoint(natural_point.clone())));
    }
}
