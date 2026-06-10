//! V3 (pw13-3, h8): WHIR initial-fold univariate skip — round-0 statement
//! shape sweep. Every roundtrip goes through the changed protocol (skip round,
//! 16-ary Lagrange leaf fold, 16-term head weight evaluation); the sweep
//! covers the shapes production hits, especially HEAD-OVERLAP statements whose
//! inner variables reach into the skipped top-4 window
//! (inner_num_variables ∈ {n, n−1, …, n−4} ⇔ selector_num_variables ∈ {0..4}).

use fiat_shamir::{ProverState, VerifierState};
use field::{PrimeCharacteristicRing, TwoAdicField};
use koala_bear::{KoalaBear, QuinticExtensionFieldKB, default_koalabear_poseidon1_16};
use poly::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use whir::*;
use zk_alloc::ArenaVec;

type F = KoalaBear;
type EF = QuinticExtensionFieldKB;

const NUM_VARIABLES: usize = 18;

fn whir_config() -> WhirConfig<EF> {
    let params = WhirConfigBuilder {
        security_level: 124,
        max_num_variables_to_send_coeffs: 9,
        pow_bits: 16,
        folding_factor: FoldingFactor::new(7, 4),
        soundness_type: SecurityAssumption::JohnsonBound,
        starting_log_inv_rate: 2,
        rs_domain_initial_reduction_factor: 5,
    };
    WhirConfig::new(&params, NUM_VARIABLES)
}

fn random_poly(seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..1 << NUM_VARIABLES).map(|_| rng.random::<F>()).collect()
}

fn roundtrip(
    polynomial: &[F],
    prove_statements: Vec<SparseStatement<EF>>,
    verify_statements: Vec<SparseStatement<EF>>,
) -> Result<(), fiat_shamir::ProofError> {
    let poseidon16 = default_koalabear_poseidon1_16();
    let params = whir_config();
    precompute_dft_twiddles::<F>(1 << F::TWO_ADICITY);

    let mut prover_state = ProverState::new(poseidon16.clone(), Default::default());
    let mle: MleOwned<EF> = MleOwned::Base(ArenaVec::from_iter(polynomial.to_vec()));
    let witness = params.commit(&mut prover_state, &mle, 1 << NUM_VARIABLES);
    params.prove(&mut prover_state, prove_statements, witness, &mle.by_ref());

    let mut verifier_state =
        VerifierState::<EF, _>::new(prover_state.into_proof(), poseidon16, Default::default()).unwrap();
    let parsed_commitment = params.parse_commitment::<F>(&mut verifier_state)?;
    params
        .verify::<F>(&mut verifier_state, &parsed_commitment, verify_statements)
        .map(|_| ())
}

/// Σ_{hi,lo} eq(point)[hi]·tail[lo]·chunk[(hi << k) | lo] over the selector chunk.
fn naive_tailed_value(polynomial: &[F], selector: usize, point: &[EF], tail: &[EF]) -> EF {
    let k = tail.len().trailing_zeros() as usize;
    let inner = point.len() + k;
    let chunk = &polynomial[selector << inner..][..1 << inner];
    let eq = eval_eq(point);
    let mut sum = EF::ZERO;
    for (hi, &w_hi) in eq.iter().enumerate() {
        for (lo, &w_lo) in tail.iter().enumerate() {
            sum += w_hi * w_lo * chunk[(hi << k) | lo];
        }
    }
    sum
}

/// dot(weights, chunk) for a next-statement over the selector chunk.
fn next_chunk_value(polynomial: &[F], selector: usize, weights: &[EF]) -> EF {
    let inner = weights.len();
    let chunk = &polynomial[selector * inner..][..inner];
    weights.iter().zip(chunk).map(|(&w, &c)| w * c).sum()
}

/// Dense + OOD-style (expand_from_univariate) statements; corruption rejected.
#[test]
fn test_uniskip_dense_and_ood_style() {
    let polynomial = random_poly(101);
    let mut rng = StdRng::seed_from_u64(11);

    let dense_point = MultilinearPoint((0..NUM_VARIABLES).map(|_| rng.random()).collect::<Vec<EF>>());
    let dense = SparseStatement::dense(dense_point.clone(), polynomial.evaluate(&dense_point));

    let ood_style_point = MultilinearPoint::expand_from_univariate(rng.random::<EF>(), NUM_VARIABLES);
    let ood_style = SparseStatement::dense(ood_style_point.clone(), polynomial.evaluate(&ood_style_point));

    roundtrip(&polynomial, vec![dense.clone(), ood_style.clone()], vec![dense.clone(), ood_style.clone()])
        .expect("dense + ood-style roundtrip");

    let mut corrupted = dense.clone();
    corrupted.values[0].value += EF::ONE;
    assert!(
        roundtrip(&polynomial, vec![dense, ood_style.clone()], vec![corrupted, ood_style]).is_err(),
        "corrupted dense value must be rejected"
    );
}

/// Selector sweep s ∈ {0..5}: inner_num_variables = n − s ∈ {n..n−5} — covers
/// every head-overlap shape (the skipped window is the top 4 variables).
#[test]
fn test_uniskip_selector_sweep_head_overlap() {
    let polynomial = random_poly(202);
    let mut rng = StdRng::seed_from_u64(22);

    let mut statements = Vec::new();
    for s in 0..=5usize {
        let point = MultilinearPoint((0..NUM_VARIABLES - s).map(|_| rng.random()).collect::<Vec<EF>>());
        let selectors: Vec<usize> = if s == 0 { vec![0] } else { vec![0, (1 << s) - 1] };
        statements.push(SparseStatement::new(
            NUM_VARIABLES,
            point.clone(),
            selectors
                .iter()
                .map(|&selector| SparseValue {
                    selector,
                    value: polynomial.evaluate_sparse(selector, &point),
                })
                .collect(),
        ));
    }
    roundtrip(&polynomial, statements.clone(), statements.clone()).expect("selector sweep roundtrip");

    // Corrupt one head-overlap statement (s = 2 → inner = n − 2, reaches into the window).
    let mut corrupted = statements.clone();
    corrupted[2].values[0].value += EF::ONE;
    assert!(
        roundtrip(&polynomial, statements, corrupted).is_err(),
        "corrupted head-overlap value must be rejected"
    );
}

/// Tensor-tail statements: full-width (inner = n, prefix spans the skipped
/// window) and selector-shifted.
#[test]
fn test_uniskip_tailed_statements() {
    let polynomial = random_poly(303);
    let mut rng = StdRng::seed_from_u64(33);

    let tail16: Vec<EF> = (0..16).map(|_| rng.random()).collect();
    let full_prefix: Vec<EF> = (0..NUM_VARIABLES - 4).map(|_| rng.random()).collect();
    let full_tailed = SparseStatement::new_with_tail(
        NUM_VARIABLES,
        MultilinearPoint(full_prefix.clone()),
        tail16.clone(),
        vec![SparseValue::new(0, naive_tailed_value(&polynomial, 0, &full_prefix, &tail16))],
    );

    let tail8: Vec<EF> = (0..8).map(|_| rng.random()).collect();
    let short_prefix: Vec<EF> = (0..NUM_VARIABLES - 3 - 2).map(|_| rng.random()).collect();
    let shifted_tailed = SparseStatement::new_with_tail(
        NUM_VARIABLES,
        MultilinearPoint(short_prefix.clone()),
        tail8.clone(),
        vec![SparseValue::new(3, naive_tailed_value(&polynomial, 3, &short_prefix, &tail8))],
    );

    roundtrip(
        &polynomial,
        vec![full_tailed.clone(), shifted_tailed.clone()],
        vec![full_tailed.clone(), shifted_tailed.clone()],
    )
    .expect("tailed roundtrip");

    let mut corrupted = full_tailed.clone();
    corrupted.values[0].value += EF::ONE;
    assert!(
        roundtrip(&polynomial, vec![full_tailed, shifted_tailed.clone()], vec![corrupted, shifted_tailed]).is_err(),
        "corrupted tailed value must be rejected"
    );
}

/// Next and next-with-tail statements (shift-by-one weights).
#[test]
fn test_uniskip_next_statements() {
    let polynomial = random_poly(404);
    let mut rng = StdRng::seed_from_u64(44);

    let next_point: Vec<EF> = (0..NUM_VARIABLES).map(|_| rng.random()).collect();
    let next_weights = matrix_next_mle_folded(&next_point);
    let plain_next = SparseStatement::new_next(
        NUM_VARIABLES,
        MultilinearPoint(next_point.clone()),
        vec![SparseValue::new(0, next_chunk_value(&polynomial, 0, &next_weights))],
    );

    let nt_prefix: Vec<EF> = (0..NUM_VARIABLES - 4).map(|_| rng.random()).collect();
    let nt_tail: Vec<EF> = (0..16).map(|_| rng.random()).collect();
    let nt_weights = matrix_next_mle_folded_with_tail(&nt_prefix, &nt_tail);
    let next_tailed = SparseStatement::new_next_with_tail(
        NUM_VARIABLES,
        MultilinearPoint(nt_prefix.clone()),
        nt_tail.clone(),
        vec![SparseValue::new(0, next_chunk_value(&polynomial, 0, &nt_weights))],
    );

    roundtrip(
        &polynomial,
        vec![plain_next.clone(), next_tailed.clone()],
        vec![plain_next.clone(), next_tailed.clone()],
    )
    .expect("next roundtrip");

    let mut corrupted = next_tailed.clone();
    corrupted.values[0].value += EF::ONE;
    assert!(
        roundtrip(&polynomial, vec![plain_next.clone(), next_tailed], vec![plain_next, corrupted]).is_err(),
        "corrupted next-tailed value must be rejected"
    );
}

/// Production-like mixed prove: every shape in one statement list.
#[test]
fn test_uniskip_mixed_production_like() {
    let polynomial = random_poly(505);
    let mut rng = StdRng::seed_from_u64(55);

    let mut statements = Vec::new();

    let dense_point = MultilinearPoint((0..NUM_VARIABLES).map(|_| rng.random()).collect::<Vec<EF>>());
    statements.push(SparseStatement::dense(
        dense_point.clone(),
        polynomial.evaluate(&dense_point),
    ));

    for s in [1usize, 4] {
        let point = MultilinearPoint((0..NUM_VARIABLES - s).map(|_| rng.random()).collect::<Vec<EF>>());
        statements.push(SparseStatement::new(
            NUM_VARIABLES,
            point.clone(),
            vec![SparseValue {
                selector: 1,
                value: polynomial.evaluate_sparse(1, &point),
            }],
        ));
    }

    let tail16: Vec<EF> = (0..16).map(|_| rng.random()).collect();
    let prefix: Vec<EF> = (0..NUM_VARIABLES - 4).map(|_| rng.random()).collect();
    statements.push(SparseStatement::new_with_tail(
        NUM_VARIABLES,
        MultilinearPoint(prefix.clone()),
        tail16.clone(),
        vec![SparseValue::new(0, naive_tailed_value(&polynomial, 0, &prefix, &tail16))],
    ));

    let next_point: Vec<EF> = (0..NUM_VARIABLES).map(|_| rng.random()).collect();
    let next_weights = matrix_next_mle_folded(&next_point);
    statements.push(SparseStatement::new_next(
        NUM_VARIABLES,
        MultilinearPoint(next_point),
        vec![SparseValue::new(0, next_chunk_value(&polynomial, 0, &next_weights))],
    ));

    statements.push(SparseStatement::unique_value(
        NUM_VARIABLES,
        7,
        EF::from(polynomial[7]),
    ));

    roundtrip(&polynomial, statements.clone(), statements).expect("mixed production-like roundtrip");
}
