//! Prove→verify transcript roundtrips for the WHIR product-skip orchestration
//! (pw13-3 V2): `run_product_sumcheck_with_skip` / `verify_product_sumcheck_with_skip`.
//!
//! FS call sequence under test (the executable discipline V3/V4/V5 mirror):
//!   prover:   add_extension_scalars(31 coeffs) → pow_grinding → sample r0
//!             → per linear round: add_sumcheck_polynomial(3, None) → pow → sample
//!   verifier: next_extension_scalars_vec(31) → window dot(coeffs, S) == sum
//!             → check_pow_grinding → sample → Horner; then legacy rounds.
//!
//! Corruption surfacing (documented + asserted):
//!   * any of the 31 skip coefficients   → round-0 window identity (InvalidProof)
//!   * grinding witness                  → InvalidGrindingWitness
//!   * linear-round wire coefficient     → absorbed by c0-elision into a
//!     DIFFERENT (point, sum) binding — rejected by the caller's final oracle
//!     check (same property as the legacy product sumcheck); asserted as
//!     `(point, sum) != honest`.

use field::*;
use fiat_shamir::*;
use koala_bear::{KoalaBear, QuinticExtensionFieldKB};
use poly::*;
use sumcheck::{
    compute_product_skip_poly, compute_product_sumcheck_polynomial, run_product_sumcheck_with_skip,
    verify_product_sumcheck_with_skip,
};
use symetric::get_poseidon16;

type F = KoalaBear;
type EF = QuinticExtensionFieldKB;

fn test_scalar_f(i: usize) -> F {
    F::from_usize(3).exp_u64(7 * i as u64 + 5)
}
fn test_scalar_ef(i: usize) -> EF {
    EF::from_basis_coefficients_fn(|j| test_scalar_f(13 * i + j))
}

fn build_inputs(n: usize) -> (Vec<F>, Vec<EF>) {
    let f: Vec<F> = (0..1 << n).map(test_scalar_f).collect();
    let w: Vec<EF> = (0..1 << n).map(|i| test_scalar_ef(i + 71)).collect();
    (f, w)
}

fn naive_cube_sum(f: &[F], w: &[EF]) -> EF {
    f.iter().zip(w).map(|(&fv, &wv)| wv * fv).fold(EF::ZERO, |a, b| a + b)
}

fn unpack_owned(m: &MleOwned<EF>) -> Vec<EF> {
    match m {
        MleOwned::Extension(v) => v.to_vec(),
        MleOwned::ExtensionPacked(v) => unpack_extension::<EF, Vec<EF>>(v),
        _ => panic!("orchestration outputs are extension-typed"),
    }
}

fn prover_state() -> ProverState<EF, symetric::Poseidon16> {
    let mut ps = ProverState::<EF, _>::new(get_poseidon16().clone(), Default::default());
    ps.duplex(); // fresh state has a stale rate; production absorbs a commitment first
    ps
}

#[allow(clippy::type_complexity)]
fn roundtrip(n: usize, k: usize, n_rounds: usize, pow_bits: usize, packed: bool) -> (MultilinearPoint<EF>, EF) {
    let (f, w) = build_inputs(n);
    let sum = naive_cube_sum(&f, &w);
    let f_ref = MleRef::<EF>::Base(&f);
    let w_ref = MleRef::<EF>::Extension(&w);

    let mut ps = prover_state();
    let (point_p, sum_p, fa, fb) = if packed {
        let (fp, wp) = (f_ref.pack(), w_ref.pack());
        run_product_sumcheck_with_skip(&fp.by_ref(), &wp.by_ref(), &mut ps, sum, k, n_rounds, pow_bits)
    } else {
        run_product_sumcheck_with_skip(&f_ref, &w_ref, &mut ps, sum, k, n_rounds, pow_bits)
    };

    // Sumcheck invariant after binding n_rounds variables: the running sum is
    // the dot product of the folded operands over the remaining cube.
    assert_eq!(point_p.len(), 1 + n_rounds - k, "point shape [r0, linear...]");
    let (fav, fbv) = (unpack_owned(&fa), unpack_owned(&fb));
    assert_eq!(fav.len(), 1 << (n - n_rounds));
    let dot = fav.iter().zip(&fbv).map(|(&a, &b)| a * b).fold(EF::ZERO, |x, y| x + y);
    assert_eq!(dot, sum_p, "prover fold consistency (n={n} k={k} rounds={n_rounds})");

    let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), get_poseidon16().clone(), Default::default()).unwrap();
    vs.duplex();
    let mut claimed = sum;
    let point_v = verify_product_sumcheck_with_skip(&mut vs, &mut claimed, k, n_rounds, pow_bits).unwrap();
    vs.check_fully_consumed().unwrap();

    assert_eq!(point_v.0, point_p.0, "transcript-identical challenge points");
    assert_eq!(claimed, sum_p, "verifier final target == prover running sum");
    (point_p, sum_p)
}

#[test]
fn test_roundtrip_shapes() {
    // Production-like (packed Base×ExtensionPacked), 7 rounds, k=4.
    let _ = roundtrip(14, 4, 7, 0, true);
    // With grinding on every event.
    let _ = roundtrip(14, 4, 7, 16, true);
    // Unpacked variant.
    let _ = roundtrip(12, 4, 7, 0, false);
    // k=3 and a single linear round.
    let _ = roundtrip(10, 3, 5, 0, true);
    let _ = roundtrip(12, 4, 5, 16, true);
    // Skip-only edge (n_rounds == k).
    let _ = roundtrip(10, 4, 4, 0, true);
}

#[test]
fn test_roundtrip_extension_times_extension() {
    let n = 12;
    let (fb, w) = build_inputs(n);
    let f: Vec<EF> = fb.iter().map(|&x| EF::from(x)).collect();
    let sum = f.iter().zip(&w).map(|(&a, &b)| a * b).fold(EF::ZERO, |x, y| x + y);
    let f_ref = MleRef::<EF>::Extension(&f);
    let w_ref = MleRef::<EF>::Extension(&w);
    let (fp, wp) = (f_ref.pack(), w_ref.pack());

    let mut ps = prover_state();
    let (point_p, sum_p, _, _) = run_product_sumcheck_with_skip(&fp.by_ref(), &wp.by_ref(), &mut ps, sum, 4, 6, 0);

    let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), get_poseidon16().clone(), Default::default()).unwrap();
    vs.duplex();
    let mut claimed = sum;
    let point_v = verify_product_sumcheck_with_skip(&mut vs, &mut claimed, 4, 6, 0).unwrap();
    assert_eq!(point_v.0, point_p.0);
    assert_eq!(claimed, sum_p);
}

/// Any tampered skip coefficient is rejected at the round-0 window identity.
#[test]
fn test_skip_coeff_corruption_rejected_at_window_identity() {
    let (n, k, n_rounds) = (12usize, 4usize, 6usize);
    let (f, w) = build_inputs(n);
    let sum = naive_cube_sum(&f, &w);
    let f_ref = MleRef::<EF>::Base(&f);
    let w_ref = MleRef::<EF>::Extension(&w);
    let (fp, wp) = (f_ref.pack(), w_ref.pack());
    let honest = compute_product_skip_poly(&fp.by_ref(), &wp.by_ref(), k, false);
    let n_coeffs = 2 * ((1usize << k) - 1) + 1;
    assert_eq!(honest.coeffs.len(), n_coeffs);

    for i in 0..n_coeffs {
        let mut cheat = honest.coeffs.clone();
        cheat[i] += EF::ONE;
        let mut ps = prover_state();
        ps.add_extension_scalars(&cheat);
        // No grinding (pow_bits = 0); nothing further is needed — the verifier
        // must fail the window check before consuming more of the transcript.
        let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), get_poseidon16().clone(), Default::default()).unwrap();
        vs.duplex();
        let mut claimed = sum;
        let res = verify_product_sumcheck_with_skip(&mut vs, &mut claimed, k, n_rounds, 0);
        assert!(
            matches!(res, Err(ProofError::InvalidProof)),
            "tampered skip coeff {i} must fail the window identity"
        );
    }
}

/// A forged grinding witness is rejected by check_pow_grinding.
#[test]
fn test_grinding_witness_corruption_rejected() {
    let (n, k, n_rounds, pow_bits) = (12usize, 4usize, 6usize, 16usize);
    let (f, w) = build_inputs(n);
    let sum = naive_cube_sum(&f, &w);
    let f_ref = MleRef::<EF>::Base(&f);
    let w_ref = MleRef::<EF>::Extension(&w);
    let (fp, wp) = (f_ref.pack(), w_ref.pack());
    let honest = compute_product_skip_poly(&fp.by_ref(), &wp.by_ref(), k, false);

    let mut ps = prover_state();
    ps.add_extension_scalars(&honest.coeffs);
    // Forge the witness: write a raw scalar where pow_grinding's output belongs.
    ps.add_base_scalars(&[F::ZERO]);
    let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), get_poseidon16().clone(), Default::default()).unwrap();
    vs.duplex();
    let mut claimed = sum;
    let res = verify_product_sumcheck_with_skip(&mut vs, &mut claimed, k, n_rounds, pow_bits);
    assert!(
        matches!(res, Err(ProofError::InvalidGrindingWitness)),
        "forged grinding witness must be rejected, got {res:?}"
    );
}

/// Linear-round wire tampering is absorbed by c0-elision into a *different*
/// (point, sum) binding — the in-protocol rounds still chain, and rejection
/// happens at the caller's final oracle check. We assert the binding diverges
/// from the honest run on an identical transcript prefix.
#[test]
fn test_linear_round_corruption_changes_binding() {
    let (n, k, n_rounds) = (12usize, 4usize, 5usize); // skip + ONE linear round
    let (f, w) = build_inputs(n);
    let sum = naive_cube_sum(&f, &w);
    let f_ref = MleRef::<EF>::Base(&f);
    let w_ref = MleRef::<EF>::Extension(&w);
    let (fp, wp) = (f_ref.pack(), w_ref.pack());

    let (honest_point, honest_sum) = roundtrip(n, k, n_rounds, 0, true);

    // Cheating prover: honest skip round, then a perturbed linear round.
    let mut ps = prover_state();
    let (point_skip, sum_after_skip, fa, fb) =
        run_product_sumcheck_with_skip(&fp.by_ref(), &wp.by_ref(), &mut ps, sum, k, k, 0);
    assert_eq!(point_skip.len(), 1);
    let (fav, fbv) = (unpack_owned(&fa), unpack_owned(&fb));
    let mut round_poly = compute_product_sumcheck_polynomial(&fav, &fbv, sum_after_skip, |e| vec![e]);
    round_poly.coeffs[1] += EF::ONE; // tamper the transmitted wire (c1)
    ps.add_sumcheck_polynomial(&round_poly.coeffs, None);
    let _r: EF = ps.sample();

    let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), get_poseidon16().clone(), Default::default()).unwrap();
    vs.duplex();
    let mut claimed = sum;
    let res = verify_product_sumcheck_with_skip(&mut vs, &mut claimed, k, n_rounds, 0);
    match res {
        Err(_) => {} // also acceptable: structural rejection
        Ok(point_v) => {
            assert!(
                point_v.0 != honest_point.0 || claimed != honest_sum,
                "tampered linear round must change the (point, sum) binding"
            );
        }
    }
}
