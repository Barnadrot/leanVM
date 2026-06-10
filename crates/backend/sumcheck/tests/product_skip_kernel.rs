//! Tests + kill-gate bench for the WHIR product-skip kernel (pw13-3 V1).
//!
//! The kill-gate bench (`whir_skip_timing`, #[ignore]) compares the skip round
//! against the legacy rounds 0..3 it replaces, on the production shape
//! (n = 26, f BasePacked × w ExtensionPacked). PASS iff
//! `skip_total ≤ 0.70 × legacy_rounds_0_3`.

use std::time::Instant;

use field::*;
use koala_bear::{KoalaBear, QuinticExtensionFieldKB};
use poly::*;
use sumcheck::{
    UNIVARIATE_SKIP_K, compute_product_skip_poly, compute_product_sumcheck_polynomial,
    fold_and_compute_product_sumcheck_polynomial, fold_product_skip, lagrange_weights_at, window_power_sums,
};

type F = KoalaBear;
type EF = QuinticExtensionFieldKB;

/// Deterministic scattered field elements (assertions below are polynomial
/// identities — any distinct values exercise them).
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

/// Σ_{x ∈ cube} f̂(x)·ŵ(x) — the eval arrays ARE the cube values.
fn naive_cube_sum(f: &[F], w: &[EF]) -> EF {
    f.iter().zip(w).map(|(&fv, &wv)| wv * fv).fold(EF::ZERO, |a, b| a + b)
}

#[test]
fn test_window_sum_identity_and_fd_equals_lagrange() {
    for k in [3usize, 4] {
        let n = k + 6;
        let (f, w) = build_inputs(n);
        let total = naive_cube_sum(&f, &w);

        // Unpacked (Base × Extension).
        let f_ref = MleRef::<EF>::Base(&f);
        let w_ref = MleRef::<EF>::Extension(&w);
        let poly_fd = compute_product_skip_poly(&f_ref, &w_ref, k, false);
        let poly_lag = compute_product_skip_poly(&f_ref, &w_ref, k, true);
        assert_eq!(poly_fd.coeffs, poly_lag.coeffs, "FD vs Lagrange k={k}");
        assert!(poly_fd.coeffs.len() <= 2 * ((1 << k) - 1) + 1);

        // Window-sum identity: Σ_{j<2^k} v'(j) == Σ_cube f·w …
        let window_sum = (0..1usize << k)
            .map(|j| poly_fd.evaluate(EF::from_usize(j)))
            .fold(EF::ZERO, |a, b| a + b);
        assert_eq!(window_sum, total, "window sum k={k}");
        // … equivalently as the verifier computes it: dot(coeffs, S).
        let sums = window_power_sums::<F>(k, poly_fd.coeffs.len());
        let dot = poly_fd
            .coeffs
            .iter()
            .zip(&sums)
            .map(|(&c, &s)| c * s)
            .fold(EF::ZERO, |a, b| a + b);
        assert_eq!(dot, total, "power-sum dot k={k}");

        // Packed (BasePacked × ExtensionPacked) gives identical coefficients.
        let f_packed = f_ref.pack();
        let w_packed = w_ref.pack();
        let poly_packed = compute_product_skip_poly(&f_packed.by_ref(), &w_packed.by_ref(), k, false);
        assert_eq!(poly_packed.coeffs, poly_fd.coeffs, "packed vs scalar k={k}");
        let poly_packed_lag = compute_product_skip_poly(&f_packed.by_ref(), &w_packed.by_ref(), k, true);
        assert_eq!(poly_packed_lag.coeffs, poly_fd.coeffs, "packed lagrange k={k}");

        // Extension × Extension (both EF) variants agree too.
        let f_ef: Vec<EF> = f.iter().map(|&v| EF::from(v)).collect();
        let fe_ref = MleRef::<EF>::Extension(&f_ef);
        let poly_ef = compute_product_skip_poly(&fe_ref, &w_ref, k, false);
        assert_eq!(poly_ef.coeffs, poly_fd.coeffs, "EF-evals vs base k={k}");
        let fe_packed = fe_ref.pack();
        let poly_ef_packed = compute_product_skip_poly(&fe_packed.by_ref(), &w_packed.by_ref(), k, false);
        assert_eq!(poly_ef_packed.coeffs, poly_fd.coeffs, "EF packed k={k}");
    }
}

#[test]
fn test_fold_delta_property_and_mle_consistency() {
    for k in [3usize, 4] {
        let n = k + 6;
        let (f, w) = build_inputs(n);
        let bp = (1usize << n) >> k;
        let f_ref = MleRef::<EF>::Base(&f);
        let w_ref = MleRef::<EF>::Extension(&w);

        // δ-property: r0 = node y folds to exactly block y.
        for y in [0usize, 1, (1 << k) - 1] {
            let lw = lagrange_weights_at::<F, EF>(k, EF::from_usize(y));
            let (gf, gw) = fold_product_skip(&f_ref, &w_ref, &lw);
            let gf = gf.by_ref().as_extension().unwrap().to_vec();
            let gw = gw.by_ref().as_extension().unwrap().to_vec();
            for p in 0..bp {
                assert_eq!(gf[p], EF::from(f[y * bp + p]), "δ f k={k} y={y} p={p}");
                assert_eq!(gw[p], w[y * bp + p], "δ w k={k} y={y} p={p}");
            }
        }

        // Random r0: folded MLE evaluation == Σ_j L_j(r0)·(block j evaluation),
        // and packed fold agrees with the scalar fold.
        let r0 = test_scalar_ef(123 + k);
        let lw = lagrange_weights_at::<F, EF>(k, r0);
        let (gf, gw) = fold_product_skip(&f_ref, &w_ref, &lw);
        let pt = MultilinearPoint((0..n - k).map(|i| test_scalar_ef(500 + i)).collect::<Vec<_>>());
        let gf_eval = gf.by_ref().evaluate(&pt);
        let gw_eval = gw.by_ref().evaluate(&pt);
        let mut expected_f = EF::ZERO;
        let mut expected_w = EF::ZERO;
        for (j, &l) in lw.iter().enumerate() {
            let fb = MleRef::<EF>::Base(&f[j * bp..(j + 1) * bp]);
            let wb = MleRef::<EF>::Extension(&w[j * bp..(j + 1) * bp]);
            expected_f += l * fb.evaluate(&pt);
            expected_w += l * wb.evaluate(&pt);
        }
        assert_eq!(gf_eval, expected_f, "fold MLE f k={k}");
        assert_eq!(gw_eval, expected_w, "fold MLE w k={k}");

        let f_packed = f_ref.pack();
        let w_packed = w_ref.pack();
        let (gf_p, gw_p) = fold_product_skip(&f_packed.by_ref(), &w_packed.by_ref(), &lw);
        assert_eq!(gf_p.by_ref().evaluate(&pt), expected_f, "packed fold f k={k}");
        assert_eq!(gw_p.by_ref().evaluate(&pt), expected_w, "packed fold w k={k}");
    }
}

#[test]
fn test_skip_then_legacy_rounds_consistency() {
    // After the skip round at r0, continuing with the LEGACY machinery on the
    // folded operands must satisfy the chained sumcheck identity:
    // h_next(0) + h_next(1) == v'(r0).
    let k = UNIVARIATE_SKIP_K;
    let n = k + 6;
    let (f, w) = build_inputs(n);
    let f_ref = MleRef::<EF>::Base(&f);
    let w_ref = MleRef::<EF>::Extension(&w);

    let poly = compute_product_skip_poly(&f_ref, &w_ref, k, false);
    let r0 = test_scalar_ef(901);
    let target = poly.evaluate(r0);
    let lw = lagrange_weights_at::<F, EF>(k, r0);
    let (gf, gw) = fold_product_skip(&f_ref, &w_ref, &lw);

    // Direct check: dot(folded f, folded w) == v'(r0).
    let gf_v = gf.by_ref().as_extension().unwrap().to_vec();
    let gw_v = gw.by_ref().as_extension().unwrap().to_vec();
    let folded_dot = gf_v.iter().zip(&gw_v).map(|(&a, &b)| a * b).fold(EF::ZERO, |x, y| x + y);
    assert_eq!(folded_dot, target, "v'(r0) equals the folded cube sum");

    // And the next legacy round polynomial sums to it.
    let next = compute_product_sumcheck_polynomial(&gf_v, &gw_v, target, |e| vec![e]);
    assert_eq!(next.evaluate(EF::ZERO) + next.evaluate(EF::ONE), target);
}

/// KILL-GATE bench (pw13-3 V1). Run with:
///   cargo test -p sumcheck --release --test product_skip_kernel -- --ignored --nocapture
#[test]
#[ignore]
fn whir_skip_timing() {
    let k = UNIVARIATE_SKIP_K;
    let n = 26usize;
    eprintln!("building 2^{n} inputs (~1.6 GB)…");
    // Cheap deterministic fill (LCG); values are irrelevant to timing.
    let f: Vec<F> = {
        let mut s = 0x12345678u64;
        (0..1usize << n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                F::from_usize((s >> 33) as usize)
            })
            .collect()
    };
    let w: Vec<EF> = {
        let mut s = 0x9abcdef0u64;
        (0..1usize << n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let base = (s >> 33) as usize;
                EF::from_basis_coefficients_fn(|j| F::from_usize(base ^ (j * 0x55555)))
            })
            .collect()
    };
    let f_ref = MleRef::<EF>::Base(&f);
    let w_ref = MleRef::<EF>::Extension(&w);
    let f_packed = f_ref.pack();
    let w_packed = w_ref.pack();
    let fp = f_packed.by_ref();
    let wp = w_packed.by_ref();
    let (fp_slice, wp_slice) = (fp.as_packed_base().unwrap(), wp.as_extension_packed().unwrap());

    // True sum (outside timed regions; needed so c1 derivations are honest).
    let sum: EF = {
        let acc = parallel::map_reduce(
            fp_slice.len(),
            || EFPacking::<EF>::ZERO,
            |i| wp_slice[i] * fp_slice[i],
            |a, b| a + b,
        );
        unpack_extension::<EF, Vec<EF>>(&[acc]).into_iter().sum::<EF>()
    };

    let decompose = |e: EFPacking<EF>| unpack_extension::<EF, Vec<EF>>(&[e]);
    let r_fixed: Vec<EF> = (0..4).map(|i| test_scalar_ef(3000 + i)).collect();

    let reps = 3usize;
    let mut t_skip_fd = f64::MAX;
    let mut t_skip_lag = f64::MAX;
    let mut t_legacy03 = f64::MAX;

    for rep in 0..reps {
        // --- skip (FD) ---
        let t0 = Instant::now();
        let poly = compute_product_skip_poly(&fp, &wp, k, false);
        let lw = lagrange_weights_at::<F, EF>(k, r_fixed[0]);
        let (gf, gw) = fold_product_skip(&fp, &wp, &lw);
        let dt = t0.elapsed().as_secs_f64();
        t_skip_fd = t_skip_fd.min(dt);
        std::hint::black_box((&poly, &gf, &gw));

        // --- skip (Lagrange reference) ---
        let t0 = Instant::now();
        let poly = compute_product_skip_poly(&fp, &wp, k, true);
        let lw = lagrange_weights_at::<F, EF>(k, r_fixed[0]);
        let (gf, gw) = fold_product_skip(&fp, &wp, &lw);
        let dt = t0.elapsed().as_secs_f64();
        t_skip_lag = t_skip_lag.min(dt);
        std::hint::black_box((&poly, &gf, &gw));

        // --- legacy rounds 0..3 (replicates run_product_sumcheck's first 4
        // rounds: round-0 poly, then 3 fused fold+compute rounds; transcript
        // ops excluded on BOTH sides) ---
        let t0 = Instant::now();
        let p1 = compute_product_sumcheck_polynomial(fp_slice, wp_slice, sum, decompose);
        let s1 = p1.evaluate(r_fixed[0]);
        let (p2, folded1) = fold_and_compute_product_sumcheck_polynomial(fp_slice, wp_slice, r_fixed[0], s1, decompose);
        let s2 = p2.evaluate(r_fixed[1]);
        let (p3, folded2) =
            fold_and_compute_product_sumcheck_polynomial(&folded1[0], &folded1[1], r_fixed[1], s2, decompose);
        let s3 = p3.evaluate(r_fixed[2]);
        let (p4, folded3) =
            fold_and_compute_product_sumcheck_polynomial(&folded2[0], &folded2[1], r_fixed[2], s3, decompose);
        let dt = t0.elapsed().as_secs_f64();
        t_legacy03 = t_legacy03.min(dt);
        std::hint::black_box((&p4, &folded3));
        eprintln!(
            "rep {rep}: skip_fd {:.1} ms | skip_lagrange {:.1} ms | legacy rounds 0..3 {:.1} ms",
            t_skip_fd * 1e3,
            t_skip_lag * 1e3,
            t_legacy03 * 1e3
        );
    }

    // Context: the full 7-round legacy arithmetic (rounds 0..3 again + 3 more
    // fused fold+compute rounds on the shrunken operands).
    let t0 = Instant::now();
    let p1 = compute_product_sumcheck_polynomial(fp_slice, wp_slice, sum, decompose);
    let mut s = p1.evaluate(r_fixed[0]);
    let (_p, mut folded) = fold_and_compute_product_sumcheck_polynomial(fp_slice, wp_slice, r_fixed[0], s, decompose);
    for i in 1..6 {
        let r = test_scalar_ef(4000 + i);
        s = EF::ZERO + s; // keep the chain honest without a transcript
        let (p_next, folded_next) =
            fold_and_compute_product_sumcheck_polynomial(&folded[0], &folded[1], r, s, decompose);
        s = p_next.evaluate(test_scalar_ef(4100 + i));
        folded = folded_next;
    }
    let t_legacy_full = t0.elapsed().as_secs_f64();
    std::hint::black_box(&folded);

    let ratio = t_skip_fd / t_legacy03;
    eprintln!("== whir_skip_timing (n={n}, K={k}, min of {reps}) ==");
    eprintln!("skip (FD):        {:8.1} ms", t_skip_fd * 1e3);
    eprintln!("skip (Lagrange):  {:8.1} ms", t_skip_lag * 1e3);
    eprintln!("legacy rounds 0-3:{:8.1} ms", t_legacy03 * 1e3);
    eprintln!("legacy full 7 rds:{:8.1} ms (context)", t_legacy_full * 1e3);
    eprintln!("ratio skip/legacy03 = {ratio:.3} (gate: ≤ 0.70)");
    if ratio <= 0.70 {
        eprintln!("VERDICT: PASS");
    } else {
        eprintln!("VERDICT: FAIL — kill condition (a)");
    }
    assert!(ratio <= 0.70, "kill-gate: skip/legacy = {ratio:.3} > 0.70");
}
