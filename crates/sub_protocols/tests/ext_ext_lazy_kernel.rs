//! T5 (plan_spec §3.2, Target B) equality oracle: the deferred-Karatsuba
//! ext-x-ext kernels must be value-identical to the eager generic paths they
//! replace, on the real Goldilocks cubic-extension types the prover dispatches
//! (run_product_sumcheck routes `EF::DIMENSION == 3` ExtensionPacked arms to the
//! lazy kernels; everything else keeps the generic path).

use backend::{
    BasedVectorSpace, CubicExtensionFieldGL, EFPacking, Goldilocks, PFPacking, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing, compute_product_sumcheck_polynomial, compute_product_sumcheck_polynomial_ext_ext_packed,
    fold_and_compute_product_sumcheck_polynomial, fold_and_compute_product_sumcheck_polynomial_ext_ext_packed,
};

type EF = CubicExtensionFieldGL;
#[allow(clippy::upper_case_acronyms)]
type EFP = EFPacking<EF>;
#[allow(clippy::upper_case_acronyms)]
type PFP = PFPacking<EF>;

/// Deterministic xorshift64* (same approach as the sumcheck crate's T4 tests;
/// no new deps).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn random_efp(rng: &mut Rng) -> EFP {
    // Coefficients over the packed base field: fill every lane with raw
    // (non-canonical-allowed) u64s, mixing in boundary values.
    EFP::from_basis_coefficients_fn(|_| {
        PFP::from_fn(|lane| {
            let r = rng.next_u64();
            backend::Goldilocks::new(match r % 11 {
                0 => 0,
                1 => 1,
                2 => 0xFFFF_FFFF_0000_0001, // p (non-canonical zero)
                3 => 0xFFFF_FFFF_0000_0000, // p - 1
                4 => u64::MAX,              // 2^64 - 1
                5 => 0xFFFF_FFFF,           // eps
                _ => r ^ (lane as u64),
            })
        })
    })
}

fn random_vec(rng: &mut Rng, n: usize) -> Vec<EFP> {
    (0..n).map(|_| random_efp(rng)).collect()
}

fn random_ef(rng: &mut Rng) -> EF {
    <EF as BasedVectorSpace<Goldilocks>>::from_basis_coefficients_fn(|_| Goldilocks::new(rng.next_u64()))
}

fn decompose(e: EFP) -> Vec<EF> {
    <EFP as PackedFieldExtension<Goldilocks, EF>>::to_ext_iter([e]).collect()
}

#[test]
fn compute_ext_ext_lazy_matches_generic() {
    // n = 1024 -> half = 512: single partial chunk (512 < EXT_EXT_CHUNK_SIZE).
    // n = 4096, 16384: multi-chunk.
    for (seed, log_n) in [
        (0x1234_5678_9abc_def1u64, 10),
        (0xdead_beef_cafe_f00du64, 12),
        (0x0bad_5eed_0bad_5eedu64, 14),
    ] {
        let mut rng = Rng(seed);
        let n = 1usize << log_n;
        let pol_0 = random_vec(&mut rng, n);
        let pol_1 = random_vec(&mut rng, n);
        let sum = random_ef(&mut rng);

        let lazy = compute_product_sumcheck_polynomial_ext_ext_packed::<3, Goldilocks, PFP, EFP, EF>(
            &pol_0, &pol_1, sum, decompose,
        );
        let eager = compute_product_sumcheck_polynomial(&pol_0, &pol_1, sum, decompose);
        assert_eq!(lazy.coeffs, eager.coeffs, "compute kernel mismatch at n=2^{log_n}");
    }
}

#[test]
fn compute_ext_ext_lazy_serial_delegation() {
    // n = 256 < PARALLEL_THRESHOLD: the lazy entry point must delegate to the
    // generic path (trivially equal, but pins the branch).
    let mut rng = Rng(7);
    let n = 256;
    let pol_0 = random_vec(&mut rng, n);
    let pol_1 = random_vec(&mut rng, n);
    let sum = random_ef(&mut rng);
    let lazy = compute_product_sumcheck_polynomial_ext_ext_packed::<3, Goldilocks, PFP, EFP, EF>(
        &pol_0, &pol_1, sum, decompose,
    );
    let eager = compute_product_sumcheck_polynomial(&pol_0, &pol_1, sum, decompose);
    assert_eq!(lazy.coeffs, eager.coeffs);
}

#[test]
fn compute_ext_ext_lazy_boundary_patterns() {
    // Zero/one-heavy inputs: zero products inside lazy subs exercise the
    // ~t_hi = 2^64-1 NOT-negation edge that mandates the separate K counter.
    let n = 2048;
    let pol_0: Vec<EFP> = (0..n)
        .map(|i| match i % 4 {
            0 => EFP::ZERO,
            1 => EFP::ONE,
            2 => EFP::from_basis_coefficients_fn(|j| {
                PFP::from_fn(|_| Goldilocks::new(if j == 1 { u64::MAX } else { 0 }))
            }),
            _ => {
                let mut rng = Rng(i as u64 + 99);
                random_efp(&mut rng)
            }
        })
        .collect();
    let pol_1: Vec<EFP> = (0..n)
        .map(|i| match i % 3 {
            0 => EFP::ZERO,
            _ => {
                let mut rng = Rng(i as u64 + 7777);
                random_efp(&mut rng)
            }
        })
        .collect();
    let sum = EF::ONE;
    let lazy = compute_product_sumcheck_polynomial_ext_ext_packed::<3, Goldilocks, PFP, EFP, EF>(
        &pol_0, &pol_1, sum, decompose,
    );
    let eager = compute_product_sumcheck_polynomial(&pol_0, &pol_1, sum, decompose);
    assert_eq!(lazy.coeffs, eager.coeffs);
}

#[test]
fn fold_ext_ext_lazy_matches_generic() {
    for (seed, log_n) in [(0xaaaa_bbbb_cccc_dddd_u64, 11), (0x1111_2222_3333_4444u64, 13)] {
        let mut rng = Rng(seed);
        let n = 1usize << log_n;
        let pol_0 = random_vec(&mut rng, n);
        let pol_1 = random_vec(&mut rng, n);
        let sum = random_ef(&mut rng);
        let ff = random_ef(&mut rng);

        let (lazy_poly, lazy_folded) =
            fold_and_compute_product_sumcheck_polynomial_ext_ext_packed::<3, Goldilocks, PFP, EFP, EF>(
                &pol_0, &pol_1, ff, sum, decompose,
            );
        let (eager_poly, eager_folded) =
            fold_and_compute_product_sumcheck_polynomial(&pol_0, &pol_1, ff, sum, decompose);

        assert_eq!(
            lazy_poly.coeffs, eager_poly.coeffs,
            "fold round-poly mismatch at n=2^{log_n}"
        );
        assert_eq!(lazy_folded.len(), eager_folded.len());
        for (a, b) in lazy_folded.iter().zip(eager_folded.iter()) {
            assert_eq!(a.as_ref(), b.as_ref(), "folded vector mismatch at n=2^{log_n}");
        }
    }
}

#[test]
fn fold_ext_ext_lazy_serial_delegation() {
    let mut rng = Rng(424242);
    let n = 256;
    let pol_0 = random_vec(&mut rng, n);
    let pol_1 = random_vec(&mut rng, n);
    let sum = random_ef(&mut rng);
    let ff = random_ef(&mut rng);
    let (lazy_poly, lazy_folded) =
        fold_and_compute_product_sumcheck_polynomial_ext_ext_packed::<3, Goldilocks, PFP, EFP, EF>(
            &pol_0, &pol_1, ff, sum, decompose,
        );
    let (eager_poly, eager_folded) = fold_and_compute_product_sumcheck_polynomial(&pol_0, &pol_1, ff, sum, decompose);
    assert_eq!(lazy_poly.coeffs, eager_poly.coeffs);
    for (a, b) in lazy_folded.iter().zip(eager_folded.iter()) {
        assert_eq!(a.as_ref(), b.as_ref());
    }
}
