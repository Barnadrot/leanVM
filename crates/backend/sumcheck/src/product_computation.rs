use fiat_shamir::*;
use field::*;
use poly::*;
use tracing::instrument;
use zk_alloc::ArenaVec;

use crate::{SumcheckComputation, sumcheck_prove_many_rounds};

#[derive(Debug)]
pub struct ProductComputation;

impl<EF: ExtensionField<PF<EF>>> SumcheckComputation<EF> for ProductComputation {
    type ExtraData = Vec<EF>;

    fn degree(&self) -> usize {
        2
    }
    #[inline(always)]
    fn eval_base(&self, _point: &[PF<EF>], _: &Self::ExtraData) -> EF {
        unreachable!()
    }
    #[inline(always)]
    fn eval_extension(&self, point: &[EF], _: &Self::ExtraData) -> EF {
        point[0] * point[1]
    }
    #[inline(always)]
    fn eval_packed_base(&self, point: &[PFPacking<EF>], _: &Self::ExtraData) -> EFPacking<EF> {
        EFPacking::<EF>::from(point[0] * point[1])
    }
    #[inline(always)]
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &Self::ExtraData) -> EFPacking<EF> {
        point[0] * point[1]
    }
}

#[instrument(skip_all)]
pub fn run_product_sumcheck<EF: ExtensionField<PF<EF>>>(
    pol_a: &MleRef<'_, EF>, // evals
    pol_b: &MleRef<'_, EF>, // weights
    prover_state: &mut impl FSProver<EF>,
    mut sum: EF,
    n_rounds: usize,
    pow_bits: usize,
) -> (MultilinearPoint<EF>, EF, MleOwned<EF>, MleOwned<EF>) {
    assert!(n_rounds >= 1);
    if let (MleRef::BasePacked(evals), MleRef::ExtensionPacked(weights)) = (pol_a, pol_b) {
        if EF::DIMENSION == 3 {
            return run_product_sumcheck_base_eager::<3, EF>(evals, weights, prover_state, sum, n_rounds, pow_bits);
        }
        unimplemented!()
    }
    let first_sumcheck_poly = match (pol_a, pol_b) {
        (MleRef::ExtensionPacked(evals), MleRef::ExtensionPacked(weights)) => {
            compute_product_sumcheck_polynomial(evals, weights, sum, |e| EFPacking::<EF>::to_ext_iter([e]).collect())
        }
        (MleRef::Base(evals), MleRef::Extension(weights)) => {
            compute_product_sumcheck_polynomial(evals, weights, sum, |e| vec![e])
        }
        (MleRef::Extension(evals), MleRef::Extension(weights)) => {
            compute_product_sumcheck_polynomial(evals, weights, sum, |e| vec![e])
        }
        _ => unimplemented!(),
    };

    prover_state.add_sumcheck_polynomial(&first_sumcheck_poly.coeffs, None);
    prover_state.pow_grinding(pow_bits);
    let r1: EF = prover_state.sample();
    sum = first_sumcheck_poly.evaluate(r1);

    if n_rounds == 1 {
        return (MultilinearPoint(vec![r1]), sum, pol_a.fold(r1), pol_b.fold(r1));
    }

    let (second_sumcheck_poly, folded) = match (pol_a, pol_b) {
        (MleRef::ExtensionPacked(evals), MleRef::ExtensionPacked(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                });
            (second_sumcheck_poly, MleGroupOwned::ExtensionPacked(folded))
        }
        (MleRef::Base(evals), MleRef::Extension(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| vec![e]);
            (second_sumcheck_poly, MleGroupOwned::Extension(folded))
        }
        (MleRef::Extension(evals), MleRef::Extension(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| vec![e]);
            (second_sumcheck_poly, MleGroupOwned::Extension(folded))
        }
        _ => unimplemented!(),
    };

    prover_state.add_sumcheck_polynomial(&second_sumcheck_poly.coeffs, None);
    prover_state.pow_grinding(pow_bits);
    let r2: EF = prover_state.sample();
    sum = second_sumcheck_poly.evaluate(r2);

    let (mut challenges, folds, sum) = sumcheck_prove_many_rounds(
        folded,
        Some(r2),
        &ProductComputation {},
        &vec![],
        None,
        prover_state,
        sum,
        None,
        n_rounds - 2,
        false,
        pow_bits,
    );

    challenges.splice(0..0, [r1, r2]);
    let [pol_a, pol_b] = folds.split().try_into().unwrap();
    (challenges, sum, pol_a, pol_b)
}

/// Eager path for the (BasePacked, ExtensionPacked) arm, extracted verbatim from
/// `run_product_sumcheck`. Kept as the equality oracle for the lazy path and as
/// the fallback for small instances. Generic over `DIM` so the full path is
/// exercisable by the KoalaBear (DIM = 5) test harness.
pub fn run_product_sumcheck_base_eager<const DIM: usize, EF: ExtensionField<PF<EF>>>(
    evals: &[PFPacking<EF>],
    weights: &[EFPacking<EF>],
    prover_state: &mut impl FSProver<EF>,
    mut sum: EF,
    n_rounds: usize,
    pow_bits: usize,
) -> (MultilinearPoint<EF>, EF, MleOwned<EF>, MleOwned<EF>) {
    assert!(n_rounds >= 1);
    let first_sumcheck_poly =
        compute_product_sumcheck_polynomial_base_ext_packed::<DIM, _, _, _, EF>(evals, weights, sum);

    prover_state.add_sumcheck_polynomial(&first_sumcheck_poly.coeffs, None);
    prover_state.pow_grinding(pow_bits);
    let r1: EF = prover_state.sample();
    sum = first_sumcheck_poly.evaluate(r1);

    if n_rounds == 1 {
        return (
            MultilinearPoint(vec![r1]),
            sum,
            MleRef::<EF>::BasePacked(evals).fold(r1),
            MleRef::<EF>::ExtensionPacked(weights).fold(r1),
        );
    }

    let (second_sumcheck_poly, folded) = {
        let (second_sumcheck_poly, folded) =
            fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| {
                EFPacking::<EF>::to_ext_iter([e]).collect()
            });
        (second_sumcheck_poly, MleGroupOwned::ExtensionPacked(folded))
    };

    prover_state.add_sumcheck_polynomial(&second_sumcheck_poly.coeffs, None);
    prover_state.pow_grinding(pow_bits);
    let r2: EF = prover_state.sample();
    sum = second_sumcheck_poly.evaluate(r2);

    let (mut challenges, folds, sum) = sumcheck_prove_many_rounds(
        folded,
        Some(r2),
        &ProductComputation {},
        &vec![],
        None,
        prover_state,
        sum,
        None,
        n_rounds - 2,
        false,
        pow_bits,
    );

    challenges.splice(0..0, [r1, r2]);
    let [pol_a, pol_b] = folds.split().try_into().unwrap();
    (challenges, sum, pol_a, pol_b)
}

pub fn compute_product_sumcheck_polynomial<
    F: PrimeCharacteristicRing + Copy + Send + Sync,
    EF: Field,
    EFPacking: Algebra<F> + Copy + Send + Sync,
>(
    pol_0: &[F],         // evals
    pol_1: &[EFPacking], // weights
    sum: EF,
    decompose: impl Fn(EFPacking) -> Vec<EF>,
) -> DensePolynomial<EF> {
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());

    let num_elements = n;

    let (c0_packed, c2_packed) = if num_elements < PARALLEL_THRESHOLD {
        pol_0[..n / 2]
            .iter()
            .zip(pol_0[n / 2..].iter())
            .zip(pol_1[..n / 2].iter().zip(pol_1[n / 2..].iter()))
            .map(sumcheck_quadratic)
            .fold((EFPacking::ZERO, EFPacking::ZERO), |(a0, a2), (b0, b2)| {
                (a0 + b0, a2 + b2)
            })
    } else {
        let half = n / 2;
        parallel::map_reduce(
            half,
            || (EFPacking::ZERO, EFPacking::ZERO),
            |i| sumcheck_quadratic(((&pol_0[i], &pol_0[half + i]), (&pol_1[i], &pol_1[half + i]))),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        )
    };

    let c0 = decompose(c0_packed).into_iter().sum::<EF>();
    let c2 = decompose(c2_packed).into_iter().sum::<EF>();
    let c1 = sum - c0.double() - c2;

    DensePolynomial::new(vec![c0, c1, c2])
}

// Generic over PrimeField64 (Goldilocks and Goldilocks both qualify). The Goldilocks-specific
// delayed u128/i128 accumulation path is retained as a specialization candidate for a future
// pass — see `crates/backend/goldilocks/README.md`.
pub fn compute_product_sumcheck_polynomial_base_ext_packed<
    const DIM: usize,
    F: PrimeField64,
    PF: PackedField<Scalar = F>,
    EFP: BasedVectorSpace<PF> + Copy + Send + Sync,
    EF: Field + BasedVectorSpace<F>,
>(
    pol_0: &[PF],
    pol_1: &[EFP],
    sum: EF,
) -> DensePolynomial<EF> {
    assert_eq!(DIM, EF::DIMENSION);
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());
    let half = n / 2;

    let chunk_size = 1024;

    let n_chunks = half.div_ceil(chunk_size);
    // Deferred-reduction packed accumulation (lazy-accumulator protocol, see
    // PrimeCharacteristicRing::unreduced_mul): each chunk keeps 2*DIM unreduced
    // accumulators and reduces once at chunk exit. All terms are raw products
    // added positively, so n_sub = 0. Chunk length 1024 <= 2^20 honors the
    // protocol bound (1 term per accumulator per iteration).
    let (c0_acc, c2_acc) = parallel::map_reduce(
        n_chunks,
        || ([PF::ZERO; DIM], [PF::ZERO; DIM]),
        |chunk| {
            let start = chunk * chunk_size;
            let end = (start + chunk_size).min(half);
            let b_lo = &pol_0[start..end];
            let b_hi = &pol_0[half + start..half + end];
            let e_lo = &pol_1[start..end];
            let e_hi = &pol_1[half + start..half + end];
            // Two passes over the (L2-resident) chunk, halving live accumulator
            // registers per loop (plan_spec §3.1 register-pressure fallback:
            // 6 accumulators x 4 zmm = 24 live exceeded the budget and spilled
            // in-loop; 3 x 4 = 12 per pass fits).
            let mut c0_lazy: [[PF; 4]; DIM] = core::array::from_fn(|_| PF::lazy_acc_zero());
            for i in 0..b_lo.len() {
                let x0 = b_lo[i];
                let y0_coords = e_lo[i].as_basis_coefficients_slice();
                for j in 0..DIM {
                    c0_lazy[j] = PF::lazy_acc_add(c0_lazy[j], PF::unreduced_mul(y0_coords[j], x0));
                }
            }
            let mut c2_lazy: [[PF; 4]; DIM] = core::array::from_fn(|_| PF::lazy_acc_zero());
            for i in 0..b_lo.len() {
                let dx = b_hi[i] - b_lo[i];
                let y0_coords = e_lo[i].as_basis_coefficients_slice();
                let y1_coords = e_hi[i].as_basis_coefficients_slice();
                for j in 0..DIM {
                    let dy = y1_coords[j] - y0_coords[j];
                    c2_lazy[j] = PF::lazy_acc_add(c2_lazy[j], PF::unreduced_mul(dy, dx));
                }
            }
            (
                core::array::from_fn(|j| PF::lazy_acc_finish(c0_lazy[j], 0)),
                core::array::from_fn(|j| PF::lazy_acc_finish(c2_lazy[j], 0)),
            )
        },
        |(mut a0, mut a2): ([PF; DIM], [PF; DIM]), (b0, b2): ([PF; DIM], [PF; DIM])| {
            for j in 0..DIM {
                a0[j] += b0[j];
                a2[j] += b2[j];
            }
            (a0, a2)
        },
    );

    // Horizontal lane sum once at the very end.
    let lane_sum = |p: PF| {
        let mut s = F::ZERO;
        for &v in p.as_slice() {
            s += v;
        }
        s
    };
    let c0 = EF::from_basis_coefficients_fn(|j| lane_sum(c0_acc[j]));
    let c2 = EF::from_basis_coefficients_fn(|j| lane_sum(c2_acc[j]));
    let c1 = sum - c0.double() - c2;

    DensePolynomial::new(vec![c0, c1, c2])
}

pub fn fold_and_compute_product_sumcheck_polynomial<
    F: PrimeCharacteristicRing + Copy + Send + Sync + 'static,
    EF: Field,
    EFPacking: Algebra<F> + From<EF> + Copy + Send + Sync + 'static,
>(
    pol_0: &[F],         // evals
    pol_1: &[EFPacking], // weights
    prev_folding_factor: EF,
    sum: EF,
    decompose: impl Fn(EFPacking) -> Vec<EF>,
) -> (DensePolynomial<EF>, Vec<ArenaVec<EFPacking>>) {
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());
    let prev_folding_factor_packed = EFPacking::from(prev_folding_factor);

    let mut pol_0_folded = unsafe { ArenaVec::<EFPacking>::uninitialized(n / 2) };
    let mut pol_1_folded = unsafe { ArenaVec::<EFPacking>::uninitialized(n / 2) };

    #[allow(clippy::type_complexity)]
    let process_element = |(p0_prev, p0_f): (((&F, &F), (&F, &F)), (&mut EFPacking, &mut EFPacking)),
                           (p1_prev, p1_f): (
        ((&EFPacking, &EFPacking), (&EFPacking, &EFPacking)),
        (&mut EFPacking, &mut EFPacking),
    )| {
        let diff_0 = *p0_prev.1.0 - *p0_prev.0.0;
        let diff_1 = *p0_prev.1.1 - *p0_prev.0.1;
        let x_0 = prev_folding_factor_packed * diff_0 + *p0_prev.0.0;
        let x_1 = prev_folding_factor_packed * diff_1 + *p0_prev.0.1;
        *p0_f.0 = x_0;
        *p0_f.1 = x_1;

        let y_0 = prev_folding_factor_packed * (*p1_prev.1.0 - *p1_prev.0.0) + *p1_prev.0.0;
        let y_1 = prev_folding_factor_packed * (*p1_prev.1.1 - *p1_prev.0.1) + *p1_prev.0.1;
        *p1_f.0 = y_0;
        *p1_f.1 = y_1;

        sumcheck_quadratic(((&x_0, &x_1), (&y_0, &y_1)))
    };

    let (c0_packed, c2_packed) = if n < PARALLEL_THRESHOLD {
        zip_fold_2(pol_0, &mut pol_0_folded)
            .zip(zip_fold_2(pol_1, &mut pol_1_folded))
            .map(|(p0, p1)| process_element(p0, p1))
            .fold((EFPacking::ZERO, EFPacking::ZERO), |(a0, a2), (b0, b2)| {
                (a0 + b0, a2 + b2)
            })
    } else {
        let quarter = n / 4;
        let p0f = parallel::SendPtr(pol_0_folded.as_mut_ptr());
        let p1f = parallel::SendPtr(pol_1_folded.as_mut_ptr());
        parallel::map_reduce(
            quarter,
            || (EFPacking::ZERO, EFPacking::ZERO),
            |i| {
                let diff_0 = pol_0[2 * quarter + i] - pol_0[i];
                let diff_1 = pol_0[3 * quarter + i] - pol_0[quarter + i];
                let x_0 = prev_folding_factor_packed * diff_0 + pol_0[i];
                let x_1 = prev_folding_factor_packed * diff_1 + pol_0[quarter + i];

                let y_0 = prev_folding_factor_packed * (pol_1[2 * quarter + i] - pol_1[i]) + pol_1[i];
                let y_1 =
                    prev_folding_factor_packed * (pol_1[3 * quarter + i] - pol_1[quarter + i]) + pol_1[quarter + i];

                unsafe {
                    *p0f.add(i) = x_0;
                    *p0f.add(quarter + i) = x_1;
                    *p1f.add(i) = y_0;
                    *p1f.add(quarter + i) = y_1;
                }

                sumcheck_quadratic(((&x_0, &x_1), (&y_0, &y_1)))
            },
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        )
    };

    let c0 = decompose(c0_packed).into_iter().sum::<EF>();
    let c2 = decompose(c2_packed).into_iter().sum::<EF>();
    let c1 = sum - c0.double() - c2;

    (DensePolynomial::new(vec![c0, c1, c2]), vec![pol_0_folded, pol_1_folded])
}

#[inline(always)]
pub fn sumcheck_quadratic<F, EF>(((&x_0, &x_1), (&y_0, &y_1)): ((&F, &F), (&EF, &EF))) -> (EF, EF)
where
    F: PrimeCharacteristicRing + Copy,
    EF: Algebra<F> + Copy,
{
    let constant = y_0 * x_0;
    let quadratic = (y_1 - y_0) * (x_1 - x_0);
    (constant, quadratic)
}

#[cfg(test)]
mod base_ext_packed_kernel_tests {
    use super::*;
    use koala_bear::{KoalaBear, PackedQuinticExtensionFieldKB, QuinticExtensionFieldKB};

    /// Pre-T4 kernel body, kept verbatim as the equality oracle (scalar per-lane
    /// accumulation with eager reduction). Proves the restructured kernel is
    /// value-preserving; the Goldilocks lazy primitives themselves are proven by
    /// the goldilocks crate's T2/T3 oracle tests (plan_spec §4.2 composition).
    fn reference_base_ext_packed<
        const DIM: usize,
        F: PrimeField64,
        PF: PackedField<Scalar = F>,
        EFP: BasedVectorSpace<PF> + Copy + Send + Sync,
        EF: Field + BasedVectorSpace<F>,
    >(
        pol_0: &[PF],
        pol_1: &[EFP],
        sum: EF,
    ) -> DensePolynomial<EF> {
        assert_eq!(DIM, EF::DIMENSION);
        let n = pol_0.len();
        assert_eq!(n, pol_1.len());
        assert!(n.is_power_of_two());
        let half = n / 2;
        let mut c0_acc = [F::ZERO; DIM];
        let mut c2_acc = [F::ZERO; DIM];
        for i in 0..half {
            let x0_lanes = pol_0[i].as_slice();
            let x1_lanes = pol_0[half + i].as_slice();
            let y0_coords = pol_1[i].as_basis_coefficients_slice();
            let y1_coords = pol_1[half + i].as_basis_coefficients_slice();
            for j in 0..DIM {
                let y0_j = y0_coords[j].as_slice();
                let y1_j = y1_coords[j].as_slice();
                for lane in 0..PF::WIDTH {
                    let x0 = x0_lanes[lane];
                    let x1 = x1_lanes[lane];
                    let y0 = y0_j[lane];
                    let y1 = y1_j[lane];
                    c0_acc[j] += y0 * x0;
                    c2_acc[j] += (y1 - y0) * (x1 - x0);
                }
            }
        }
        let c0 = EF::from_basis_coefficients_fn(|j| c0_acc[j]);
        let c2 = EF::from_basis_coefficients_fn(|j| c2_acc[j]);
        let c1 = sum - c0.double() - c2;
        DensePolynomial::new(vec![c0, c1, c2])
    }

    // Minimal deterministic PRNG (same xorshift precedent as the goldilocks
    // lazy_acc tests; no new deps).
    struct XorShift(u64);
    impl XorShift {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    type PFKb = <KoalaBear as Field>::Packing;
    const DIM_KB: usize = 5;

    fn random_inputs(seed: u64, log_n: usize) -> (Vec<PFKb>, Vec<PackedQuinticExtensionFieldKB>) {
        let mut rng = XorShift(seed | 1);
        let n = 1 << log_n;
        let base: Vec<PFKb> = (0..n)
            .map(|_| PFKb::from_fn(|_| KoalaBear::from_u64(rng.next_u64())))
            .collect();
        let ext: Vec<PackedQuinticExtensionFieldKB> = (0..n)
            .map(|_| {
                PackedQuinticExtensionFieldKB::from_basis_coefficients_fn(|_| {
                    PFKb::from_fn(|_| KoalaBear::from_u64(rng.next_u64()))
                })
            })
            .collect();
        (base, ext)
    }

    #[test]
    fn rewritten_kernel_matches_reference_randomized() {
        for (seed, log_n) in [(1u64, 1usize), (2, 2), (3, 4), (4, 7), (5, 11), (6, 12)] {
            let (base, ext) = random_inputs(seed, log_n);
            let sum = QuinticExtensionFieldKB::from_basis_coefficients_fn(|j| KoalaBear::from_u64(seed + j as u64));
            let new = compute_product_sumcheck_polynomial_base_ext_packed::<DIM_KB, _, _, _, QuinticExtensionFieldKB>(
                &base, &ext, sum,
            );
            let reference = reference_base_ext_packed::<DIM_KB, _, _, _, QuinticExtensionFieldKB>(&base, &ext, sum);
            assert_eq!(new.coeffs, reference.coeffs, "seed={seed} log_n={log_n}");
        }
    }

    #[test]
    fn rewritten_kernel_matches_reference_boundary() {
        // Zero blocks, all-ones, and max-representative patterns interleaved:
        // exercises zero products and the chunk-boundary path (non-multiple of
        // chunk_size handled by the (start + chunk_size).min(half) slicing).
        let n = 1 << 8;
        let mut rng = XorShift(0xDEAD_BEEF);
        let base: Vec<PFKb> = (0..n)
            .map(|i| match i % 4 {
                0 => PFKb::ZERO,
                1 => PFKb::ONE,
                2 => PFKb::from_fn(|_| KoalaBear::from_u64(u64::MAX)),
                _ => PFKb::from_fn(|_| KoalaBear::from_u64(rng.next_u64())),
            })
            .collect();
        let ext: Vec<PackedQuinticExtensionFieldKB> = (0..n)
            .map(|i| match i % 3 {
                0 => PackedQuinticExtensionFieldKB::ZERO,
                1 => PackedQuinticExtensionFieldKB::ONE,
                _ => PackedQuinticExtensionFieldKB::from_basis_coefficients_fn(|_| {
                    PFKb::from_fn(|_| KoalaBear::from_u64(rng.next_u64()))
                }),
            })
            .collect();
        let sum = QuinticExtensionFieldKB::ONE;
        let new = compute_product_sumcheck_polynomial_base_ext_packed::<DIM_KB, _, _, _, QuinticExtensionFieldKB>(
            &base, &ext, sum,
        );
        let reference = reference_base_ext_packed::<DIM_KB, _, _, _, QuinticExtensionFieldKB>(&base, &ext, sum);
        assert_eq!(new.coeffs, reference.coeffs);
    }
}

#[cfg(test)]
mod full_path_equality_tests {
    use super::*;
    use koala_bear::{KoalaBear, PackedQuinticExtensionFieldKB, QuinticExtensionFieldKB};

    type EFKb = QuinticExtensionFieldKB;
    type PFKb = <KoalaBear as Field>::Packing;
    const DIM_KB: usize = 5;

    struct XorShift(u64);
    impl XorShift {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Deterministic recording Fiat-Shamir prover: challenges come from a seeded
    /// xorshift stream (independent of absorbed data), and every coefficient
    /// vector passed to `add_sumcheck_polynomial` is captured. Two runs with the
    /// same seed and the same number of sample() calls see identical challenge
    /// sequences, so any divergence between the eager and lazy paths surfaces as
    /// a recorded-coefficient or returned-value mismatch.
    pub(super) struct RecordingFs {
        rng: XorShift,
        pub polys: Vec<Vec<EFKb>>,
        pub challenges: Vec<EFKb>,
    }

    impl RecordingFs {
        pub fn new(seed: u64) -> Self {
            Self {
                rng: XorShift(seed | 1),
                polys: vec![],
                challenges: vec![],
            }
        }
    }

    impl ChallengeSampler<EFKb> for RecordingFs {
        fn sample_vec(&mut self, len: usize) -> Vec<EFKb> {
            (0..len)
                .map(|_| {
                    let c = EFKb::from_basis_coefficients_fn(|_| KoaBearRand::draw(&mut self.rng));
                    self.challenges.push(c);
                    c
                })
                .collect()
        }
        fn sample_in_range(&mut self, _bits: usize, _n_samples: usize) -> Vec<usize> {
            unimplemented!("not used by run_product_sumcheck")
        }
    }

    /// Helper so the closure in sample_vec stays readable.
    struct KoaBearRand;
    impl KoaBearRand {
        fn draw(rng: &mut XorShift) -> KoalaBear {
            KoalaBear::from_u64(rng.next_u64())
        }
    }

    impl FSProver<EFKb> for RecordingFs {
        fn state(&self) -> String {
            format!("recording[{}]", self.polys.len())
        }
        fn add_base_scalars(&mut self, _scalars: &[KoalaBear]) {}
        fn observe_scalars(&mut self, _scalars: &[KoalaBear]) {}
        fn duplex(&mut self) {}
        fn pow_grinding(&mut self, _bits: usize) {}
        fn hint_merkle_paths_base(&mut self, _paths: Vec<MerklePath<KoalaBear, KoalaBear>>) {}
        fn add_sumcheck_polynomial(&mut self, coeffs: &[EFKb], _eq_alpha: Option<EFKb>) {
            self.polys.push(coeffs.to_vec());
        }
    }

    pub(super) fn random_full_inputs(seed: u64, n_vars: usize) -> (Vec<PFKb>, Vec<PackedQuinticExtensionFieldKB>) {
        let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let n_packed = (1usize << n_vars) / PFKb::WIDTH;
        let base: Vec<PFKb> = (0..n_packed)
            .map(|_| PFKb::from_fn(|_| KoalaBear::from_u64(rng.next_u64())))
            .collect();
        let ext: Vec<PackedQuinticExtensionFieldKB> = (0..n_packed)
            .map(|_| {
                PackedQuinticExtensionFieldKB::from_basis_coefficients_fn(|_| {
                    PFKb::from_fn(|_| KoalaBear::from_u64(rng.next_u64()))
                })
            })
            .collect();
        (base, ext)
    }

    pub(super) fn true_sum(base: &[PFKb], ext: &[PackedQuinticExtensionFieldKB]) -> EFKb {
        let mut acc = PackedQuinticExtensionFieldKB::ZERO;
        for (b, e) in base.iter().zip(ext.iter()) {
            acc += *e * *b;
        }
        <PackedQuinticExtensionFieldKB as PackedFieldExtension<KoalaBear, EFKb>>::to_ext_iter([acc]).sum::<EFKb>()
    }

    pub(super) fn mle_owned_to_ext_vec(m: &MleOwned<EFKb>) -> Vec<EFKb> {
        match m {
            MleOwned::Base(v) => v.iter().map(|&x| EFKb::from(x)).collect(),
            MleOwned::Extension(v) => v.to_vec(),
            MleOwned::BasePacked(v) => v.iter().flat_map(|p| p.as_slice().to_vec()).map(EFKb::from).collect(),
            MleOwned::ExtensionPacked(v) => {
                <PackedQuinticExtensionFieldKB as PackedFieldExtension<KoalaBear, EFKb>>::to_ext_iter(v.iter().copied())
                    .collect()
            }
        }
    }

    /// Harness self-check: the eager full path is deterministic under the
    /// recording FS prover (same seed -> identical transcript and outputs).
    #[test]
    fn eager_full_path_deterministic_under_mock() {
        for (seed, n_vars, n_rounds) in [(1u64, 8usize, 2usize), (2, 10, 3), (3, 12, 6)] {
            let (base, ext) = random_full_inputs(seed, n_vars);
            let sum = true_sum(&base, &ext);

            let mut fs_a = RecordingFs::new(seed);
            let out_a = run_product_sumcheck_base_eager::<DIM_KB, EFKb>(&base, &ext, &mut fs_a, sum, n_rounds, 0);
            let mut fs_b = RecordingFs::new(seed);
            let out_b = run_product_sumcheck_base_eager::<DIM_KB, EFKb>(&base, &ext, &mut fs_b, sum, n_rounds, 0);

            assert_eq!(fs_a.polys, fs_b.polys, "seed={seed}");
            assert_eq!(fs_a.challenges, fs_b.challenges, "seed={seed}");
            assert_eq!(out_a.0.0, out_b.0.0, "challenge points seed={seed}");
            assert_eq!(out_a.1, out_b.1, "final sum seed={seed}");
            assert_eq!(mle_owned_to_ext_vec(&out_a.2), mle_owned_to_ext_vec(&out_b.2));
            assert_eq!(mle_owned_to_ext_vec(&out_a.3), mle_owned_to_ext_vec(&out_b.3));
        }
    }

    /// Sanity: the eager path's final claim is consistent — the returned folds
    /// evaluated as a product reproduce the final sum.
    #[test]
    fn eager_full_path_final_claim_consistent() {
        for (seed, n_vars, n_rounds) in [(7u64, 9usize, 2usize), (8, 11, 4)] {
            let (base, ext) = random_full_inputs(seed, n_vars);
            let sum = true_sum(&base, &ext);
            let mut fs = RecordingFs::new(seed);
            let (_point, final_sum, pol_a, pol_b) =
                run_product_sumcheck_base_eager::<DIM_KB, EFKb>(&base, &ext, &mut fs, sum, n_rounds, 0);
            let a = mle_owned_to_ext_vec(&pol_a);
            let b = mle_owned_to_ext_vec(&pol_b);
            let recomposed: EFKb = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            assert_eq!(recomposed, final_sum, "seed={seed} n_vars={n_vars} n_rounds={n_rounds}");
        }
    }
}
