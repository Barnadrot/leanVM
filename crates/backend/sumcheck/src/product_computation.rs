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
    let first_sumcheck_poly = match (pol_a, pol_b) {
        (MleRef::BasePacked(evals), MleRef::ExtensionPacked(weights)) => {
            if EF::DIMENSION == 3 {
                compute_product_sumcheck_polynomial_base_ext_packed::<3, _, _, _, EF>(evals, weights, sum)
            } else {
                unimplemented!()
            }
        }
        (MleRef::ExtensionPacked(evals), MleRef::ExtensionPacked(weights)) => {
            if EF::DIMENSION == 3 {
                compute_product_sumcheck_polynomial_ext_ext_packed::<3, _, _, _, EF>(evals, weights, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                })
            } else {
                compute_product_sumcheck_polynomial(evals, weights, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                })
            }
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
        (MleRef::BasePacked(evals), MleRef::ExtensionPacked(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                });
            (second_sumcheck_poly, MleGroupOwned::ExtensionPacked(folded))
        }
        (MleRef::ExtensionPacked(evals), MleRef::ExtensionPacked(weights)) => {
            let (second_sumcheck_poly, folded) = if EF::DIMENSION == 3 {
                fold_and_compute_product_sumcheck_polynomial_ext_ext_packed::<3, _, _, _, EF>(
                    evals,
                    weights,
                    r1,
                    sum,
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                )
            } else {
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                })
            };
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

/// Deferred-reduction cubic Karatsuba multiply-accumulate (plan_spec §3.2, Target B).
///
/// Accumulates the product `a * b` of two cubic-extension elements (given as their
/// 3 packed base coefficients) into per-output-coefficient lazy accumulators,
/// reductions deferred to `lazy_acc_finish`. Encodes the `F_p[X]/(X^3 - X - 1)`
/// Karatsuba combination of `cubic_mul_generic` (cubic_extension.rs:484-512):
///   res0 = m0 + m5 - m1 - m2          (2 subs)
///   res1 = m3 + m5 - m0 - m1 - m1     (3 subs)
///   res2 = m4 + m1 - m0               (1 sub)
/// so callers must pass `n_sub = [2, 3, 1] * n_iterations` to the finishes
/// (see `finish_cubic_lazy`).
///
/// The `-2*m1` term in res1 is TWO `lazy_acc_sub(m1)` calls: pre-doubling a
/// multiplicand before `unreduced_mul` is unsound (operands are non-canonical
/// u64 — doubling can wrap mod 2^64), and a separate `unreduced_mul(2*a1, b1)`
/// with a canonically doubled operand would cost a full extra mul (16 instr)
/// vs one extra lazy sub (8 instr).
///
/// Per-accumulator terms per call: res0: 4, res1: 5, res2: 3 — within the
/// `unreduced_mul` protocol bound of 5 terms/iteration (field.rs).
/// Each product is computed once and placed immediately into every accumulator
/// that consumes it, keeping the live working set at one product + the three
/// 4-slot accumulators (T4 register-pressure lesson).
#[inline(always)]
pub fn lazy_cubic_quadratic_acc<PF: PrimeCharacteristicRing + Copy>(acc: &mut [[PF; 4]; 3], a: &[PF], b: &[PF]) {
    let (a0, a1, a2) = (a[0], a[1], a[2]);
    let (b0, b1, b2) = (b[0], b[1], b[2]);

    let m0 = PF::unreduced_mul(a0, b0);
    acc[0] = PF::lazy_acc_add(acc[0], m0);
    acc[1] = PF::lazy_acc_sub(acc[1], m0);
    acc[2] = PF::lazy_acc_sub(acc[2], m0);

    let m1 = PF::unreduced_mul(a1, b1);
    acc[0] = PF::lazy_acc_sub(acc[0], m1);
    acc[1] = PF::lazy_acc_sub(acc[1], m1);
    acc[1] = PF::lazy_acc_sub(acc[1], m1);
    acc[2] = PF::lazy_acc_add(acc[2], m1);

    let m2 = PF::unreduced_mul(a2, b2);
    acc[0] = PF::lazy_acc_sub(acc[0], m2);

    let m3 = PF::unreduced_mul(a0 + a1, b0 + b1);
    acc[1] = PF::lazy_acc_add(acc[1], m3);

    let m4 = PF::unreduced_mul(a0 + a2, b0 + b2);
    acc[2] = PF::lazy_acc_add(acc[2], m4);

    let m5 = PF::unreduced_mul(a1 + a2, b1 + b2);
    acc[0] = PF::lazy_acc_add(acc[0], m5);
    acc[1] = PF::lazy_acc_add(acc[1], m5);
}

/// Sub-counts per output coefficient for one `lazy_cubic_quadratic_acc` call.
pub const CUBIC_LAZY_SUBS_PER_ITER: [u64; 3] = [2, 3, 1];

/// Finish the three per-coefficient lazy accumulators after `n_iters` calls of
/// `lazy_cubic_quadratic_acc`.
#[inline(always)]
pub fn finish_cubic_lazy<PF: PrimeCharacteristicRing + Copy>(acc: [[PF; 4]; 3], n_iters: u64) -> [PF; 3] {
    core::array::from_fn(|j| PF::lazy_acc_finish(acc[j], CUBIC_LAZY_SUBS_PER_ITER[j] * n_iters))
}

/// Chunk length for the ext-x-ext lazy kernels. res1 receives `m1` twice in the
/// same iteration, so its per-iteration term count is 5 — the protocol maximum;
/// 1024 <= 2^20 honors the chunk bound of the lazy-accumulator contract.
const EXT_EXT_CHUNK_SIZE: usize = 1024;

/// Round-1 ext-x-ext product-sumcheck kernel with deferred cubic Karatsuba
/// accumulation (plan_spec §3.2, Target B). Value-identical to
/// `compute_product_sumcheck_polynomial`; dispatched only for `EF::DIMENSION == 3`
/// (the workspace's only cubic extension is `CubicExtensionFieldGL`, whose
/// reduction rule this kernel hardcodes — pinned by the oracle equality test in
/// crates/sub_protocols/tests/ext_ext_lazy_kernel.rs).
pub fn compute_product_sumcheck_polynomial_ext_ext_packed<
    const DIM: usize,
    F: PrimeField64,
    PF: PackedField<Scalar = F>,
    EFP: Algebra<PF> + BasedVectorSpace<PF> + Copy + Send + Sync,
    EF: Field + BasedVectorSpace<F>,
>(
    pol_0: &[EFP],
    pol_1: &[EFP],
    sum: EF,
    decompose: impl Fn(EFP) -> Vec<EF>,
) -> DensePolynomial<EF> {
    assert_eq!(DIM, 3);
    assert_eq!(DIM, EF::DIMENSION);
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());

    if n < PARALLEL_THRESHOLD {
        // Tiny inputs: the eager generic path, identical semantics, not hot.
        return compute_product_sumcheck_polynomial(pol_0, pol_1, sum, decompose);
    }

    let half = n / 2;
    let n_chunks = half.div_ceil(EXT_EXT_CHUNK_SIZE);

    let (c0_acc, c2_acc) = parallel::map_reduce(
        n_chunks,
        || ([PF::ZERO; 3], [PF::ZERO; 3]),
        |chunk| {
            let start = chunk * EXT_EXT_CHUNK_SIZE;
            let end = (start + EXT_EXT_CHUNK_SIZE).min(half);
            let n_iters = (end - start) as u64;
            // Two passes over the (L2-resident) chunk, halving live accumulator
            // registers per loop (T4 lesson: 12 accumulator zmm per pass fits).
            let mut c0_lazy: [[PF; 4]; 3] = core::array::from_fn(|_| PF::lazy_acc_zero());
            for i in start..end {
                lazy_cubic_quadratic_acc(
                    &mut c0_lazy,
                    pol_1[i].as_basis_coefficients_slice(),
                    pol_0[i].as_basis_coefficients_slice(),
                );
            }
            let c0 = finish_cubic_lazy(c0_lazy, n_iters);
            let mut c2_lazy: [[PF; 4]; 3] = core::array::from_fn(|_| PF::lazy_acc_zero());
            for i in start..end {
                let dx = pol_0[half + i] - pol_0[i];
                let dy = pol_1[half + i] - pol_1[i];
                lazy_cubic_quadratic_acc(
                    &mut c2_lazy,
                    dy.as_basis_coefficients_slice(),
                    dx.as_basis_coefficients_slice(),
                );
            }
            let c2 = finish_cubic_lazy(c2_lazy, n_iters);
            (c0, c2)
        },
        |(mut a0, mut a2): ([PF; 3], [PF; 3]), (b0, b2): ([PF; 3], [PF; 3])| {
            for j in 0..3 {
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

/// Round-2 fold + ext-x-ext product-sumcheck kernel with deferred cubic Karatsuba
/// accumulation (plan_spec §3.2, Target B). The fold/store part writing
/// `pol_*_folded` is value-identical to `fold_and_compute_product_sumcheck_polynomial`
/// (those values feed the transcript path and stay canonical); only the round-poly
/// accumulation defers reductions. Per chunk: pass 1 folds (writing the chunk's
/// disjoint lo/hi quarters), passes 2-3 lazily accumulate from the just-written
/// L2-resident folded data (T4 register-pressure split).
pub fn fold_and_compute_product_sumcheck_polynomial_ext_ext_packed<
    const DIM: usize,
    F: PrimeField64,
    PF: PackedField<Scalar = F>,
    EFP: Algebra<PF> + BasedVectorSpace<PF> + From<EF> + Copy + Send + Sync + 'static,
    EF: Field + BasedVectorSpace<F>,
>(
    pol_0: &[EFP],
    pol_1: &[EFP],
    prev_folding_factor: EF,
    sum: EF,
    decompose: impl Fn(EFP) -> Vec<EF>,
) -> (DensePolynomial<EF>, Vec<ArenaVec<EFP>>) {
    assert_eq!(DIM, 3);
    assert_eq!(DIM, EF::DIMENSION);
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());

    if n < PARALLEL_THRESHOLD {
        // Tiny inputs: the eager generic path, identical semantics, not hot.
        return fold_and_compute_product_sumcheck_polynomial(pol_0, pol_1, prev_folding_factor, sum, decompose);
    }

    let prev_folding_factor_packed = EFP::from(prev_folding_factor);
    let quarter = n / 4;
    let n_chunks = quarter.div_ceil(EXT_EXT_CHUNK_SIZE);

    let mut pol_0_folded = unsafe { ArenaVec::<EFP>::uninitialized(n / 2) };
    let mut pol_1_folded = unsafe { ArenaVec::<EFP>::uninitialized(n / 2) };
    let p0f = parallel::SendPtr(pol_0_folded.as_mut_ptr());
    let p1f = parallel::SendPtr(pol_1_folded.as_mut_ptr());

    let (c0_acc, c2_acc) = parallel::map_reduce(
        n_chunks,
        || ([PF::ZERO; 3], [PF::ZERO; 3]),
        |chunk| {
            let start = chunk * EXT_EXT_CHUNK_SIZE;
            let end = (start + EXT_EXT_CHUNK_SIZE).min(quarter);
            let n_iters = (end - start) as u64;
            // Pass 1: fold (canonical arithmetic, identical to the generic path).
            // SAFETY: chunks own disjoint index ranges {i, quarter+i : i in [start, end)}.
            for i in start..end {
                let x_0 = prev_folding_factor_packed * (pol_0[2 * quarter + i] - pol_0[i]) + pol_0[i];
                let x_1 =
                    prev_folding_factor_packed * (pol_0[3 * quarter + i] - pol_0[quarter + i]) + pol_0[quarter + i];
                let y_0 = prev_folding_factor_packed * (pol_1[2 * quarter + i] - pol_1[i]) + pol_1[i];
                let y_1 =
                    prev_folding_factor_packed * (pol_1[3 * quarter + i] - pol_1[quarter + i]) + pol_1[quarter + i];
                unsafe {
                    *p0f.add(i) = x_0;
                    *p0f.add(quarter + i) = x_1;
                    *p1f.add(i) = y_0;
                    *p1f.add(quarter + i) = y_1;
                }
            }
            // Passes 2-3: lazy round-poly accumulation from the folded chunk.
            // SAFETY: reads only this chunk's just-written slots.
            let mut c0_lazy: [[PF; 4]; 3] = core::array::from_fn(|_| PF::lazy_acc_zero());
            for i in start..end {
                let (x_0, y_0) = unsafe { (*p0f.add(i), *p1f.add(i)) };
                lazy_cubic_quadratic_acc(
                    &mut c0_lazy,
                    y_0.as_basis_coefficients_slice(),
                    x_0.as_basis_coefficients_slice(),
                );
            }
            let c0 = finish_cubic_lazy(c0_lazy, n_iters);
            let mut c2_lazy: [[PF; 4]; 3] = core::array::from_fn(|_| PF::lazy_acc_zero());
            for i in start..end {
                let (dx, dy) = unsafe { (*p0f.add(quarter + i) - *p0f.add(i), *p1f.add(quarter + i) - *p1f.add(i)) };
                lazy_cubic_quadratic_acc(
                    &mut c2_lazy,
                    dy.as_basis_coefficients_slice(),
                    dx.as_basis_coefficients_slice(),
                );
            }
            let c2 = finish_cubic_lazy(c2_lazy, n_iters);
            (c0, c2)
        },
        |(mut a0, mut a2): ([PF; 3], [PF; 3]), (b0, b2): ([PF; 3], [PF; 3])| {
            for j in 0..3 {
                a0[j] += b0[j];
                a2[j] += b2[j];
            }
            (a0, a2)
        },
    );

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
