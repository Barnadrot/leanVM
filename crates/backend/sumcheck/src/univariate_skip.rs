//! Kernels for the univariate-skip round (Gruen, eprint 2024/108 §5-6).
//!
//! These are the convention-critical building blocks for replacing the first
//! `k` rounds of the batched AIR sumcheck with one univariate round over a
//! multiplicative subgroup `D` of size `2^k`. Everything here is generic over
//! runtime `k`/`b` (no benchmark shapes hardcoded).
//!
//! # Convention notes (load-bearing — pinned by the tests in this module)
//!
//! The sumcheck folds variables right-to-left: round `r` binds variable
//! `X_{L-1-r}`, which is storage-index bit `r` (LSB first). Folding a
//! multilinear LSB-first at the synthetic challenges
//! `c̃_r = r0^(2^(k-1-r))` (see [`synthetic_challenges`]) is exactly
//! evaluation at the spliced point whose last `k` coordinates are
//! `MultilinearPoint::expand_from_univariate(r0, k) = (r0, r0², …, r0^(2^(k-1)))`
//! (test `fold_at_synthetic_challenges_equals_expand_eval`).
//!
//! **However** (and this is a deviation from plan_spec §0/§2.4 discovered at
//! T1, see `report/iter1_T1_notes.md`): the univariate polynomial
//! `g(y) := MLE(v).evaluate(expand_from_univariate(y, k))` does **not** pass
//! through the raw block values `v_i` on `D` — its values on `D` are the
//! evals-DFT of `v` (cf. `whir/src/dft.rs::test_eval_dft`). Consequently:
//!
//! * the "expand reading" (sending `g`) is compatible with the downstream
//!   multilinear machinery (synthetic challenges) but does **not** satisfy
//!   the coset-sum identity `Σ_{y∈D} g(y) = Σ_z v_z`;
//! * the "values-on-`D` reading" (sending the interpolant `p` with
//!   `p(ω^i) = v_i`) satisfies the coset-sum identity, but `p(r0)` is the
//!   **Lagrange-weighted** combination `Σ_i L_i(r0)·v_i`
//!   ([`lagrange_evals_on_subgroup`]), which is *not* a tensor/multilinear
//!   point evaluation (test `values_on_d_binding_differs_from_fold`).
//!
//! A sound skip protocol over a non-zero sum (leanVM's batched AIR sum has
//! bus terms) therefore needs the values-on-`D` reading **plus** a terminal
//! Lagrange-to-tensor conversion. The kernels below support that protocol;
//! the decision on the conversion step is above this module's pay grade.

use fiat_shamir::FSProver;
use field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use poly::PF;

/// Synthetic challenges `c̃_r = r0^(2^(k-1-r))` for `r = 0..k`, in round order
/// (`c̃_0` first). For `k = 4`: `(r0⁸, r0⁴, r0², r0)`.
///
/// Computed with `k-1` squarings.
#[must_use]
pub fn synthetic_challenges<EF: Field>(r0: EF, k: usize) -> Vec<EF> {
    let mut out = vec![EF::ZERO; k];
    let mut cur = r0;
    for slot in out.iter_mut().rev() {
        *slot = cur;
        cur = cur.square();
    }
    out
}

/// `2^k · Σ_j coeffs[j · 2^k]`.
///
/// For a univariate `p` with coefficients `coeffs` and the multiplicative
/// subgroup `D` of order `2^k`, this equals `Σ_{y∈D} p(y)` (power sums over a
/// subgroup vanish except when `2^k` divides the exponent).
#[must_use]
pub fn coset_sum<EF: PrimeCharacteristicRing + Copy>(coeffs: &[EF], k: usize) -> EF {
    let step = 1usize << k;
    let mut acc = EF::ZERO;
    let mut j = 0;
    while j < coeffs.len() {
        acc += coeffs[j];
        j += step;
    }
    acc * EF::from_usize(step)
}

/// Horner evaluation of a univariate polynomial given by `coeffs` (constant
/// term first) at `x`.
#[must_use]
pub fn horner<EF: PrimeCharacteristicRing + Copy>(coeffs: &[EF], x: EF) -> EF {
    let mut acc = EF::ZERO;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Lagrange basis values `L_i(r0)` for the subgroup `D = <ω>` of order `2^k`,
/// with the convention that node `i` is `ω^i`:
///
/// `L_i(r0) = (r0^(2^k) − 1) · ω^i / (2^k · (r0 − ω^i))`.
///
/// `p(r0) = Σ_i v_i · L_i(r0)` for the unique degree `< 2^k` interpolant `p`
/// of values `v_i` at `ω^i`. Panics (division by zero) if `r0 ∈ D`; callers
/// sample `r0` from an extension field where `Pr[r0 ∈ D]` is negligible.
#[must_use]
pub fn lagrange_evals_on_subgroup<F: TwoAdicField, EF: ExtensionField<F>>(r0: EF, k: usize) -> Vec<EF> {
    let m = 1usize << k;
    let omega = F::two_adic_generator(k);
    let zh = r0.exp_power_of_2(k) - EF::ONE; // r0^{2^k} − 1
    let m_inv = F::from_usize(m).inverse();
    let scale = zh * m_inv;
    let mut out = Vec::with_capacity(m);
    let mut w = F::ONE; // ω^i
    for _ in 0..m {
        out.push(scale * w * (r0 - w).inverse());
        w *= omega;
    }
    out
}

/// Coherence subset-sums (plan_spec v2 §1): from the global Lagrange values
/// `global[ĩ] = L^{(k)}_ĩ(r0)` (length `2^k`, node `ĩ ↔ ω_k^ĩ`), produce the
/// length-`2^b` vector `out[m] = L^{(b)}_m(r0^{2^{k−b}})` via
/// `out[m] = Σ_{ĩ ≡ m (mod 2^b)} global[ĩ]`.
///
/// Identity: values that are `2^b`-periodic in the node index interpolate to a
/// polynomial in `y^{2^{k−b}}` (degree `< 2^k` uniqueness), so the weight any
/// `v_m` receives at `r0` is the same on both sides — for all `v` — giving
/// coefficient-wise equality. Pinned by `sub_block_lagrange_coherence` below.
#[must_use]
pub fn sub_block_lagrange_from_global<EF: PrimeCharacteristicRing + Copy>(global: &[EF], b: usize) -> Vec<EF> {
    let m = 1usize << b;
    assert!(global.len() >= m && global.len().is_multiple_of(m));
    let mut out = vec![EF::ZERO; m];
    for (i, &g) in global.iter().enumerate() {
        out[i & (m - 1)] += g;
    }
    out
}

/// Prover side of the terminal Lagrange-to-tensor conversion sumcheck
/// (plan_spec v2 §2.3): a `b`-round, degree-2 product sumcheck of the two
/// `2^b`-sized extension arrays `weights` (public Lagrange weights `ℓ_j`) and
/// `g_gamma` (the γ-RLC'd block partial evaluations `G_γ(j)`).
///
/// Round `r` binds bit `r` of the block-local index `j` (LSB-first, mirroring
/// the global fold convention). Each round message is sent with the
/// `eq_alpha: None` Fiat–Shamir path (`add_sumcheck_polynomial(&[c0,c1,c2],
/// None)`), matching what `sumcheck_verify(state, b, 2, claim, None)` expects.
///
/// Returns `(challenges in round order, w(s)·g(s))`. The MLE point for either
/// array is the REVERSED challenge vector (round `r` ↔ MLE coordinate
/// `b−1−r`).
pub fn prove_weighted_block_sumcheck<EF: ExtensionField<PF<EF>>>(
    prover_state: &mut impl FSProver<EF>,
    mut weights: Vec<EF>,
    mut g_gamma: Vec<EF>,
) -> (Vec<EF>, EF) {
    assert_eq!(weights.len(), g_gamma.len());
    assert!(weights.len().is_power_of_two() && weights.len() >= 2);
    let b = weights.len().ilog2() as usize;
    let mut challenges = Vec::with_capacity(b);
    for _ in 0..b {
        let half = weights.len() / 2;
        let (mut c0, mut c1, mut c2) = (EF::ZERO, EF::ZERO, EF::ZERO);
        for m in 0..half {
            let (w0, w1) = (weights[2 * m], weights[2 * m + 1]);
            let (g0, g1) = (g_gamma[2 * m], g_gamma[2 * m + 1]);
            let dw = w1 - w0;
            let dg = g1 - g0;
            c0 += w0 * g0;
            c1 += w0 * dg + g0 * dw;
            c2 += dw * dg;
        }
        prover_state.add_sumcheck_polynomial(&[c0, c1, c2], None);
        let r = prover_state.sample();
        challenges.push(r);
        for m in 0..half {
            weights[m] = weights[2 * m] + (weights[2 * m + 1] - weights[2 * m]) * r;
            g_gamma[m] = g_gamma[2 * m] + (g_gamma[2 * m + 1] - g_gamma[2 * m]) * r;
        }
        weights.truncate(half);
        g_gamma.truncate(half);
    }
    (challenges, weights[0] * g_gamma[0])
}

fn bit_reverse_indices(b: usize) -> Vec<usize> {
    let m = 1usize << b;
    (0..m).map(|i| i.reverse_bits() >> (usize::BITS as usize - b)).collect()
}

/// In-place radix-2 DFT across whole arrays.
///
/// `slices[i]` holds, for every position `p`, the coefficient (forward) or
/// evaluation (inverse) of one univariate polynomial per position. Forward:
/// coefficients in, evaluations out with `out[j] = p(ω^j)`, `ω = root` of
/// order `2^b`, indices in natural order.
fn dft_arrays_in_place<F: Field, A: Algebra<F> + Copy>(slices: &mut [Vec<A>], root: F, b: usize) {
    let m = 1usize << b;
    debug_assert_eq!(slices.len(), m);
    // Decimation-in-time: bit-reverse the slice order, then butterfly upward.
    let rev = bit_reverse_indices(b);
    for (i, &ri) in rev.iter().enumerate() {
        if ri > i {
            slices.swap(i, ri);
        }
    }
    for s in 1..=b {
        let half = 1usize << (s - 1);
        let step = root.exp_u64((m >> s) as u64); // ω^(2^b / 2^s)
        let mut chunk = 0;
        while chunk < m {
            let mut w = F::ONE;
            for t in 0..half {
                let (i0, i1) = (chunk + t, chunk + t + half);
                let (lo, hi) = slices.split_at_mut(i1);
                let a0 = &mut lo[i0];
                let a1 = &mut hi[0];
                for p in 0..a0.len() {
                    let u = a0[p];
                    let v = a1[p] * w;
                    a0[p] = u + v;
                    a1[p] = u - v;
                }
                w *= step;
            }
            chunk += 2 * half;
        }
    }
}

/// In-place inverse evals-to-coeffs DFT across whole arrays.
///
/// On entry `slices[i]` holds, for every position, the value at `ω^i` of one
/// univariate polynomial per position (`ω = two_adic_generator(b)`, node
/// `i ↔ ω^i`). On exit `slices[e]` holds coefficient `e` (degree `< 2^b`).
pub fn block_ifft_arrays<F: TwoAdicField, A: Algebra<F> + Copy>(slices: &mut [Vec<A>], b: usize) {
    let m = 1usize << b;
    assert_eq!(slices.len(), m, "need exactly 2^b slices");
    let root_inv = F::two_adic_generator(b).inverse();
    dft_arrays_in_place(slices, root_inv, b);
    let m_inv = F::from_usize(m).inverse();
    for slice in slices.iter_mut() {
        for v in slice.iter_mut() {
            *v *= m_inv;
        }
    }
}

/// From coefficient arrays (as produced by [`block_ifft_arrays`]), evaluate
/// every per-position polynomial on the coset `g·D`:
/// `out[j][p] = Σ_e coeffs[e][p] · (g·ω^j)^e`.
///
/// `out` must have the same shape as `coeff_slices`.
pub fn block_coset_evals<F: TwoAdicField, A: Algebra<F> + Copy>(
    coeff_slices: &[Vec<A>],
    b: usize,
    coset_gen: F,
    out: &mut [Vec<A>],
) {
    let m = 1usize << b;
    assert_eq!(coeff_slices.len(), m);
    assert_eq!(out.len(), m);
    // out ← coeffs scaled by g^e, then forward DFT in place.
    let mut g_pow = F::ONE;
    for (e, (dst, src)) in out.iter_mut().zip(coeff_slices.iter()).enumerate() {
        debug_assert!(e < m);
        dst.clear();
        dst.extend(src.iter().map(|&c| c * g_pow));
        g_pow *= coset_gen;
    }
    dft_arrays_in_place(out, F::two_adic_generator(b), b);
}

#[cfg(test)]
mod tests {
    use super::*;
    use koala_bear::{KoalaBear, QuinticExtensionFieldKB};
    use poly::{EvaluationsList, MultilinearPoint};

    type F = KoalaBear;
    type EF = QuinticExtensionFieldKB;

    /// Minimal deterministic PRNG (xorshift64*), avoiding a dev-dependency.
    /// The identities under test are basis-independent, so base-field-embedded
    /// pseudo-random extension elements are sufficient.
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
        fn ef(&mut self) -> EF {
            // Random across all 5 basis coefficients (not just the embedded base).
            let coeffs: [KoalaBear; 5] = std::array::from_fn(|_| KoalaBear::from_u64(self.next_u64()));
            field::BasedVectorSpace::<KoalaBear>::from_basis_coefficients_slice(&coeffs).unwrap()
        }
    }

    /// LSB-first fold: round `r` binds storage bit 0 of the current array at
    /// challenge `c` — the canonical semantics of the prover's fold schedule
    /// (air_sumcheck.rs folds right-to-left; custom bit-reversed storage is an
    /// internal layout detail with identical semantics).
    fn fold_lsb<EF2: Field>(values: &[EF2], c: EF2) -> Vec<EF2> {
        values.chunks_exact(2).map(|p| p[0] + (p[1] - p[0]) * c).collect()
    }

    #[test]
    fn synthetic_challenges_are_squaring_chain() {
        let mut rng = Rng(1);
        for k in 1..=6 {
            let r0 = rng.ef();
            let c = synthetic_challenges(r0, k);
            assert_eq!(c.len(), k);
            for (r, &cr) in c.iter().enumerate() {
                assert_eq!(cr, r0.exp_u64(1u64 << (k - 1 - r)), "k={k} r={r}");
            }
        }
    }

    /// Plan §0 equivalence, multilinear-binding side (PASSES):
    /// folding the first k rounds at c̃ equals evaluating the full MLE at the
    /// spliced point whose last k coordinates are expand_from_univariate(r0,k).
    #[test]
    fn fold_at_synthetic_challenges_equals_expand_eval() {
        let mut rng = Rng(2);
        for n in 6..=8 {
            for k in 2..=4 {
                let evals: Vec<EF> = (0..1usize << n).map(|_| rng.ef()).collect();
                let r0 = rng.ef();

                // OLD way: k LSB-first folds at synthetic challenges, then bind
                // the remaining n-k variables at random challenges γ.
                let mut arr = evals.clone();
                for &c in &synthetic_challenges(r0, k) {
                    arr = fold_lsb(&arr, c);
                }
                let gammas: Vec<EF> = (0..n - k).map(|_| rng.ef()).collect();
                for &g in &gammas {
                    arr = fold_lsb(&arr, g);
                }
                assert_eq!(arr.len(), 1);

                // Reference: full MLE evaluation. Variable X_j is bound at
                // round n-1-j, so point[j] = challenge[n-1-j]: the first n-k
                // coordinates are the γ's reversed, the last k coordinates are
                // expand_from_univariate(r0, k) = (r0, r0², …).
                let mut point: Vec<EF> = gammas.iter().rev().copied().collect();
                point.extend(MultilinearPoint::expand_from_univariate(r0, k).0);
                let reference = evals.evaluate(&MultilinearPoint(point));

                assert_eq!(arr[0], reference, "n={n} k={k}");
            }
        }
    }

    /// Plan §0/§2.4 incompatibility (DOCUMENTED DEVIATION, see module docs and
    /// report/iter1_T1_notes.md): the values-on-D interpolant of the raw block
    /// values, evaluated at r0, is the Lagrange-weighted combination — it does
    /// NOT equal the multilinear fold at the synthetic challenges. k=1
    /// closed-form: fold = (1−r0)v0 + r0·v1, interp = (v0+v1+r0(v0−v1))/2.
    #[test]
    fn values_on_d_binding_differs_from_fold() {
        let mut rng = Rng(3);
        for k in 1..=4 {
            let m = 1usize << k;
            let evals: Vec<EF> = (0..m).map(|_| rng.ef()).collect();
            let r0 = rng.ef();

            // Values-on-D interpolation (node i ↔ ω^i) via the block kernels.
            let mut slices: Vec<Vec<EF>> = evals.iter().map(|&v| vec![v]).collect();
            block_ifft_arrays::<F, EF>(&mut slices, k);
            let coeffs: Vec<EF> = slices.iter().map(|s| s[0]).collect();
            let interp_at_r0 = horner(&coeffs, r0);

            // Lagrange-weight form agrees with the interpolant (kernel sanity).
            let weights = lagrange_evals_on_subgroup::<F, EF>(r0, k);
            let lagrange_combo: EF = evals.iter().zip(&weights).map(|(&v, &w)| v * w).sum();
            assert_eq!(interp_at_r0, lagrange_combo, "k={k}");

            // Multilinear fold at synthetic challenges — a DIFFERENT value.
            let mut arr = evals.clone();
            for &c in &synthetic_challenges(r0, k) {
                arr = fold_lsb(&arr, c);
            }
            assert_ne!(arr[0], interp_at_r0, "k={k}: the two readings coincided unexpectedly");

            if k == 1 {
                let half = EF::from_usize(2).inverse();
                assert_eq!(interp_at_r0, (evals[0] + evals[1] + r0 * (evals[0] - evals[1])) * half);
                assert_eq!(arr[0], evals[0] + (evals[1] - evals[0]) * r0);
            }
        }
    }

    /// Test (b): coset_sum equals the direct sum of p over the subgroup.
    #[test]
    fn coset_sum_matches_direct_subgroup_sum() {
        let mut rng = Rng(4);
        for k in 2..=5 {
            let n_coeffs = 9 * ((1usize << k) - 1) + 1; // plan's N+1 at d=9
            let coeffs: Vec<EF> = (0..n_coeffs).map(|_| rng.ef()).collect();
            let omega = F::two_adic_generator(k);
            let mut direct = EF::ZERO;
            let mut y = F::ONE;
            for _ in 0..1usize << k {
                direct += horner(&coeffs, EF::from(y));
                y *= omega;
            }
            assert_eq!(coset_sum(&coeffs, k), direct, "k={k}");
        }
    }

    /// Block kernels round-trip: evals → coeffs → evals (g = 1), and coset
    /// evaluations match direct Horner at g·ω^j.
    #[test]
    fn block_kernels_roundtrip_and_coset() {
        let mut rng = Rng(5);
        for b in 1..=5 {
            let m = 1usize << b;
            let n_pos = 7;
            let evals: Vec<Vec<EF>> = (0..m).map(|_| (0..n_pos).map(|_| rng.ef()).collect()).collect();

            let mut coeffs = evals.clone();
            block_ifft_arrays::<F, EF>(&mut coeffs, b);

            // Round-trip with g = 1 reproduces the input evals.
            let mut back: Vec<Vec<EF>> = vec![Vec::new(); m];
            block_coset_evals::<F, EF>(&coeffs, b, F::ONE, &mut back);
            assert_eq!(back, evals, "b={b} round-trip");

            // Off-coset: compare against per-position Horner at g·ω^j.
            let g = F::two_adic_generator(b + 2); // any non-D coset rep
            let mut coset: Vec<Vec<EF>> = vec![Vec::new(); m];
            block_coset_evals::<F, EF>(&coeffs, b, g, &mut coset);
            let omega = F::two_adic_generator(b);
            let mut point = g;
            for (j, coset_j) in coset.iter().enumerate() {
                for p in 0..n_pos {
                    let poly: Vec<EF> = coeffs.iter().map(|s| s[p]).collect();
                    assert_eq!(coset_j[p], horner(&poly, EF::from(point)), "b={b} j={j} p={p}");
                }
                point *= omega;
            }
        }
    }

    /// Coherence identity (plan_spec v2 §1, review invariant 2d):
    /// `sub_block_lagrange_from_global(lagrange_evals_on_subgroup(r0, k), b)`
    /// equals `lagrange_evals_on_subgroup(r0^(2^(k−b)), b)`.
    #[test]
    fn sub_block_lagrange_coherence() {
        let mut rng = Rng(7);
        for k in 3..=5usize {
            for b in 1..=k {
                let r0 = rng.ef();
                let global = lagrange_evals_on_subgroup::<F, EF>(r0, k);
                let derived = sub_block_lagrange_from_global(&global, b);
                let rho = r0.exp_power_of_2(k - b);
                let direct = lagrange_evals_on_subgroup::<F, EF>(rho, b);
                assert_eq!(derived, direct, "k={k} b={b}");
                // Σ_m L_m = 1 (interpolant of the constant 1) — the property
                // that makes Lagrange-folding preserve constant padding blocks.
                let one: EF = derived.iter().copied().sum();
                assert_eq!(one, EF::ONE, "k={k} b={b}");
            }
        }
    }

    /// Test (d), values-on-D reading: a degree-2 composition over a reindexed
    /// D×H table. v0's coefficients are recovered from its values on the
    /// size-2^{k+1} subgroup (D ∪ ω_{k+1}·D), and coset_sum(v0) equals the
    /// direct table sum Σ_{i,x} C(P[x·2^k+i]) — the identity the verifier
    /// checks against initial_sum.
    #[test]
    fn mini_e2e_values_on_d_coset_sum() {
        let comp = |v: EF| v * v + v; // degree-2 "constraint"
        let mut rng = Rng(6);
        let (n, k) = (8usize, 3usize);
        let m = 1usize << k;
        let table: Vec<EF> = (0..1usize << n).map(|_| rng.ef()).collect();

        // Direct sum over the reindexed table (= initial_sum in the protocol).
        let direct: EF = table.iter().map(|&v| comp(v)).sum();

        // Prover side: per remaining-index x, the block values live at ω_k^i.
        let n_pos = 1usize << (n - k);
        let mut slices: Vec<Vec<EF>> = (0..m)
            .map(|i| (0..n_pos).map(|x| table[(x << k) | i]).collect())
            .collect();

        // v0 on D: composition applied pointwise to raw values, summed over x.
        let v0_on_d: Vec<EF> = slices.iter().map(|s| s.iter().map(|&v| comp(v)).sum::<EF>()).collect();

        // v0 on ω_{k+1}·D: extend each block polynomial off-coset, compose, sum.
        block_ifft_arrays::<F, EF>(&mut slices, k);
        let g = F::two_adic_generator(k + 1);
        let mut off: Vec<Vec<EF>> = vec![Vec::new(); m];
        block_coset_evals::<F, EF>(&slices, k, g, &mut off);
        let v0_on_gd: Vec<EF> = off.iter().map(|s| s.iter().map(|&v| comp(v)).sum::<EF>()).collect();

        // Interleave into evals on the size-2^{k+1} subgroup: ω_{k+1}^{2j} = ω_k^j,
        // ω_{k+1}^{2j+1} = g·ω_k^j. deg v0 ≤ 2(2^k−1) < 2^{k+1} ✓.
        let mut v0_slices: Vec<Vec<EF>> = (0..2 * m)
            .map(|idx| vec![if idx % 2 == 0 { v0_on_d[idx / 2] } else { v0_on_gd[idx / 2] }])
            .collect();
        block_ifft_arrays::<F, EF>(&mut v0_slices, k + 1);
        let v0_coeffs: Vec<EF> = v0_slices.iter().map(|s| s[0]).collect();

        // Top coefficient must vanish (degree bound) and the coset-sum
        // identity must reproduce the direct table sum.
        assert_eq!(v0_coeffs[2 * m - 1], EF::ZERO);
        assert_eq!(coset_sum(&v0_coeffs, k), direct);

        // Internal consistency: v0(r0) == Σ_x C(f_x(r0)).
        let r0 = rng.ef();
        let per_x: EF = (0..n_pos)
            .map(|p| {
                let poly: Vec<EF> = slices.iter().map(|s| s[p]).collect();
                comp(horner(&poly, r0))
            })
            .sum();
        assert_eq!(horner(&v0_coeffs, r0), per_x);
    }
}
