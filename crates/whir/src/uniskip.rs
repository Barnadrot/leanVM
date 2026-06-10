//! h8 (pw13-3): univariate skip for the initial WHIR folding sumcheck.
//!
//! The first `UNIVARIATE_SKIP_K` variables of the initial `folding_factor`
//! window are bound by ONE univariate challenge `r0` (Lagrange weights over
//! the 16-point integer window) instead of K tensor challenges; the remaining
//! `folding_factor − K` rounds run unchanged. The initial folding randomness
//! therefore has `1 + folding_factor − K` entries `[r0, r_K, …]`.
//!
//! This module hosts every skip-specific helper so that the protected files
//! (`open.rs`, `verify.rs`) only carry surgical call-sites. Soundness:
//! `report/hypothesis_8/security_regime.md` (BCIKS 2020/654 Thm 1.5/7.2 —
//! correlated agreement for degree-15 parameterized curves; same conjecture
//! class as the deployed gamma-power statement batching).

use field::ExtensionField;
use sumcheck::{UNIVARIATE_SKIP_K, lagrange_weights_at};

use crate::*;

/// Number of challenges produced by the initial sumcheck under the skip: one
/// univariate challenge `r0` (binding the top `UNIVARIATE_SKIP_K` variables)
/// plus one per remaining linear round.
pub(crate) fn n_initial_challenges(folding_factor_0: usize) -> usize {
    assert!(
        UNIVARIATE_SKIP_K < folding_factor_0,
        "the univariate skip must leave at least one linear round in the initial folding window"
    );
    1 + folding_factor_0 - UNIVARIATE_SKIP_K
}

/// The `2^folding_factor_0` per-leaf fold weights under the skip, for the
/// initial-round folding randomness `fr = [r0, r_K, …]`.
///
/// Leaf order derivation: a round-0 Merkle leaf holds the `2^ff0` values whose
/// global indices share the query's low bits; within the leaf, index `m`
/// corresponds to the global TOP-`ff0` bits, big-endian (`m`'s MSB = the
/// global MSB). The legacy fold `answer.evaluate(&[c0..c6])` is the MSB-first
/// MLE evaluation: challenge `c_b` pairs with leaf-index bit `ff0−1−b`, i.e.
/// `c0` ↔ the global MSB — the variable the first sumcheck round binds
/// (`product_computation` pairs `i` with `n/2 + i`). Under the skip, the top
/// `K` global variables are bound by `r0` with Lagrange weights over the
/// 16-block decomposition (block `j` = leaf-index bits `ff0−1..ff0−K`, i.e.
/// `j = m >> (ff0−K)`), and the remaining challenges `fr[1..]` bind the lower
/// leaf bits MSB-first, exactly `eval_eq`'s big-endian indexing. Hence
/// `w[m] = L_{m >> t}(r0) · eq_tail[m & (2^t − 1)]` with `t = ff0 − K`
/// (j-major) — the same linear functional the prover's `fold_product_skip`
/// plus the remaining linear rounds apply to the committed function.
pub(crate) fn skip_leaf_weights<EF: ExtensionField<PF<EF>>>(fr: &[EF]) -> Vec<EF> {
    let l = lagrange_weights_at::<PF<EF>, EF>(UNIVARIATE_SKIP_K, fr[0]);
    let eq_tail = eval_eq(&fr[1..]);
    let mut w = Vec::with_capacity(l.len() * eq_tail.len());
    for &lj in &l {
        for &et in eq_tail.iter() {
            w.push(lj * et);
        }
    }
    w
}

/// Verifier-side round-0 leaf fold: `dot(leaf, skip_leaf_weights(fr))`.
pub(crate) fn eval_leaf_skip<EF: ExtensionField<PF<EF>>>(leaf: &[EF], fr: &[EF]) -> EF {
    let w = skip_leaf_weights(fr);
    assert_eq!(leaf.len(), w.len());
    leaf.iter().zip(&w).map(|(&v, &wi)| wi * v).sum()
}

/// Prover-side round-0 leaf fold over a Merkle answer (base or extension
/// leaves, always unpacked).
pub(crate) fn eval_leaf_skip_owned<EF: ExtensionField<PF<EF>>>(
    answer: &MleOwned<EF>,
    fr: &MultilinearPoint<EF>,
) -> EF {
    let w = skip_leaf_weights(&fr.0);
    match answer {
        MleOwned::Base(leaf) => {
            assert_eq!(leaf.len(), w.len());
            leaf.iter().zip(&w).map(|(&v, &wi)| wi * EF::from(v)).sum()
        }
        MleOwned::Extension(leaf) => {
            assert_eq!(leaf.len(), w.len());
            leaf.iter().zip(&w).map(|(&v, &wi)| wi * v).sum()
        }
        _ => unreachable!("merkle answers are unpacked"),
    }
}

/// Round-0 constraint-weight evaluation under the skip.
///
/// For each of the `2^K` boolean prefixes `j`, run the LEGACY per-statement
/// weight body (`eval_constraints_poly`'s round loop) at the expanded point
/// `p_j = bits_K(j) ∥ point[1..]` (big-endian bits: `p_j[0]` = the global MSB
/// = `j`'s MSB) and combine with the Lagrange weights `L_j(r0)`,
/// `r0 = point[0]`. Exact for every statement shape (dense, sparse-selector,
/// tensor-tail, next, and head-overlap statements whose inner variables reach
/// into the skipped window) because binding `K` variables univariately is the
/// linear functional `W ↦ Σ_j L_j(r0)·W(bits_K(j), ·)`.
pub(crate) fn eval_round0_constraints<EF: ExtensionField<PF<EF>>>(
    randomness: &[EF],
    constraints: &[SparseStatement<EF>],
    point: &[EF],
) -> EF {
    let l = lagrange_weights_at::<PF<EF>, EF>(UNIVARIATE_SKIP_K, point[0]);
    let mut total = EF::ZERO;
    let mut p: Vec<EF> = Vec::with_capacity(UNIVARIATE_SKIP_K + point.len() - 1);
    for (j, &lj) in l.iter().enumerate() {
        p.clear();
        for b in 0..UNIVARIATE_SKIP_K {
            p.push(if (j >> (UNIVARIATE_SKIP_K - 1 - b)) & 1 == 1 {
                EF::ONE
            } else {
                EF::ZERO
            });
        }
        p.extend_from_slice(&point[1..]);
        // ——— legacy round body of `eval_constraints_poly`, evaluated at p ———
        let mut value = EF::ZERO;
        let mut i = 0;
        for smt in constraints {
            let inner_point = &p[p.len() - smt.inner_num_variables()..];
            let common_weight = match (&smt.tail, smt.is_next) {
                (Some(tail), true) => next_mle_with_tail(&smt.point.0, tail, inner_point),
                (Some(tail), false) => {
                    let (prefix, low) = inner_point.split_at(smt.point.len());
                    smt.point.eq_poly_outside(&MultilinearPoint(prefix.to_vec()))
                        * tail.evaluate(&MultilinearPoint(low.to_vec()))
                }
                (None, true) => next_mle(&smt.point.0, inner_point),
                (None, false) => smt.point.eq_poly_outside(&MultilinearPoint(inner_point.to_vec())),
            };
            for e in &smt.values {
                let eval = (0..smt.selector_num_variables())
                    .map(|sb| {
                        if e.selector & (1 << (smt.selector_num_variables() - 1 - sb)) == 0 {
                            EF::ONE - p[sb]
                        } else {
                            p[sb]
                        }
                    })
                    .product::<EF>()
                    * common_weight;
                value += eval * randomness[i];
                i += 1;
            }
        }
        debug_assert_eq!(i, randomness.len());
        total += lj * value;
    }
    total
}
