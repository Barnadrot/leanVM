//! Univariate-skip kernel for the WHIR initial folding sumcheck (pw13-3, h8).
//!
//! The initial WHIR sumcheck runs `folding_factor` degree-2 rounds of
//! `Σ f̂·ŵ` over the stacked polynomial (f base field at round 0, extension
//! afterwards). The skip round binds the first `k` variables at once:
//!
//!   v′(X) = Σ_rest f̂(X, rest)·ŵ(X, rest),   deg v′ ≤ 2·(2^k − 1),
//!
//! sampled on the consecutive-integer nodes of [`skip_all_nodes`]`(k, 2)` and
//! sent in FULL coefficient form. The verifier's round-0 identity is the
//! window sum `Σ_{j<2^k} v′(j) == claimed_sum` (a dot product with the
//! [`window_power_sums`](crate::window_power_sums) constants — there is NO eq
//! factor here), and the next target is `v′(r0)`.
//!
//! Index convention (matches `product_computation.rs`, which pairs `i` with
//! `n/2 + i`, i.e. binds the MOST-significant variable per round): window node
//! `j` is the j-th CONTIGUOUS block of length `len >> k` — top-k index bits
//! equal `j`, big-endian, NO bit-reversal (unlike the AIR skip's chunk-reversed
//! storage). Blocks stay contiguous in packed storage because their length is
//! far above the SIMD lane bits for every production shape.
//!
//! Extension to the integer nodes beyond the window uses the shared
//! forward-difference helpers ([`fd_init_in_place`]/[`fd_advance`]): base-field
//! adds for f, extension adds for w — bit-identical to the Lagrange reference
//! path (`force_lagrange = true`), which is kept for tests and the kill-gate
//! bench.

use std::ops::Mul;

use field::*;
use poly::*;
use zk_alloc::ArenaVec;

use crate::{fd_advance, fd_init_in_place, lagrange_coeffs_for_targets, skip_all_nodes};

/// v′(X) = Σ_rest f̂(X, rest)·ŵ(X, rest) in coefficient form (degree ≤
/// 2·(2^k − 1)). `force_lagrange` selects the Lagrange-coefficient reference
/// extension instead of the default forward-difference one (bit-identical
/// outputs; used by tests and the kill-gate bench).
///
/// Supported operand shapes mirror `run_product_sumcheck`: the production
/// combination is `BasePacked × ExtensionPacked` (round-0 stacked polynomial ×
/// combined statement weights); `ExtensionPacked × ExtensionPacked` and the
/// unpacked `Base/Extension × Extension` variants serve tests and sub-packing
/// sizes.
pub fn compute_product_skip_poly<EF: ExtensionField<PF<EF>>>(
    evals: &MleRef<'_, EF>,
    weights: &MleRef<'_, EF>,
    k: usize,
    force_lagrange: bool,
) -> DensePolynomial<EF> {
    let n_nodes = 2 * ((1usize << k) - 1) + 1;
    let node_sums: Vec<EF> = match (evals, weights) {
        (MleRef::BasePacked(f), MleRef::ExtensionPacked(w)) => skip_poly_node_sums(
            f,
            w,
            k,
            n_nodes,
            force_lagrange,
            |v, c| v * c,
            |v: EFPacking<EF>, c| v * PFPacking::<EF>::from(c),
            sum_packed_lanes::<EF>,
        ),
        (MleRef::ExtensionPacked(f), MleRef::ExtensionPacked(w)) => skip_poly_node_sums(
            f,
            w,
            k,
            n_nodes,
            force_lagrange,
            |v: EFPacking<EF>, c| v * PFPacking::<EF>::from(c),
            |v: EFPacking<EF>, c| v * PFPacking::<EF>::from(c),
            sum_packed_lanes::<EF>,
        ),
        (MleRef::Base(f), MleRef::Extension(w)) => {
            skip_poly_node_sums(f, w, k, n_nodes, force_lagrange, |v, c| v * c, |v, c| v * c, |e| e)
        }
        (MleRef::Extension(f), MleRef::Extension(w)) => {
            skip_poly_node_sums(f, w, k, n_nodes, force_lagrange, |v, c| v * c, |v, c| v * c, |e| e)
        }
        _ => panic!(
            "compute_product_skip_poly: unsupported (evals, weights) variant combination — \
             the WHIR opening only produces BasePacked/ExtensionPacked evals against \
             ExtensionPacked weights (or the unpacked test equivalents); mixed \
             packed/unpacked operands are structurally impossible there"
        ),
    };
    let nodes = skip_all_nodes::<PF<EF>>(k, 2);
    debug_assert_eq!(nodes.len(), n_nodes);
    let pairs: Vec<(PF<EF>, EF)> = nodes.into_iter().zip(node_sums).collect();
    DensePolynomial::lagrange_interpolation(&pairs).expect("distinct integer nodes")
}

/// Rest-positions processed per task: the SoA buffers (`2^k` rows of `CHUNK`
/// values per operand) stay cache-resident while the FD cascades run as
/// CHUNK-wide vectorized row operations (the same lockstep-width pattern as
/// the AIR skip kernel's degree-split cascades).
const SKIP_CHUNK: usize = 16;

/// Per-node sums `Σ_rest f̂(node, rest)·ŵ(node, rest)` for every node of
/// `skip_all_nodes(k, 2)`. One parallel pass; per chunk of rest-positions the
/// window values are gathered block-by-block into row-major SoA buffers
/// (sequential streams), window products accumulated directly, and the
/// extended nodes produced by chunk-wide FD cascades (or Lagrange dots on the
/// reference path).
#[allow(clippy::too_many_arguments)]
fn skip_poly_node_sums<Fe, We, EF>(
    f: &[Fe],
    w: &[We],
    k: usize,
    n_nodes: usize,
    force_lagrange: bool,
    scale_f: impl Fn(Fe, PF<EF>) -> Fe + Sync,
    scale_w: impl Fn(We, PF<EF>) -> We + Sync,
    finish: impl Fn(We) -> EF,
) -> Vec<EF>
where
    Fe: PrimeCharacteristicRing + Copy + Send + Sync,
    We: PrimeCharacteristicRing + Mul<Fe, Output = We> + Copy + Send + Sync,
    EF: ExtensionField<PF<EF>>,
{
    let window = 1usize << k;
    assert_eq!(f.len(), w.len());
    assert!(f.len().is_power_of_two() && f.len() >= window);
    let bp = f.len() >> k; // block length (storage elements); block j = [j·bp, (j+1)·bp)
    let n_ext = n_nodes - window;
    let lag: Option<Vec<Vec<PF<EF>>>> = force_lagrange.then(|| {
        let targets: Vec<PF<EF>> = (window..n_nodes).map(PF::<EF>::from_usize).collect();
        lagrange_coeffs_for_targets::<PF<EF>>(k, &targets)
    });

    let n_chunks = bp.div_ceil(SKIP_CHUNK);
    let acc = parallel::map_reduce_with_state(
        n_chunks,
        || {
            (
                vec![Fe::ZERO; window * SKIP_CHUNK],
                vec![We::ZERO; window * SKIP_CHUNK],
            )
        },
        || vec![We::ZERO; n_nodes],
        |(fbuf, wbuf), acc, ci| {
            let lo = ci * SKIP_CHUNK;
            let c = SKIP_CHUNK.min(bp - lo);
            // Gather: row j = the chunk's slice of block j (sequential copies).
            for j in 0..window {
                fbuf[j * c..(j + 1) * c].copy_from_slice(&f[j * bp + lo..j * bp + lo + c]);
                wbuf[j * c..(j + 1) * c].copy_from_slice(&w[j * bp + lo..j * bp + lo + c]);
            }
            // Window nodes: acc[j] += Σ_p ŵ·f̂ (row dot).
            for j in 0..window {
                let mut s = We::ZERO;
                for p in 0..c {
                    s += wbuf[j * c + p] * fbuf[j * c + p];
                }
                acc[j] += s;
            }
            if let Some(lag) = &lag {
                // Lagrange reference path: extend each row set per target.
                for (t, coeffs) in lag.iter().enumerate() {
                    let mut s = We::ZERO;
                    for p in 0..c {
                        let mut fz = Fe::ZERO;
                        let mut wz = We::ZERO;
                        for j in 0..window {
                            fz += scale_f(fbuf[j * c + p], coeffs[j]);
                            wz += scale_w(wbuf[j * c + p], coeffs[j]);
                        }
                        s += wz * fz;
                    }
                    acc[window + t] += s;
                }
            } else {
                // FD cascades, c-wide rows (vectorized row ops).
                fd_init_in_place(&mut fbuf[..window * c], window, c);
                fd_init_in_place(&mut wbuf[..window * c], window, c);
                let last = (window - 1) * c;
                for t in 0..n_ext {
                    fd_advance(&mut fbuf[..window * c], window, c);
                    fd_advance(&mut wbuf[..window * c], window, c);
                    let mut s = We::ZERO;
                    for p in 0..c {
                        s += wbuf[last + p] * fbuf[last + p];
                    }
                    acc[window + t] += s;
                }
            }
        },
        |mut a, b| {
            for (ai, bi) in a.iter_mut().zip(b) {
                *ai += bi;
            }
            a
        },
    );
    acc.into_iter().map(finish).collect()
}

#[inline]
fn sum_packed_lanes<EF: ExtensionField<PF<EF>>>(e: EFPacking<EF>) -> EF {
    EFPacking::<EF>::to_ext_iter([e]).sum::<EF>()
}

/// `g[rest] = Σ_j weights[j] · x[j·bp + rest]` over one operand, parallel,
/// output extension-packed. `weights` are in window-node (= block) order.
fn fold_blocks_packed<Fe, EF>(x: &[Fe], weights: &[EFPacking<EF>]) -> ArenaVec<EFPacking<EF>>
where
    Fe: Copy + Send + Sync,
    EF: ExtensionField<PF<EF>>,
    EFPacking<EF>: Mul<Fe, Output = EFPacking<EF>>,
{
    let window = weights.len();
    let bp = x.len() / window;
    assert_eq!(bp * window, x.len());
    let mut out = unsafe { ArenaVec::<EFPacking<EF>>::uninitialized(bp) };
    const CHUNK: usize = 1 << 12;
    parallel::par_chunks_mut(&mut out, CHUNK, |chunk_idx, slots| {
        let base = chunk_idx * CHUNK;
        for (i, slot) in slots.iter_mut().enumerate() {
            let p = base + i;
            let mut acc = weights[0] * x[p];
            for (j, &wj) in weights.iter().enumerate().skip(1) {
                acc += wj * x[j * bp + p];
            }
            *slot = acc;
        }
    });
    out
}

/// Scalar (unpacked) variant of [`fold_blocks_packed`].
fn fold_blocks_scalar<Fe, EF>(x: &[Fe], weights: &[EF]) -> ArenaVec<EF>
where
    Fe: Copy + Send + Sync,
    EF: ExtensionField<PF<EF>> + Mul<Fe, Output = EF>,
{
    let window = weights.len();
    let bp = x.len() / window;
    assert_eq!(bp * window, x.len());
    (0..bp)
        .map(|p| {
            let mut acc = weights[0] * x[p];
            for (j, &wj) in weights.iter().enumerate().skip(1) {
                acc += wj * x[j * bp + p];
            }
            acc
        })
        .collect()
}

/// Folds both operands `2^k → 1` with the Lagrange weights `L_j(r0)`
/// (`lagrange_weights_at(k, r0)`, window-node order = block order):
/// `g[rest] = Σ_j L_j(r0) · x[j·block + rest]`. Outputs are extension-typed
/// (`ExtensionPacked` for packed inputs, `Extension` for unpacked).
pub fn fold_product_skip<EF: ExtensionField<PF<EF>>>(
    evals: &MleRef<'_, EF>,
    weights: &MleRef<'_, EF>,
    lagrange_at_r0: &[EF],
) -> (MleOwned<EF>, MleOwned<EF>) {
    assert!(lagrange_at_r0.len().is_power_of_two());
    let packed_weights = || -> Vec<EFPacking<EF>> { lagrange_at_r0.iter().map(|&l| EFPacking::<EF>::from(l)).collect() };
    match (evals, weights) {
        (MleRef::BasePacked(f), MleRef::ExtensionPacked(w)) => {
            let lw = packed_weights();
            (
                MleOwned::ExtensionPacked(fold_blocks_packed::<PFPacking<EF>, EF>(f, &lw)),
                MleOwned::ExtensionPacked(fold_blocks_packed::<EFPacking<EF>, EF>(w, &lw)),
            )
        }
        (MleRef::ExtensionPacked(f), MleRef::ExtensionPacked(w)) => {
            let lw = packed_weights();
            (
                MleOwned::ExtensionPacked(fold_blocks_packed::<EFPacking<EF>, EF>(f, &lw)),
                MleOwned::ExtensionPacked(fold_blocks_packed::<EFPacking<EF>, EF>(w, &lw)),
            )
        }
        (MleRef::Base(f), MleRef::Extension(w)) => (
            MleOwned::Extension(fold_blocks_scalar(f, lagrange_at_r0)),
            MleOwned::Extension(fold_blocks_scalar(w, lagrange_at_r0)),
        ),
        (MleRef::Extension(f), MleRef::Extension(w)) => (
            MleOwned::Extension(fold_blocks_scalar(f, lagrange_at_r0)),
            MleOwned::Extension(fold_blocks_scalar(w, lagrange_at_r0)),
        ),
        _ => panic!(
            "fold_product_skip: unsupported (evals, weights) variant combination — \
             see compute_product_skip_poly"
        ),
    }
}
