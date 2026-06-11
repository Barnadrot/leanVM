use backend::*;
use tracing::info_span;

// Univariate-skip round prover engine (plan_spec v2 §3).
//
// Replaces the first `k` rounds of the batched AIR sumcheck with ONE
// univariate round over the multiplicative subgroup `D` of order `2^k`
// (values-on-D reading + Lagrange block-fold; the terminal Lagrange→tensor
// conversion lives in `backend::sumcheck::univariate_skip`).
//
// # Conventions (load-bearing; pinned by tests/skip_round_correctness.rs)
//
// * Global block index `i` (k bits, bit r = row bit r) ↔ node `ω_k^{rev_k(i)}`.
// * Table t with `p_t = n_max − n_t` prefix rounds has `b_t = max(0, k − p_t)`
//   block-resident row bits; its block-local row index `j` (low `b_t` row
//   bits) sits at global node `ω_k^{(2^k − 2^{b_t}) + rev_{b_t}(j)}`, i.e. at
//   sub-block node `η_t^{rev_{b_t}(j)}` with `η_t = ω_k^{2^{k−b_t}}`.
// * `eq_factor` (β_t) is in MLE-coordinate order: coordinate `m` ↔ row bit
//   `n_t − 1 − m`; the split is `β^x = β[..n−b]`, `β^blk = β[n−b..]`.
// * `eval_eq` is MSB-first: `eval_eq(β^blk)[j]` is the eq weight of
//   block-local row index `j`; `eval_eq(β^x)[x]` the weight of the remaining
//   index `x` (row = `x·2^b + j`).
// * Fold weights: `ℓ_j = L^{(b)}_{rev_b(j)}(ρ)`, `ρ = r0^{2^{k−b}}`
//   (`sub_block_lagrange_from_global` + bit reversal).

/// Constraint evaluation interface for the engine — object-safe so one engine
/// invocation can span heterogeneous `Air` types (mirrors how the batched
/// driver erases sessions behind `OuterSumcheckSession`). The constraint
/// evaluation itself stays monomorphized inside [`SkipAir`].
pub trait SkipComputation<EF: ExtensionField<PF<EF>>>: Sync {
    fn degree(&self) -> usize;
    fn eval_packed_base(&self, point: &[PFPacking<EF>]) -> EFPacking<EF>;
    fn eval_base(&self, point: &[PF<EF>]) -> EF;
}

/// Adapter from any `Air` (+ its extra data) to [`SkipComputation`].
#[derive(Debug)]
pub struct SkipAir<'a, EF: ExtensionField<PF<EF>>, A: Air>
where
    A::ExtraData: AlphaPowers<EF>,
{
    pub air: &'a A,
    pub extra_data: &'a A::ExtraData,
    _marker: std::marker::PhantomData<EF>,
}

impl<'a, EF: ExtensionField<PF<EF>>, A: Air> SkipAir<'a, EF, A>
where
    A::ExtraData: AlphaPowers<EF>,
{
    pub fn new(air: &'a A, extra_data: &'a A::ExtraData) -> Self {
        Self {
            air,
            extra_data,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, EF: ExtensionField<PF<EF>>, A: Air> SkipComputation<EF> for SkipAir<'a, EF, A>
where
    A::ExtraData: AlphaPowers<EF>,
{
    fn degree(&self) -> usize {
        <A as SumcheckComputation<EF>>::degree(self.air)
    }
    fn eval_packed_base(&self, point: &[PFPacking<EF>]) -> EFPacking<EF> {
        <A as SumcheckComputation<EF>>::eval_packed_base(self.air, point, self.extra_data)
    }
    fn eval_base(&self, point: &[PF<EF>]) -> EF {
        <A as SumcheckComputation<EF>>::eval_base(self.air, point, self.extra_data)
    }
}

/// One table's inputs to the skip round.
pub struct SkipTableInput<'a, EF: ExtensionField<PF<EF>>> {
    /// Raw base columns, NATURAL row layout, flat ++ shift (same order the
    /// regular session consumes), each of length `2^log_n_rows`.
    pub columns: Vec<&'a [PF<EF>]>,
    /// β_t = `from_end(gkr_point, n_t)`, MLE-coordinate order, length
    /// `log_n_rows`.
    pub eq_factor: Vec<EF>,
    pub computation: &'a dyn SkipComputation<EF>,
    /// The table's full claimed sum (`bus_final_value`); used for the b=0
    /// contribution to v0 and the engine's internal consistency checks.
    pub sum: EF,
    pub non_padded_n_rows: usize,
}

impl<'a, EF: ExtensionField<PF<EF>>> SkipTableInput<'a, EF> {
    fn log_n_rows(&self) -> usize {
        log2_strict_usize(self.columns[0].len())
    }
}

impl<'a, EF: ExtensionField<PF<EF>>> std::fmt::Debug for SkipTableInput<'a, EF> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkipTableInput")
            .field("n_columns", &self.columns.len())
            .field("log_n_rows", &self.log_n_rows())
            .field("sum", &self.sum)
            .field("non_padded_n_rows", &self.non_padded_n_rows)
            .finish_non_exhaustive()
    }
}

/// Per-table artifacts after the skip round.
#[derive(Debug)]
pub struct SkipTableOutput<EF: ExtensionField<PF<EF>>> {
    pub b: usize,
    /// `κ_t = A_t(r0)` — the public initial factor for the post-skip driver.
    pub kappa: EF,
    /// `Q'_t(ρ_t)` for `b > 0`; the original claimed sum for `b = 0`.
    pub sum_post: EF,
    /// Lagrange fold weights `ℓ_{t,j}` (block-local index order); empty for
    /// `b = 0`.
    pub ell: Vec<EF>,
    /// The Lagrange-folded columns (`b > 0` only) — `ExtensionPacked` when the
    /// folded size supports it, `Extension` otherwise (both accepted by
    /// `AirSumcheckSession::new_post_skip`).
    pub folded: Option<MleGroupOwned<EF>>,
    /// `β^x_t` for `b > 0`; the full β_t for `b = 0`.
    pub eq_factor_post: Vec<EF>,
    /// `non_padded_n_rows.div_ceil(2^b)` — the folded-domain active length.
    pub non_padded_post: usize,
}

/// Result of the skip round.
#[derive(Debug)]
pub struct SkipOutput<EF: ExtensionField<PF<EF>>> {
    pub r0: EF,
    /// `lagrange_evals_on_subgroup(r0, k)` — reusable by the caller (κ checks,
    /// conversion weights).
    pub lagrange_global: Vec<EF>,
    /// The full wire message (length `n_uni_coeffs`), exposed for tests.
    pub v0_coeffs: Vec<EF>,
    pub tables: Vec<SkipTableOutput<EF>>,
}

fn bit_reverse(x: usize, bits: usize) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS as usize - bits)
    }
}

/// `Q'_t` evaluation-point set: the order-`2^b` subgroup, then
/// `n_cosets` disjoint cosets `g_c·D_b`, `g_c = ω_{b+4}^c` (disjoint for
/// `c ∈ 1..16`; asserted).
fn coset_gens<F: TwoAdicField>(b: usize, n_cosets: usize) -> Vec<F> {
    assert!(n_cosets < 16, "coset budget exceeded (degree too high for the point-set construction)");
    let g = F::two_adic_generator(b + 4);
    let mut out = Vec::with_capacity(n_cosets);
    let mut cur = F::ONE;
    for _ in 0..n_cosets {
        cur *= g;
        out.push(cur);
    }
    out
}

/// All evaluation points (base field) in accumulator order: on-D nodes
/// `η^m`, then per coset `g_c·η^j` (natural `j`).
fn point_set<F: TwoAdicField>(b: usize, gens: &[F]) -> Vec<F> {
    let m = 1usize << b;
    let eta = F::two_adic_generator(b);
    let mut points = Vec::with_capacity(m * (1 + gens.len()));
    let mut w = F::ONE;
    for _ in 0..m {
        points.push(w);
        w *= eta;
    }
    for &g in gens {
        let mut w = g;
        for _ in 0..m {
            points.push(w);
            w *= eta;
        }
    }
    points
}

/// Per-table Q' computation (plan §3.1–§3.3): one packed-base on-D sweep plus
/// off-coset block-LDE sweeps, eq-weighted and aggregated over `x`. Returns
/// the Q' values in `point_set` order.
fn compute_qprime_values<EF: ExtensionField<PF<EF>>>(
    input: &SkipTableInput<'_, EF>,
    b: usize,
    gens: &[PF<EF>],
) -> Vec<EF>
where
    PF<EF>: TwoAdicField,
{
    let n = input.log_n_rows();
    let n_x_vars = n - b;
    let beta_x = &input.eq_factor[..n_x_vars];
    let w_log = packing_log_width::<EF>();

    if n_x_vars >= w_log.max(1) && w_log > 0 {
        compute_qprime_values_packed(input, b, gens, beta_x)
    } else {
        compute_qprime_values_scalar(input, b, gens, beta_x)
    }
}

fn compute_qprime_values_packed<EF: ExtensionField<PF<EF>>>(
    input: &SkipTableInput<'_, EF>,
    b: usize,
    gens: &[PF<EF>],
    beta_x: &[EF],
) -> Vec<EF>
where
    PF<EF>: TwoAdicField,
{
    let w_log = packing_log_width::<EF>();
    let n_nodes = 1usize << b;
    let n_cosets = gens.len();
    let n_points = (1 + n_cosets) * n_nodes;
    let m_cols = input.columns.len();
    let n_xp = 1usize << (beta_x.len() - w_log); // packed x-positions
    let rev: Vec<usize> = (0..n_nodes).map(|m| bit_reverse(m, b)).collect();

    let eq_packed: ArenaVec<EFPacking<EF>> = eval_eq_packed(beta_x);

    // Chunking keeps the per-thread scratch (2 × m_cols × 2^b × CHUNK packed
    // elems) L2-resident: poseidon k=5 (111 cols, b=3) → 2×111×8×8×64B ≈ 0.9 MB.
    const CHUNK: usize = 8;
    let n_chunks = n_xp.div_ceil(CHUNK);
    let comp = input.computation;
    let columns = &input.columns;

    struct Scratch<T> {
        slices: Vec<Vec<T>>,
        coset_out: Vec<Vec<T>>,
        point: Vec<T>,
    }

    let acc = parallel::map_reduce_with_state(
        n_chunks,
        || Scratch {
            slices: vec![vec![PFPacking::<EF>::default(); CHUNK]; m_cols * n_nodes],
            coset_out: vec![Vec::with_capacity(CHUNK); m_cols * n_nodes],
            point: Vec::with_capacity(m_cols),
        },
        || EFPacking::<EF>::zero_vec(n_points),
        |st, acc, chunk_idx| {
            let xp0 = chunk_idx * CHUNK;
            let len = CHUNK.min(n_xp - xp0);

            // Gather: slices[c·2^b + node][u] = packed col values at rows
            // ((xp0+u)·W + lane)·2^b + rev_b(node).
            for (c, col) in columns.iter().enumerate() {
                for (node, &j) in rev.iter().enumerate() {
                    let dst = &mut st.slices[c * n_nodes + node];
                    for (u, slot) in dst.iter_mut().enumerate().take(len) {
                        let x_base = (xp0 + u) << w_log;
                        *slot = PFPacking::<EF>::from_fn(|lane| col[((x_base + lane) << b) | j]);
                    }
                }
            }

            // on-D: acc[node] += C(slices[·][node]) · eq(x).
            for u in 0..len {
                let eq = eq_packed[xp0 + u];
                for (node, acc_node) in acc.iter_mut().enumerate().take(n_nodes) {
                    st.point.clear();
                    st.point.extend((0..m_cols).map(|c| st.slices[c * n_nodes + node][u]));
                    *acc_node += comp.eval_packed_base(&st.point) * eq;
                }
            }

            // Block iDFT per column (evals at η^node → coefficients).
            for c in 0..m_cols {
                block_ifft_arrays::<PF<EF>, PFPacking<EF>>(&mut st.slices[c * n_nodes..(c + 1) * n_nodes], b);
            }

            // Off-coset sweeps.
            for (ci, &g) in gens.iter().enumerate() {
                for c in 0..m_cols {
                    block_coset_evals::<PF<EF>, PFPacking<EF>>(
                        &st.slices[c * n_nodes..(c + 1) * n_nodes],
                        b,
                        g,
                        &mut st.coset_out[c * n_nodes..(c + 1) * n_nodes],
                    );
                }
                for u in 0..len {
                    let eq = eq_packed[xp0 + u];
                    for node in 0..n_nodes {
                        st.point.clear();
                        st.point.extend((0..m_cols).map(|c| st.coset_out[c * n_nodes + node][u]));
                        acc[(ci + 1) * n_nodes + node] += comp.eval_packed_base(&st.point) * eq;
                    }
                }
            }
        },
        |mut a, b| {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
            a
        },
    );

    acc.into_iter()
        .map(|s| EFPacking::<EF>::to_ext_iter([s]).sum::<EF>())
        .collect()
}

fn compute_qprime_values_scalar<EF: ExtensionField<PF<EF>>>(
    input: &SkipTableInput<'_, EF>,
    b: usize,
    gens: &[PF<EF>],
    beta_x: &[EF],
) -> Vec<EF>
where
    PF<EF>: TwoAdicField,
{
    let n_nodes = 1usize << b;
    let n_cosets = gens.len();
    let n_points = (1 + n_cosets) * n_nodes;
    let m_cols = input.columns.len();
    let n_x = 1usize << beta_x.len();
    let rev: Vec<usize> = (0..n_nodes).map(|m| bit_reverse(m, b)).collect();
    let eq_table: ArenaVec<EF> = eval_eq(beta_x);
    let comp = input.computation;

    let mut acc = vec![EF::ZERO; n_points];
    let mut slices: Vec<Vec<PF<EF>>> = vec![vec![PF::<EF>::ZERO; 1]; m_cols * n_nodes];
    let mut coset_out: Vec<Vec<PF<EF>>> = vec![Vec::with_capacity(1); m_cols * n_nodes];
    let mut point: Vec<PF<EF>> = Vec::with_capacity(m_cols);

    for x in 0..n_x {
        let eq = eq_table[x];
        for (c, col) in input.columns.iter().enumerate() {
            for (node, &j) in rev.iter().enumerate() {
                slices[c * n_nodes + node][0] = col[(x << b) | j];
            }
        }
        for node in 0..n_nodes {
            point.clear();
            point.extend((0..m_cols).map(|c| slices[c * n_nodes + node][0]));
            acc[node] += comp.eval_base(&point) * eq;
        }
        for c in 0..m_cols {
            block_ifft_arrays::<PF<EF>, PF<EF>>(&mut slices[c * n_nodes..(c + 1) * n_nodes], b);
        }
        for (ci, &g) in gens.iter().enumerate() {
            for c in 0..m_cols {
                block_coset_evals::<PF<EF>, PF<EF>>(
                    &slices[c * n_nodes..(c + 1) * n_nodes],
                    b,
                    g,
                    &mut coset_out[c * n_nodes..(c + 1) * n_nodes],
                );
            }
            for node in 0..n_nodes {
                point.clear();
                point.extend((0..m_cols).map(|c| coset_out[c * n_nodes + node][0]));
                acc[(ci + 1) * n_nodes + node] += comp.eval_base(&point) * eq;
            }
        }
    }
    acc
}

/// `A_t` coefficients (deg < 2^k) from its public node values (plan §1.1):
/// `A_t(ω^{(2^k−2^b)+rev_b(j)}) = eq(β^blk)[j]` for b > 0;
/// `A_t = L^{(k)}_{2^k−1}` for b = 0.
fn a_t_coeffs<EF: ExtensionField<PF<EF>>>(eq_blk: &[EF], b: usize, k: usize) -> Vec<EF>
where
    PF<EF>: TwoAdicField,
{
    let mut vals: Vec<Vec<EF>> = vec![vec![EF::ZERO]; 1 << k];
    if b == 0 {
        vals[(1 << k) - 1][0] = EF::ONE;
    } else {
        for (j, &w) in eq_blk.iter().enumerate() {
            vals[(1 << k) - (1 << b) + bit_reverse(j, b)][0] = w;
        }
    }
    block_ifft_arrays::<PF<EF>, EF>(&mut vals, k);
    vals.into_iter().map(|v| v[0]).collect()
}

/// Lagrange block-fold (plan §3.5): `folded_c(x) = Σ_j ℓ_j · col_c[x·2^b + j]`.
fn lagrange_fold_columns<EF: ExtensionField<PF<EF>>>(
    columns: &[&[PF<EF>]],
    ell: &[EF],
    b: usize,
) -> MleGroupOwned<EF> {
    let n_x = columns[0].len() >> b;
    let folded: Vec<Vec<EF>> = columns
        .iter()
        .map(|col| {
            let mut out = vec![EF::ZERO; n_x];
            parallel::par_chunks_mut(&mut out, 1 << 12, |chunk_idx, chunk| {
                let x0 = chunk_idx << 12;
                for (u, slot) in chunk.iter_mut().enumerate() {
                    let base = (x0 + u) << b;
                    let mut acc = EF::ZERO;
                    for (j, &l) in ell.iter().enumerate() {
                        acc += l * col[base | j];
                    }
                    *slot = acc;
                }
            });
            out
        })
        .collect();

    if n_x >= packing_width::<EF>() {
        MleGroupOwned::ExtensionPacked(folded.iter().map(|c| pack_extension(c)).collect())
    } else {
        MleGroupOwned::Extension(folded.iter().map(|c| ArenaVec::from_slice(c)).collect())
    }
}

/// The g-pass (plan §3.8): `G_c(j) = Σ_x eq(x_nat, x) · col_c[x·2^b + j]`,
/// one streaming pass over the ORIGINAL columns. `x_nat` is the
/// natural-ordering point over the remaining `n − b` variables (coordinate
/// `m` ↔ folded row bit `n − b − 1 − m`). Returns `out[c][j]`.
pub fn fold_columns_at_x_point<EF: ExtensionField<PF<EF>>>(
    columns: &[&[PF<EF>]],
    x_nat: &[EF],
    b: usize,
) -> Vec<Vec<EF>> {
    let n_nodes = 1usize << b;
    let m_cols = columns.len();
    let n_x = 1usize << x_nat.len();
    assert_eq!(columns[0].len(), n_x << b);
    let eq_table: ArenaVec<EF> = eval_eq(x_nat);

    parallel::map_reduce_with_state(
        n_x,
        || (),
        || vec![EF::ZERO; m_cols * n_nodes],
        |(), acc, x| {
            let eq = eq_table[x];
            for (c, col) in columns.iter().enumerate() {
                let base = x << b;
                let dst = &mut acc[c * n_nodes..(c + 1) * n_nodes];
                for (j, slot) in dst.iter_mut().enumerate() {
                    *slot += eq * col[base | j];
                }
            }
        },
        |mut a, b| {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
            a
        },
    )
    .chunks_exact(n_nodes)
    .map(<[EF]>::to_vec)
    .collect()
}

/// The skip-round prover engine (plan_spec v2 §3 steps 1–5).
///
/// Sends the `n_uni_coeffs` coefficients of `v0` (zero-padded), samples `r0`,
/// and returns everything T4' needs to construct post-skip sessions
/// (`AirSumcheckSession::new_post_skip`) and run
/// `prove_batched_air_sumcheck_with_factors` with `initial_k = κ`.
///
/// `n_uni_coeffs` is the protocol-pinned wire length
/// (`(d_max + 1)·(2^k − 1) + 1`); the engine asserts the actual degree fits.
pub fn prove_air_univariate_skip<EF: ExtensionField<PF<EF>>>(
    prover_state: &mut impl FSProver<EF>,
    tables: &[SkipTableInput<'_, EF>],
    k: usize,
    n_uni_coeffs: usize,
) -> SkipOutput<EF>
where
    PF<EF>: TwoAdicField,
{
    assert!(!tables.is_empty());
    let n_max = tables.iter().map(SkipTableInput::log_n_rows).max().unwrap();

    struct TableArtifacts<EF> {
        b: usize,
        q_coeffs: Vec<EF>, // empty for b = 0
        a_coeffs: Vec<EF>,
        eq_blk: Vec<EF>, // empty for b = 0
    }

    // Per-table Q' + A_t (plan §3.1–§3.4).
    let artifacts: Vec<TableArtifacts<EF>> = tables
        .iter()
        .map(|input| {
            let n = input.log_n_rows();
            let p = n_max - n;
            let b = k.saturating_sub(p);
            if b == 0 {
                return TableArtifacts {
                    b,
                    q_coeffs: Vec::new(),
                    a_coeffs: a_t_coeffs::<EF>(&[], 0, k),
                    eq_blk: Vec::new(),
                };
            }
            let _span = info_span!("skip_qprime", n_vars = n, b = b).entered();
            let d = input.computation.degree();
            let n_nodes = 1usize << b;
            let n_pts_needed = (n_nodes - 1) * d + 1;
            let n_cosets = (n_pts_needed.saturating_sub(n_nodes)).div_ceil(n_nodes);
            let gens = coset_gens::<PF<EF>>(b, n_cosets);

            let q_values = compute_qprime_values(input, b, &gens);
            let points = point_set::<PF<EF>>(b, &gens);
            let pairs: Vec<(PF<EF>, EF)> = points.into_iter().zip(q_values).collect();
            let q_poly = DensePolynomial::lagrange_interpolation(&pairs)
                .expect("distinct interpolation points by construction");
            let mut q_coeffs = q_poly.coeffs;
            // Interpolation over the full point set returns one coefficient
            // per point; everything above the true degree must vanish.
            assert!(
                q_coeffs[n_pts_needed..].iter().all(|&c| c == EF::ZERO),
                "Q' degree exceeds (2^b - 1)·d — node/coset convention broken"
            );
            q_coeffs.resize(n_pts_needed, EF::ZERO);

            let eq_blk: Vec<EF> = eval_eq(&input.eq_factor[n - b..]).to_vec();
            let a_coeffs = a_t_coeffs::<EF>(&eq_blk, b, k);
            TableArtifacts {
                b,
                q_coeffs,
                a_coeffs,
                eq_blk,
            }
        })
        .collect();

    // Assemble v0 = Σ_t A_t(y)·Q'_t(y^{2^{k−b_t}}) in coefficient space (§3.4).
    let mut v0 = vec![EF::ZERO; n_uni_coeffs];
    for (input, art) in tables.iter().zip(&artifacts) {
        if art.b == 0 {
            for (e, &ac) in art.a_coeffs.iter().enumerate() {
                v0[e] += input.sum * ac;
            }
        } else {
            let stride = 1usize << (k - art.b);
            for (eq_, &qc) in art.q_coeffs.iter().enumerate() {
                if qc == EF::ZERO {
                    continue;
                }
                let base = eq_ * stride;
                for (ea, &ac) in art.a_coeffs.iter().enumerate() {
                    v0[base + ea] += qc * ac;
                }
            }
        }
    }

    // Internal consistency (plan §9.5): Σ_{y∈D} v0(y) == Σ_t claimed sums.
    debug_assert_eq!(
        coset_sum(&v0, k),
        tables.iter().map(|t| t.sum).sum::<EF>(),
        "coset-sum identity violated: v0 does not reproduce the claimed sums"
    );

    prover_state.add_extension_scalars(&v0);
    let r0: EF = prover_state.sample();

    let lagrange_global = lagrange_evals_on_subgroup::<PF<EF>, EF>(r0, k);

    // Fold + handoff (§3.5).
    let outputs: Vec<SkipTableOutput<EF>> = tables
        .iter()
        .zip(&artifacts)
        .map(|(input, art)| {
            let n = input.log_n_rows();
            let b = art.b;
            if b == 0 {
                return SkipTableOutput {
                    b,
                    kappa: lagrange_global[(1 << k) - 1],
                    sum_post: input.sum,
                    ell: Vec::new(),
                    folded: None,
                    eq_factor_post: input.eq_factor.clone(),
                    non_padded_post: input.non_padded_n_rows,
                };
            }
            let _span = info_span!("skip_fold", n_vars = n, b = b).entered();
            let sub = sub_block_lagrange_from_global(&lagrange_global, b);
            let ell: Vec<EF> = (0..1usize << b).map(|j| sub[bit_reverse(j, b)]).collect();

            // κ_t = Σ_j eq_blk[j]·L^{(k)}[(2^k − 2^b) + rev_b(j)] (§1.1).
            let kappa: EF = art
                .eq_blk
                .iter()
                .enumerate()
                .map(|(j, &w)| w * lagrange_global[(1 << k) - (1 << b) + bit_reverse(j, b)])
                .sum();
            debug_assert_eq!(kappa, horner(&art.a_coeffs, r0), "A_t closed form vs coefficients");

            let rho = r0.exp_power_of_2(k - b);
            let sum_post = horner(&art.q_coeffs, rho);
            let folded = lagrange_fold_columns(&input.columns, &ell, b);

            SkipTableOutput {
                b,
                kappa,
                sum_post,
                ell,
                folded: Some(folded),
                eq_factor_post: input.eq_factor[..n - b].to_vec(),
                non_padded_post: input.non_padded_n_rows.div_ceil(1 << b),
            }
        })
        .collect();

    // Internal consistency (plan §9.5): v0(r0) == Σ_t κ_t·sum_post_t.
    debug_assert_eq!(
        horner(&v0, r0),
        outputs.iter().map(|o| o.kappa * o.sum_post).sum::<EF>(),
        "post-fold handoff inconsistent with v0(r0)"
    );

    SkipOutput {
        r0,
        lagrange_global,
        v0_coeffs: v0,
        tables: outputs,
    }
}
