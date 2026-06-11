//! T3' isolation tests (plan_spec v2 §8): brute-force correctness of the
//! univariate-skip prover engine on small heterogeneous systems.
//!
//! Conventions pinned here (review invariant §9.2):
//! (a) coset_sum(v0, k) == Σ_t (direct table sums);
//! (b) horner(v0, r0) == Σ_t κ_t·sum_post_t, with the right side recomputed
//!     INDEPENDENTLY (naive Lagrange fold + eval_extension + explicit eq);
//! (c) the engine's Lagrange fold uses ℓ_j = L^{(b)}_{rev_b(j)}(ρ), and the
//!     folded columns' MLE at any point equals Σ_j ℓ_j·col-MLE(point ++ ⟨j⟩);
//! (d) the g-pass returns col-MLE(x_nat ++ ⟨j⟩) exactly;
//! (e) END-TO-END: engine → post-skip sessions (T2') → factors driver →
//!     final check → γ-RLC conversion sumcheck → w(s)·ĝ check, all verified
//!     against the EXISTING `sumcheck_verify` chain (the §2 wire protocol
//!     minus verify_execution wiring, which lands in T4').

use backend::*;
use lean_vm::{EF, ExtensionOpPrecompile, ExtraDataForBuses, F, Poseidon8Precompile};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sub_protocols::{
    AirSumcheckSession, OuterSumcheckSession, SkipAir, SkipTableInput, compute_shifted_columns,
    fold_columns_at_x_point, natural_ordering_point_for_session, prove_air_univariate_skip,
    prove_batched_air_sumcheck_with_factors,
};

const MAX_AIR_DEGREE: usize = 8; // poseidon8; the wire length driver

fn n_uni_coeffs(k: usize) -> usize {
    (MAX_AIR_DEGREE + 1) * ((1 << k) - 1) + 1
}

fn bit_reverse(x: usize, bits: usize) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS as usize - bits)
    }
}

/// Random table data: `n_flat` random columns of length `2^log_n_rows`, rows
/// beyond `non_padded` constant (copies of the last active row), plus the
/// shifted views of the first `n_shift` columns.
struct TestTable {
    flat: Vec<ArenaVec<F>>,
    shifted: Vec<ArenaVec<F>>,
    eq_factor: Vec<EF>,
    non_padded: usize,
}

impl TestTable {
    fn random(rng: &mut StdRng, log_n_rows: usize, n_flat: usize, n_shift: usize, non_padded: usize) -> Self {
        let n_rows = 1usize << log_n_rows;
        assert!(non_padded <= n_rows && non_padded > 0);
        let flat: Vec<ArenaVec<F>> = (0..n_flat)
            .map(|_| {
                let mut col: Vec<F> = (0..n_rows).map(|_| rng.random()).collect();
                for r in non_padded..n_rows {
                    col[r] = col[non_padded - 1];
                }
                ArenaVec::from_slice(&col)
            })
            .collect();
        let refs: Vec<&[F]> = flat.iter().map(|c| c.as_slice()).collect();
        let shifted = compute_shifted_columns(n_shift, &refs);
        let eq_factor: Vec<EF> = (0..log_n_rows).map(|_| rng.random()).collect();
        Self {
            flat,
            shifted,
            eq_factor,
            non_padded,
        }
    }

    fn columns(&self) -> Vec<&[F]> {
        self.flat
            .iter()
            .map(|c| c.as_slice())
            .chain(self.shifted.iter().map(|c| c.as_slice()))
            .collect()
    }

    fn log_n_rows(&self) -> usize {
        log2_strict_usize(self.flat[0].len())
    }

    /// eq(β, row) via the explicit per-bit product (independent of eval_eq):
    /// coordinate m ↔ row bit n−1−m.
    fn eq_weight(&self, row: usize) -> EF {
        let n = self.log_n_rows();
        let mut w = EF::ONE;
        for (m, &beta) in self.eq_factor.iter().enumerate() {
            let bit = (row >> (n - 1 - m)) & 1;
            w *= if bit == 1 { beta } else { EF::ONE - beta };
        }
        w
    }
}

/// Direct table sum Σ_row eq(β,row)·C(row) via eval_extension.
fn brute_sum<A: Air>(table: &TestTable, air: &A, extra: &A::ExtraData) -> EF
where
    A::ExtraData: AlphaPowers<EF>,
{
    let cols = table.columns();
    let n_rows = table.flat[0].len();
    (0..n_rows)
        .map(|row| {
            let point: Vec<EF> = cols.iter().map(|c| EF::from(c[row])).collect();
            table.eq_weight(row) * <A as SumcheckComputation<EF>>::eval_extension(air, &point, extra)
        })
        .sum()
}

/// col-MLE(x_point ++ ⟨j⟩) where ⟨j⟩ occupies the last b natural coordinates
/// (coordinate for row bit s at natural slot n−1−s).
fn col_mle_at_block(col: &[F], x_point: &[EF], j: usize, b: usize) -> EF {
    let mut point = x_point.to_vec();
    for m in 0..b {
        let bit = (j >> (b - 1 - m)) & 1;
        point.push(if bit == 1 { EF::ONE } else { EF::ZERO });
    }
    col.evaluate(&MultilinearPoint(point))
}

/// Identities (a)–(c) + sum_post/κ on a 3-table heterogeneous system
/// (b = k, b = k−1, and — for k ≥ 4 — a late-join table with b = max(0, k−4)),
/// for k ∈ {3, 4, 5}, with non-pow2 active row counts.
#[test]
fn skip_engine_identities() {
    let mut rng = StdRng::seed_from_u64(11);
    for k in 3..=5usize {
        let n_max = k + 6;
        let pos_air = Poseidon8Precompile::<false>;
        let ext_air = ExtensionOpPrecompile::<false>;
        let alpha: EF = rng.random();
        let pos_extra = ExtraDataForBuses::new(&[], alpha.powers().collect_n(pos_air.n_constraints()));
        let ext_extra = ExtraDataForBuses::new(&[], alpha.powers().collect_n(ext_air.n_constraints()));

        let t0 = TestTable::random(
            &mut rng,
            n_max,
            pos_air.n_columns(),
            pos_air.n_shift_columns(),
            (1 << n_max) - 5,
        );
        let t1 = TestTable::random(
            &mut rng,
            n_max - 1,
            ext_air.n_columns(),
            ext_air.n_shift_columns(),
            (1 << (n_max - 1)) - 3,
        );
        let t2 = TestTable::random(
            &mut rng,
            n_max - 4,
            ext_air.n_columns(),
            ext_air.n_shift_columns(),
            1 << (n_max - 4),
        );

        let sums = [
            brute_sum(&t0, &pos_air, &pos_extra),
            brute_sum(&t1, &ext_air, &ext_extra),
            brute_sum(&t2, &ext_air, &ext_extra),
        ];

        let skip_pos = SkipAir::<EF, _>::new(&pos_air, &pos_extra);
        let skip_ext = SkipAir::<EF, _>::new(&ext_air, &ext_extra);
        let tables = [
            (&t0, &skip_pos as &dyn sub_protocols::SkipComputation<EF>),
            (&t1, &skip_ext),
            (&t2, &skip_ext),
        ];
        let inputs: Vec<SkipTableInput<'_, EF>> = tables
            .iter()
            .zip(&sums)
            .map(|((t, comp), &sum)| SkipTableInput {
                columns: t.columns(),
                eq_factor: t.eq_factor.clone(),
                computation: *comp,
                sum,
                non_padded_n_rows: t.non_padded,
            })
            .collect();

        let mut ps = ProverState::<EF, _>::new(*get_poseidon8(), Default::default());
        let out = prove_air_univariate_skip(&mut ps, &inputs, k, n_uni_coeffs(k));

        // (a) coset-sum identity against the brute-force sums.
        assert_eq!(
            coset_sum(&out.v0_coeffs, k),
            sums.iter().copied().sum::<EF>(),
            "k={k}: coset sum"
        );

        // Per-table independent recomputation of κ, ℓ, sum_post.
        let r0 = out.r0;
        let mut rhs = EF::ZERO;
        for (idx, ((t, _), table_out)) in tables.iter().zip(&out.tables).enumerate() {
            let n = t.log_n_rows();
            let p = n_max - n;
            let b = k.saturating_sub(p);
            assert_eq!(table_out.b, b, "k={k} t={idx}");

            if b == 0 {
                assert_eq!(
                    table_out.kappa,
                    out.lagrange_global[(1 << k) - 1],
                    "k={k} t={idx}: κ (b=0)"
                );
                assert_eq!(table_out.sum_post, sums[idx], "k={k} t={idx}: sum_post (b=0)");
                rhs += table_out.kappa * table_out.sum_post;
                continue;
            }

            // ℓ_j = L^{(b)}_{rev_b(j)}(ρ) — directly, not via subset sums.
            let rho = r0.exp_power_of_2(k - b);
            let lag_b = lagrange_evals_on_subgroup::<F, EF>(rho, b);
            let ell_direct: Vec<EF> = (0..1usize << b).map(|j| lag_b[bit_reverse(j, b)]).collect();
            assert_eq!(table_out.ell, ell_direct, "k={k} t={idx}: ℓ convention");

            // κ via the explicit block-eq product (independent of eval_eq).
            let beta_blk = &t.eq_factor[n - b..];
            let kappa_direct: EF = (0..1usize << b)
                .map(|j| {
                    let mut w = EF::ONE;
                    for (m, &beta) in beta_blk.iter().enumerate() {
                        let bit = (j >> (b - 1 - m)) & 1;
                        w *= if bit == 1 { beta } else { EF::ONE - beta };
                    }
                    w * out.lagrange_global[(1 << k) - (1 << b) + bit_reverse(j, b)]
                })
                .sum();
            assert_eq!(table_out.kappa, kappa_direct, "k={k} t={idx}: κ");

            // Naive Lagrange fold + brute-force Q'(ρ) = Σ_x eq^x(x)·C(folded(x)).
            let cols = t.columns();
            let n_x = 1usize << (n - b);
            let folded_naive: Vec<Vec<EF>> = cols
                .iter()
                .map(|col| {
                    (0..n_x)
                        .map(|x| (0..1usize << b).map(|j| ell_direct[j] * col[(x << b) | j]).sum::<EF>())
                        .collect()
                })
                .collect();
            let comp = inputs[idx].computation;
            let mut q_at_rho = EF::ZERO;
            for x in 0..n_x {
                let mut w = EF::ONE;
                for (m, &beta) in t.eq_factor[..n - b].iter().enumerate() {
                    let bit = (x >> (n - b - 1 - m)) & 1;
                    w *= if bit == 1 { beta } else { EF::ONE - beta };
                }
                let point: Vec<EF> = folded_naive.iter().map(|c| c[x]).collect();
                // eval through the same dyn interface is packed/base only; use
                // the underlying air via eval_extension equivalence: the dyn
                // trait has no extension eval, so recompute via the concrete
                // airs below instead.
                let c_eval = if idx == 0 {
                    <Poseidon8Precompile<false> as SumcheckComputation<EF>>::eval_extension(
                        &pos_air, &point, &pos_extra,
                    )
                } else {
                    <ExtensionOpPrecompile<false> as SumcheckComputation<EF>>::eval_extension(
                        &ext_air, &point, &ext_extra,
                    )
                };
                let _ = comp; // silence unused in this branch
                q_at_rho += w * c_eval;
            }
            assert_eq!(table_out.sum_post, q_at_rho, "k={k} t={idx}: Q'(ρ)");
            rhs += table_out.kappa * q_at_rho;

            // (c) engine fold == naive fold, and the block-MLE identity.
            let folded_engine: Vec<Vec<EF>> = match table_out.folded.as_ref().unwrap() {
                MleGroupOwned::ExtensionPacked(cs) => cs.iter().map(|c| unpack_extension(c)).collect(),
                MleGroupOwned::Extension(cs) => cs.iter().map(|c| c.to_vec()).collect(),
                _ => panic!("unexpected folded storage"),
            };
            assert_eq!(folded_engine, folded_naive, "k={k} t={idx}: folded columns");

            let probe: Vec<EF> = (0..n - b).map(|_| rng.random()).collect();
            for (c, folded_col) in folded_naive.iter().enumerate().take(3) {
                let lhs = folded_col.evaluate(&MultilinearPoint(probe.clone()));
                let rhs_mle: EF = (0..1usize << b)
                    .map(|j| ell_direct[j] * col_mle_at_block(cols[c], &probe, j, b))
                    .sum();
                assert_eq!(lhs, rhs_mle, "k={k} t={idx} col={c}: fold/MLE round-trip");
            }
        }

        // (b) v0(r0) == Σ_t κ_t·sum_post_t with the independent right side.
        assert_eq!(horner(&out.v0_coeffs, r0), rhs, "k={k}: horner handoff");
    }
}

/// (d) The g-pass returns exactly col-MLE(x_nat ++ ⟨j⟩).
#[test]
fn g_pass_matches_direct_mles() {
    let mut rng = StdRng::seed_from_u64(12);
    for (n, b) in [(9usize, 3usize), (8, 4), (7, 2), (6, 1)] {
        let air = ExtensionOpPrecompile::<false>;
        let t = TestTable::random(&mut rng, n, air.n_columns(), air.n_shift_columns(), (1 << n) - 7);
        let cols = t.columns();
        let x_nat: Vec<EF> = (0..n - b).map(|_| rng.random()).collect();
        let g = fold_columns_at_x_point::<EF>(&cols, &x_nat, b);
        assert_eq!(g.len(), cols.len());
        for (c, g_c) in g.iter().enumerate() {
            for (j, &val) in g_c.iter().enumerate() {
                assert_eq!(val, col_mle_at_block(cols[c], &x_nat, j, b), "n={n} b={b} c={c} j={j}");
            }
        }
    }
}

/// (e) End-to-end: skip round + post-skip driver + final check + γ-RLC
/// conversion, verified against the existing `sumcheck_verify` chain. This is
/// the §2 wire protocol (steps 1–7) in miniature; T4' moves it into
/// prove_execution/verify_execution.
#[test]
fn end_to_end_mini_protocol() {
    let mut rng = StdRng::seed_from_u64(13);
    let k = 4usize;
    let n_max = 10usize;
    let n_uni = n_uni_coeffs(k);

    let pos_air = Poseidon8Precompile::<false>;
    let ext_air = ExtensionOpPrecompile::<false>;
    let alpha: EF = rng.random();
    let pos_extra = ExtraDataForBuses::new(&[], alpha.powers().collect_n(pos_air.n_constraints()));
    let ext_extra = ExtraDataForBuses::new(&[], alpha.powers().collect_n(ext_air.n_constraints()));

    // b = 4 (poseidon, n = 10), b = 3 (ext_op, n = 9), b = 0 (ext_op, n = 5).
    let t0 = TestTable::random(
        &mut rng,
        n_max,
        pos_air.n_columns(),
        pos_air.n_shift_columns(),
        (1 << n_max) - 9,
    );
    let t1 = TestTable::random(
        &mut rng,
        9,
        ext_air.n_columns(),
        ext_air.n_shift_columns(),
        (1 << 9) - 2,
    );
    let t2 = TestTable::random(&mut rng, 5, ext_air.n_columns(), ext_air.n_shift_columns(), 1 << 5);
    let sums = [
        brute_sum(&t0, &pos_air, &pos_extra),
        brute_sum(&t1, &ext_air, &ext_extra),
        brute_sum(&t2, &ext_air, &ext_extra),
    ];
    let initial_sum: EF = sums.iter().copied().sum();

    let skip_pos = SkipAir::<EF, _>::new(&pos_air, &pos_extra);
    let skip_ext = SkipAir::<EF, _>::new(&ext_air, &ext_extra);
    let table_refs: [(&TestTable, &dyn sub_protocols::SkipComputation<EF>); 3] =
        [(&t0, &skip_pos), (&t1, &skip_ext), (&t2, &skip_ext)];

    // ---------------- Prover ----------------
    let mut ps = ProverState::<EF, _>::new(*get_poseidon8(), Default::default());

    let inputs: Vec<SkipTableInput<'_, EF>> = table_refs
        .iter()
        .zip(&sums)
        .map(|((t, comp), &sum)| SkipTableInput {
            columns: t.columns(),
            eq_factor: t.eq_factor.clone(),
            computation: *comp,
            sum,
            non_padded_n_rows: t.non_padded,
        })
        .collect();
    let mut skip_out = prove_air_univariate_skip(&mut ps, &inputs, k, n_uni);

    // Post-skip sessions: b>0 via new_post_skip; b=0 via the regular ctor.
    let mut sessions: Vec<Box<dyn OuterSumcheckSession<EF> + '_>> = Vec::new();
    let mut initial_k = Vec::new();
    for (idx, (t, _)) in table_refs.iter().enumerate() {
        let tout = &mut skip_out.tables[idx];
        initial_k.push(tout.kappa);
        if tout.b > 0 {
            let folded = tout.folded.take().unwrap();
            if idx == 0 {
                sessions.push(Box::new(AirSumcheckSession::new_post_skip(
                    folded,
                    tout.eq_factor_post.clone(),
                    tout.sum_post,
                    pos_air,
                    ExtraDataForBuses::new(&[], alpha.powers().collect_n(pos_air.n_constraints())),
                    tout.non_padded_post,
                )));
            } else {
                sessions.push(Box::new(AirSumcheckSession::new_post_skip(
                    folded,
                    tout.eq_factor_post.clone(),
                    tout.sum_post,
                    ext_air,
                    ExtraDataForBuses::new(&[], alpha.powers().collect_n(ext_air.n_constraints())),
                    tout.non_padded_post,
                )));
            }
        } else {
            let packed = MleGroupRef::<EF>::Base(t.columns()).pack();
            sessions.push(Box::new(AirSumcheckSession::new(
                packed,
                t.eq_factor.clone(),
                tout.sum_post,
                ext_air,
                ExtraDataForBuses::new(&[], alpha.powers().collect_n(ext_air.n_constraints())),
                t.non_padded,
            )));
        }
    }
    let c_post = prove_batched_air_sumcheck_with_factors(&mut ps, &mut sessions, initial_k.clone());
    assert_eq!(c_post.0.len(), n_max - k);

    // fold_evals per table (wire step 4).
    let fold_evals: Vec<Vec<EF>> = sessions.iter().map(|s| s.final_column_evals()).collect();
    for evals in &fold_evals {
        ps.add_extension_scalars(evals);
    }

    // γ + conversions (wire steps 5–7).
    let gamma: EF = ps.sample();
    let mut ghats: Vec<Vec<EF>> = Vec::new();
    for ((t, _), tout) in table_refs.iter().zip(&skip_out.tables) {
        if tout.b == 0 {
            ghats.push(Vec::new());
            continue;
        }
        let n = t.log_n_rows();
        let x_nat = natural_ordering_point_for_session(&c_post.0, n - tout.b);
        let g = fold_columns_at_x_point::<EF>(&t.columns(), &x_nat, tout.b);
        let mut g_gamma = vec![EF::ZERO; 1 << tout.b];
        let mut pow = EF::ONE;
        for g_c in &g {
            for (j, slot) in g_gamma.iter_mut().enumerate() {
                *slot += pow * g_c[j];
            }
            pow *= gamma;
        }
        let (s_chals, _final_val) = prove_weighted_block_sumcheck(&mut ps, tout.ell.clone(), g_gamma);
        // ĝ_c = G_c-MLE at the reversed challenges.
        let s_rev: Vec<EF> = s_chals.iter().rev().copied().collect();
        let ghat: Vec<EF> = g
            .iter()
            .map(|g_c| g_c.evaluate(&MultilinearPoint(s_rev.clone())))
            .collect();
        ps.add_extension_scalars(&ghat);
        ghats.push(ghat);
    }

    // ---------------- Verifier ----------------
    let mut vs = VerifierState::<EF, _>::new(ps.into_proof(), *get_poseidon8(), Default::default()).unwrap();

    // Step 1–2: v0, coset-sum, r0, target.
    let v0: Vec<EF> = vs.next_extension_scalars_vec(n_uni).unwrap();
    assert_eq!(coset_sum(&v0, k), initial_sum, "verifier coset-sum check");
    let r0_v: EF = vs.sample();
    assert_eq!(r0_v, skip_out.r0);
    let target = horner(&v0, r0_v);
    let lagrange = lagrange_evals_on_subgroup::<F, EF>(r0_v, k);

    // Step 3: post-skip rounds (existing verifier).
    let max_full_degree = MAX_AIR_DEGREE + 1;
    let Evaluation {
        point: c_post_v,
        value: t_final,
    } = sumcheck_verify(&mut vs, n_max - k, max_full_degree, target, None).unwrap();
    assert_eq!(c_post_v.0, c_post.0);

    // Step 4: fold_evals + final AIR check (§2.2).
    let mut my_final = EF::ZERO;
    let mut fold_evals_v: Vec<Vec<EF>> = Vec::new();
    for (idx, ((t, _), tout)) in table_refs.iter().zip(&skip_out.tables).enumerate() {
        let n = t.log_n_rows();
        let b = tout.b;
        let m_cols = t.columns().len();
        let evals: Vec<EF> = vs.next_extension_scalars_vec(m_cols).unwrap();
        assert_eq!(evals, fold_evals[idx]);

        let x_nat = natural_ordering_point_for_session(&c_post_v.0, n - b);
        let beta_x = MultilinearPoint(t.eq_factor[..n - b].to_vec());
        let eq_val = beta_x.eq_poly_outside(&MultilinearPoint(x_nat.clone()));
        let c_eval = if idx == 0 {
            <Poseidon8Precompile<false> as SumcheckComputation<EF>>::eval_extension(&pos_air, &evals, &pos_extra)
        } else {
            <ExtensionOpPrecompile<false> as SumcheckComputation<EF>>::eval_extension(&ext_air, &evals, &ext_extra)
        };
        let kappa = if b > 0 {
            // κ from public data (lagrange + β^blk), as the real verifier does.
            let beta_blk = &t.eq_factor[n - b..];
            (0..1usize << b)
                .map(|j| {
                    let mut w = EF::ONE;
                    for (m, &beta) in beta_blk.iter().enumerate() {
                        let bit = (j >> (b - 1 - m)) & 1;
                        w *= if bit == 1 { beta } else { EF::ONE - beta };
                    }
                    w * lagrange[(1 << k) - (1 << b) + bit_reverse(j, b)]
                })
                .sum::<EF>()
        } else {
            // Late join: κ = L[2^k−1] · Π of pre-join post-skip challenges.
            let join = (n_max - k) - n;
            lagrange[(1 << k) - 1] * c_post_v.0[..join].iter().copied().product::<EF>()
        };
        assert_eq!(
            kappa,
            initial_k[idx]
                * if b > 0 {
                    EF::ONE
                } else {
                    c_post_v.0[..(n_max - k) - n].iter().copied().product::<EF>()
                }
        );
        my_final += kappa * eq_val * c_eval;
        fold_evals_v.push(evals);
    }
    assert_eq!(my_final, t_final, "final AIR check (§2.2)");

    // Steps 5–7: γ, conversions, w(s)·ĝ checks, PCS-claim equality.
    let gamma_v: EF = vs.sample();
    assert_eq!(gamma_v, gamma);
    for (idx, ((t, _), tout)) in table_refs.iter().zip(&skip_out.tables).enumerate() {
        let b = tout.b;
        if b == 0 {
            continue;
        }
        let n = t.log_n_rows();
        let m_cols = t.columns().len();
        let v_t: EF = fold_evals_v[idx]
            .iter()
            .enumerate()
            .map(|(c, &e)| gamma_v.exp_u64(c as u64) * e)
            .sum();
        let Evaluation {
            point: s_t,
            value: v_final,
        } = sumcheck_verify(&mut vs, b, 2, v_t, None).unwrap();
        let ghat: Vec<EF> = vs.next_extension_scalars_vec(m_cols).unwrap();
        assert_eq!(ghat, ghats[idx]);

        // w_t(s_t) from public ℓ (subset-sum coherence path, as the verifier).
        let sub = sub_block_lagrange_from_global(&lagrange, b);
        let ell: Vec<EF> = (0..1usize << b).map(|j| sub[bit_reverse(j, b)]).collect();
        assert_eq!(ell, tout.ell);
        let s_rev: Vec<EF> = s_t.0.iter().rev().copied().collect();
        let w_at_s = ell.evaluate(&MultilinearPoint(s_rev.clone()));
        let ghat_gamma: EF = ghat
            .iter()
            .enumerate()
            .map(|(c, &g)| gamma_v.exp_u64(c as u64) * g)
            .sum();
        assert_eq!(w_at_s * ghat_gamma, v_final, "conversion terminal check");

        // The ĝ are tensor-point claims on the ORIGINAL columns at the spliced
        // natural point [x_nat ++ reverse(s_t)] (§2.4).
        let x_nat = natural_ordering_point_for_session(&c_post_v.0, n - b);
        let mut natural_point = x_nat.clone();
        natural_point.extend(s_rev.iter().copied());
        for (c, col) in t.columns().iter().enumerate().take(3) {
            assert_eq!(
                ghat[c],
                col.evaluate(&MultilinearPoint(natural_point.clone())),
                "t={idx} c={c}: ĝ vs col-MLE"
            );
        }
    }
}
