use std::collections::BTreeMap;

use crate::*;
use backend::{ArenaVec, prove_weighted_block_sumcheck};
use backend::ansi::Colorize;
use lean_vm::*;
use serde::{Deserialize, Serialize};
use sub_protocols::*;
use tracing::info_span;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProof {
    pub proof: Proof<F>,
    // benchmark / debug purpose
    #[serde(skip, default)]
    pub metadata: Option<ExecutionMetadata>,
}

pub fn prove_execution(
    bytecode: &Bytecode,
    public_input: &[F; PUBLIC_INPUT_LEN],
    witness: &ExecutionWitness,
    whir_config: &WhirConfigBuilder,
    vm_profiler: bool,
) -> Result<ExecutionProof, ProverError> {
    check_rate(whir_config.starting_log_inv_rate).map_err(|_| ProverError::InvalidRate)?;
    let ExecutionTrace {
        traces,
        mut memory, // padded with zeros to next power of two
        metadata,
    } = info_span!("Witness generation").in_scope(|| -> Result<_, ProverError> {
        let execution_result = info_span!("Executing bytecode")
            .in_scope(|| try_execute_bytecode(bytecode, public_input, witness, vm_profiler))?;
        Ok(info_span!("Building execution trace")
            .in_scope(|| get_execution_trace(bytecode, execution_result, &witness.min_table_log_n_rows)))
    })?;

    // Memory must be at least MIN_LOG_MEMORY_SIZE and at least bytecode size
    // (required by the stacked polynomial ordering)
    let min_memory_size = (1 << MIN_LOG_MEMORY_SIZE).max(1 << bytecode.log_size());
    if memory.len() < min_memory_size {
        memory.resize(min_memory_size, F::ZERO);
    }
    let mut prover_state = ProverState::new(*get_poseidon8(), fiat_shamir_domain_sep(bytecode));
    prover_state.observe_scalars(public_input);
    prover_state.add_base_scalars(
        &[
            vec![whir_config.starting_log_inv_rate, log2_strict_usize(memory.len())],
            traces.values().map(|t| t.log_n_rows).collect::<Vec<_>>(),
        ]
        .concat()
        .into_iter()
        .map(F::from_usize)
        .collect::<Vec<_>>(),
    );
    for (table, table_trace) in &traces {
        let log_n_rows = table_trace.log_n_rows;
        assert!(log_n_rows >= MIN_LOG_N_ROWS_PER_TABLE, "missing padding");
        let log_limit = max_log_n_rows_per_table(table);
        if log_n_rows > log_limit {
            return Err(TooBigTableError {
                table_name: table.name(),
                log_n_rows,
                log_limit,
            }
            .into());
        }
    }

    let mut table_log = String::new();
    for (table, trace) in &traces {
        table_log.push_str(&format!(
            "{}: 2^{} * (1 + {:.2}) rows | ",
            table.name(),
            trace.log_n_rows - 1,
            (trace.non_padded_n_rows as f64) / (1 << (trace.log_n_rows - 1)) as f64 - 1.0
        ));
    }
    table_log = table_log.trim_end_matches(" | ").to_string();
    tracing::info!("Trace tables sizes: {}", table_log.magenta());

    // TODO parrallelize
    let mut memory_acc = unsafe { ArenaVec::<F>::zeroed(memory.len()) };
    info_span!("Building memory access count").in_scope(|| -> Result<(), ProverError> {
        for (table, trace) in &traces {
            let buses = table.bus_interactions();
            for group in memory_lookup_groups(&buses) {
                let idx_col = &trace.columns[group.idx_col];
                let n = group.value_cols.len();
                for idx in idx_col {
                    let base = idx.to_usize();
                    let cells = memory_acc.get_mut(base..base + n).ok_or(RunnerError::OutOfMemory)?;
                    for cell in cells {
                        *cell += F::ONE;
                    }
                }
            }
        }
        Ok(())
    })?;

    // // TODO parrallelize
    let mut bytecode_acc = unsafe { ArenaVec::<F>::zeroed(bytecode.padded_size()) };
    info_span!("Building bytecode access count").in_scope(|| -> Result<(), ProverError> {
        for pc in traces[&Table::execution()].columns[EXEC_COL_PC].iter() {
            *bytecode_acc.get_mut(pc.to_usize()).ok_or(RunnerError::PCOutOfBounds)? += F::ONE;
        }
        Ok(())
    })?;

    // 1st Commitment
    let stacked_pcs_witness = stack_polynomials_and_commit(
        &mut prover_state,
        whir_config,
        &memory,
        &memory_acc,
        &bytecode_acc,
        &traces,
    );

    // logup (GKR)
    let logup_c = prover_state.sample();
    prover_state.duplex();
    let logup_alphas = prover_state.sample_vec(LOG_MAX_BUS_WIDTH);
    let logup_alphas_eq_poly = eval_eq(&logup_alphas);

    let logup_statements = prove_generic_logup(
        &mut prover_state,
        logup_c,
        &logup_alphas_eq_poly,
        &memory,
        &memory_acc,
        bytecode.instructions_multilinear(),
        &bytecode_acc,
        &traces,
    );
    let gkr_point = &logup_statements.gkr_point;
    let mut committed_statements: CommittedStatements = Default::default();
    for table in ALL_TABLES {
        let log_n_rows = traces[&table].log_n_rows;
        committed_statements.insert(
            table,
            vec![(
                MultilinearPoint(from_end(gkr_point, log_n_rows).to_vec()),
                logup_statements.columns_values[&table].clone(),
                BTreeMap::new(),
            )],
        );
    }

    let air_alpha = prover_state.sample();
    let air_alpha_powers: Vec<EF> = air_alpha.powers().collect_n(total_air_constraints());

    let tables_log_heights: BTreeMap<Table, VarCount> =
        traces.iter().map(|(table, trace)| (*table, trace.log_n_rows)).collect();

    let column_refs: Vec<Vec<&[F]>> = ALL_TABLES
        .iter()
        .map(|table| {
            traces[table].columns[..table.n_columns()]
                .iter()
                .map(|c| c.as_slice())
                .collect()
        })
        .collect();
    let _span = info_span!("Computing shifted columns for AIR sumcheck").entered();
    let shifted_rows: Vec<Vec<ArenaVec<F>>> = ALL_TABLES
        .iter()
        .zip(&column_refs)
        .map(|(table, cols)| compute_shifted_columns(table.n_shift_columns(), cols))
        .collect();
    std::mem::drop(_span);

    // Per-table data shared by the skip engine, the post-skip sessions and the
    // terminal conversion (plan_spec v2 §3).
    struct AirTableData<'c> {
        flat_and_shift: Vec<&'c [F]>,
        eq_factor: Vec<EF>,
        bus_final_value: EF,
        alpha_slice: Vec<EF>,
        non_padded: usize,
    }
    let mut table_data: Vec<AirTableData<'_>> = Vec::with_capacity(ALL_TABLES.len());
    let mut alpha_offset = 0;
    for (idx, table) in ALL_TABLES.iter().enumerate() {
        let log_n_rows = tables_log_heights[table];
        let n_constraints = table.n_constraints();
        let bus_numerator_value = logup_statements.bus_numerators_values[table];
        let bus_denominator_value = logup_statements.bus_denominators_values[table];
        let signed_numerator = bus_numerator_value
            * match table.bus_interactions()[0].direction {
                BusDirection::Pull => EF::NEG_ONE,
                BusDirection::Push => EF::ONE,
            };
        // Each table consumes a disjoint range of alpha powers; alpha^offset weights the bus
        // numerator (multiplicity), alpha^{offset+1} weights the bus fingerprint, alpha^{offset+2..}
        // weight the remaining AIR constraints.
        let bus_final_value = air_alpha_powers[alpha_offset] * signed_numerator
            + air_alpha_powers[alpha_offset + 1] * (logup_c - bus_denominator_value);

        let mut flat_and_shift: Vec<&[PF<EF>]> = column_refs[idx].to_vec();
        flat_and_shift.extend(shifted_rows[idx].iter().map(|c| c.as_slice()));

        table_data.push(AirTableData {
            flat_and_shift,
            eq_factor: from_end(gkr_point, log_n_rows).to_vec(),
            bus_final_value,
            alpha_slice: air_alpha_powers[alpha_offset..alpha_offset + n_constraints].to_vec(),
            non_padded: traces[table].non_padded_n_rows,
        });
        alpha_offset += n_constraints;
    }

    // Univariate skip round + post-skip rounds + terminal conversions
    // (plan_spec v2 §2.1; wire order is normative).
    let k_skip = AIR_UNIVARIATE_SKIP;
    let max_full_degree = ALL_TABLES.iter().map(|t| t.degree_air() + 1).max().unwrap();
    let n_uni_coeffs = air_skip_n_uni_coeffs(max_full_degree, k_skip);

    /// Owned [`SkipComputation`] adapter: keeps the (Copy) air and its own
    /// `ExtraDataForBuses` instance alive for the engine's borrow-free use.
    struct OwnedSkipAir<EF: ExtensionField<PF<EF>>, A: Air>
    where
        A::ExtraData: AlphaPowers<EF>,
    {
        air: A,
        extra: A::ExtraData,
        _marker: std::marker::PhantomData<EF>,
    }
    impl<EF: ExtensionField<PF<EF>>, A: Air> SkipComputation<EF> for OwnedSkipAir<EF, A>
    where
        A::ExtraData: AlphaPowers<EF>,
    {
        fn degree(&self) -> usize {
            <A as SumcheckComputation<EF>>::degree(&self.air)
        }
        fn eval_packed_base(&self, point: &[PFPacking<EF>]) -> EFPacking<EF> {
            <A as SumcheckComputation<EF>>::eval_packed_base(&self.air, point, &self.extra)
        }
        fn eval_base(&self, point: &[PF<EF>]) -> EF {
            <A as SumcheckComputation<EF>>::eval_base(&self.air, point, &self.extra)
        }
    }

    let skip_airs: Vec<Box<dyn SkipComputation<EF>>> = ALL_TABLES
        .iter()
        .zip(&table_data)
        .map(|(table, td)| {
            macro_rules! make_skip_air {
                ($t:expr) => {{
                    Box::new(OwnedSkipAir {
                        air: *$t,
                        extra: ExtraDataForBuses::new(&logup_alphas_eq_poly, td.alpha_slice.clone()),
                        _marker: std::marker::PhantomData,
                    }) as Box<dyn SkipComputation<EF>>
                }};
            }
            delegate_to_inner!(table => make_skip_air)
        })
        .collect();

    let _air_span = info_span!("batched AIR sumcheck").entered();
    let skip_inputs: Vec<SkipTableInput<'_, EF>> = table_data
        .iter()
        .zip(&skip_airs)
        .map(|(td, comp)| SkipTableInput {
            columns: td.flat_and_shift.clone(),
            eq_factor: td.eq_factor.clone(),
            computation: comp.as_ref(),
            sum: td.bus_final_value,
            non_padded_n_rows: td.non_padded,
        })
        .collect();
    let mut skip_out = prove_air_univariate_skip(&mut prover_state, &skip_inputs, k_skip, n_uni_coeffs);
    drop(skip_inputs);

    let mut sessions: Vec<Box<dyn OuterSumcheckSession<EF> + '_>> = Vec::with_capacity(ALL_TABLES.len());
    let mut kappas = Vec::with_capacity(ALL_TABLES.len());
    for (idx, table) in ALL_TABLES.iter().enumerate() {
        let td = &table_data[idx];
        let out = &mut skip_out.tables[idx];
        kappas.push(out.kappa);
        let extra_data = ExtraDataForBuses::new(&logup_alphas_eq_poly, td.alpha_slice.clone());
        if out.b > 0 {
            let folded = out.folded.take().unwrap();
            let eq_factor_post = out.eq_factor_post.clone();
            let sum_post = out.sum_post;
            let non_padded_post = out.non_padded_post;
            macro_rules! make_post_skip_session {
                ($t:expr) => {{
                    let session = AirSumcheckSession::new_post_skip(
                        folded,
                        eq_factor_post,
                        sum_post,
                        *$t,
                        extra_data,
                        non_padded_post,
                    );
                    Box::new(session) as Box<dyn OuterSumcheckSession<EF> + '_>
                }};
            }
            sessions.push(delegate_to_inner!(table => make_post_skip_session));
        } else {
            let packed = MleGroupRef::<EF>::Base(td.flat_and_shift.clone()).pack();
            let eq_factor = td.eq_factor.clone();
            let bus_final_value = td.bus_final_value;
            let non_padded = td.non_padded;
            macro_rules! make_session {
                ($t:expr) => {{
                    let session =
                        AirSumcheckSession::new(packed, eq_factor, bus_final_value, *$t, extra_data, non_padded);
                    Box::new(session) as Box<dyn OuterSumcheckSession<EF> + '_>
                }};
            }
            sessions.push(delegate_to_inner!(table => make_session));
        }
    }

    let c_post = prove_batched_air_sumcheck_with_factors(&mut prover_state, &mut sessions, kappas);

    // Step 4 of §2.1: per-table folded column evals (flat ++ shift).
    let mut fold_evals_all: Vec<Vec<EF>> = Vec::with_capacity(ALL_TABLES.len());
    for session in &sessions {
        let col_evals = session.final_column_evals();
        prover_state.add_extension_scalars(&col_evals);
        fold_evals_all.push(col_evals);
    }

    // Steps 5–7 of §2.1: γ-RLC + per-table Lagrange→tensor conversion.
    let gamma: EF = prover_state.sample();
    for (idx, table) in ALL_TABLES.iter().enumerate() {
        let out = &skip_out.tables[idx];
        let n_t = tables_log_heights[table];
        let claim = if out.b == 0 {
            // Tensor claim already; same path as the pre-skip protocol.
            let natural_ordering_point = natural_ordering_point_for_session(&c_post.0, n_t);
            macro_rules! split {
                ($t:expr) => {{ columns_evals_flat_and_shift($t, &fold_evals_all[idx], &natural_ordering_point) }};
            }
            delegate_to_inner!(table => split)
        } else {
            let b = out.b;
            let x_nat = natural_ordering_point_for_session(&c_post.0, n_t - b);
            // g-pass over the ORIGINAL columns (plan §3.8).
            let g_cols = fold_columns_at_x_point::<EF>(&table_data[idx].flat_and_shift, &x_nat, b);
            let m_t = g_cols.len();
            let gamma_pows: Vec<EF> = gamma.powers().collect_n(m_t);
            // Invariant §9.2b: fold_evals == Σ_j ℓ_j · G_c(j).
            debug_assert!(
                fold_evals_all[idx]
                    .iter()
                    .zip(&g_cols)
                    .all(|(&fe, g)| fe == g.iter().zip(&out.ell).map(|(&gj, &lj)| gj * lj).sum::<EF>()),
                "Lagrange-fold / g-pass mismatch"
            );
            let mut g_gamma = vec![EF::ZERO; 1 << b];
            for (g, &gp) in g_cols.iter().zip(&gamma_pows) {
                for (slot, &gj) in g_gamma.iter_mut().zip(g) {
                    *slot += gj * gp;
                }
            }
            let (s_t, _w_g_final) =
                prove_weighted_block_sumcheck(&mut prover_state, out.ell.clone(), g_gamma);
            // ĝ_c = G_c folded LSB-first by s_t = col_c-MLE(x_nat ++ reverse(s_t)).
            let ghat: Vec<EF> = g_cols
                .iter()
                .map(|g| {
                    let mut v = g.clone();
                    for &r in &s_t {
                        let half = v.len() / 2;
                        for m in 0..half {
                            v[m] = v[2 * m] + (v[2 * m + 1] - v[2 * m]) * r;
                        }
                        v.truncate(half);
                    }
                    v[0]
                })
                .collect();
            prover_state.add_extension_scalars(&ghat);
            let mut point_t = s_t;
            point_t.extend_from_slice(&c_post.0);
            let natural_ordering_point = natural_ordering_point_for_session(&point_t, n_t);
            macro_rules! split {
                ($t:expr) => {{ columns_evals_flat_and_shift($t, &ghat, &natural_ordering_point) }};
            }
            delegate_to_inner!(table => split)
        };
        committed_statements.get_mut(table).unwrap().push(claim);
    }
    drop(_air_span);

    let public_memory_random_point = MultilinearPoint(prover_state.sample_vec(log2_strict_usize(PUBLIC_INPUT_LEN)));
    let public_memory_eval = (&memory[..PUBLIC_INPUT_LEN]).evaluate(&public_memory_random_point);

    let previous_statements = vec![
        SparseStatement::new(
            stacked_pcs_witness.stacked_n_vars,
            logup_statements.memory_and_acc_point,
            vec![
                SparseValue::new(0, logup_statements.value_memory),
                SparseValue::new(1, logup_statements.value_memory_acc),
            ],
        ),
        SparseStatement::new(
            stacked_pcs_witness.stacked_n_vars,
            public_memory_random_point,
            vec![SparseValue::new(0, public_memory_eval)],
        ),
        SparseStatement::new(
            stacked_pcs_witness.stacked_n_vars,
            logup_statements.bytecode_and_acc_point,
            vec![SparseValue::new(
                (2 * memory.len()) >> bytecode.log_size(),
                logup_statements.value_bytecode_acc,
            )],
        ),
    ];

    let global_statements_base = stacked_pcs_global_statements(
        stacked_pcs_witness.stacked_n_vars,
        log2_strict_usize(memory.len()),
        bytecode.log_size(),
        bytecode.ending_pc(),
        previous_statements,
        &tables_log_heights,
        &committed_statements,
    );

    WhirConfig::new(whir_config, stacked_pcs_witness.global_polynomial.by_ref().n_vars()).prove(
        &mut prover_state,
        global_statements_base,
        stacked_pcs_witness.inner_witness,
        &stacked_pcs_witness.global_polynomial.by_ref(),
    );

    tracing::info!("total pow_grinding time: {} ms", pow_grinding_time().as_millis());
    reset_pow_grinding_time();

    Ok(ExecutionProof {
        proof: prover_state.into_proof(),
        metadata: Some(metadata),
    })
}
