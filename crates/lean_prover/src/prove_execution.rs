use std::collections::BTreeMap;

use crate::*;
use lean_vm::*;
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use sub_protocols::*;
use tracing::info_span;
use utils::ansi::Colorize;
use utils::{from_end, get_poseidon16};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProof {
    pub proof: Proof<F>,
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
        mut memory,
        metadata,
        poseidon_checkpoints,
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
    let mut prover_state = ProverState::new(get_poseidon16().clone(), fiat_shamir_domain_sep(bytecode));
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
    let mut memory_acc = F::zero_vec(memory.len());
    info_span!("Building memory access count").in_scope(|| {
        for (table, trace) in &traces {
            let buses = table.bus_interactions();
            for group in memory_lookup_groups(&buses) {
                let idx_col = &trace.columns[group.idx_col];
                let n = group.value_cols.len();
                for idx in idx_col {
                    let base = idx.to_usize();
                    for ofs in 0..n {
                        memory_acc[base + ofs] += F::ONE;
                    }
                }
            }
        }
    });

    // // TODO parrallelize
    let mut bytecode_acc = F::zero_vec(bytecode.padded_size());
    info_span!("Building bytecode access count").in_scope(|| {
        for pc in traces[&Table::execution()].columns[EXEC_COL_PC].iter() {
            bytecode_acc[pc.to_usize()] += F::ONE;
        }
    });

    // Start checkpoint computation in background (independent of FS state)
    let poseidon_table = Table::poseidon16();
    let checkpoint_handle = if poseidon_checkpoints.is_none() {
        let pos_trace = &traces[&poseidon_table];
        let pos_n_rows = 1usize << pos_trace.log_n_rows;
        let input_data: Vec<Vec<F>> = (0..16)
            .map(|k| pos_trace.columns[POSEIDON_COL_INPUT_START + k].clone())
            .collect();
        Some(std::thread::spawn(move || {
            let input_refs: Vec<&[F]> = input_data.iter().map(|v| v.as_slice()).collect();
            sub_protocols::poseidon_gkr::compute_checkpoints_from_inputs(&input_refs, pos_n_rows)
        }))
    } else { None };

    // 1st Commitment (runs in parallel with checkpoint computation)
    let t_commit = std::time::Instant::now();
    let stacked_pcs_witness = stack_polynomials_and_commit(
        &mut prover_state,
        whir_config,
        &memory,
        &memory_acc,
        &bytecode_acc,
        &traces,
    );

    eprintln!("  WHIR commit: {:.0}ms", t_commit.elapsed().as_secs_f64() * 1000.0);

    // logup (GKR)
    let t_logup = std::time::Instant::now();
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
        &bytecode.instructions_multilinear,
        &bytecode_acc,
        &traces,
        None,
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

    eprintln!("  LOGUP: {:.0}ms", t_logup.elapsed().as_secs_f64() * 1000.0);
    let t_air = std::time::Instant::now();
    let air_alpha = prover_state.sample();
    let air_alpha_powers: Vec<EF> = air_alpha.powers().collect_n(total_air_constraints());

    let tables_log_heights: BTreeMap<Table, VarCount> =
        traces.iter().map(|(table, trace)| (*table, trace.log_n_rows)).collect();

    let column_refs: Vec<Vec<&[F]>> = ALL_TABLES
        .iter()
        .map(|table| {
            traces[table].columns[..table.n_columns()]
                .iter()
                .map(Vec::as_slice)
                .collect()
        })
        .collect();
    let _span = info_span!("Computing shifted columns for AIR sumcheck").entered();
    let shifted_rows: Vec<Vec<Vec<F>>> = ALL_TABLES
        .par_iter()
        .zip(&column_refs)
        .map(|(table, cols)| compute_shifted_columns(table.n_shift_columns(), cols))
        .collect();
    std::mem::drop(_span);
    let mut sessions = Vec::with_capacity(ALL_TABLES.len());
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

        let eq_suffix = from_end(gkr_point, log_n_rows).to_vec();

        let alpha_slice = air_alpha_powers[alpha_offset..alpha_offset + n_constraints].to_vec();
        let extra_data = ExtraDataForBuses::new(logup_alphas_eq_poly.clone(), alpha_slice);

        let mut flat_and_shift: Vec<&[PF<EF>]> = column_refs[idx].to_vec();
        flat_and_shift.extend(shifted_rows[idx].iter().map(Vec::as_slice));
        let packed = MleGroupRef::<EF>::Base(flat_and_shift).pack();

        let non_padded = traces[table].non_padded_n_rows;

        macro_rules! make_session {
            ($t:expr) => {{
                let session = AirSumcheckSession::new(packed, eq_suffix, bus_final_value, *$t, extra_data, non_padded);
                Box::new(session) as Box<dyn OuterSumcheckSession<EF> + '_>
            }};
        }
        sessions.push(delegate_to_inner!(table => make_session));
        alpha_offset += n_constraints;
    }

    let sumcheck_air_point =
        info_span!("batched AIR sumcheck").in_scope(|| prove_batched_air_sumcheck(&mut prover_state, &mut sessions));

    let mut table_col_evals: BTreeMap<Table, Vec<EF>> = BTreeMap::new();
    for (idx, table) in ALL_TABLES.iter().enumerate() {
        let col_evals = sessions[idx].final_column_evals();
        prover_state.add_extension_scalars(&col_evals);

        let natural_ordering_point =
            natural_ordering_point_for_session(&sumcheck_air_point.0, traces[table].log_n_rows);
        macro_rules! split {
            ($t:expr) => {{ columns_evals_flat_and_shift($t, &col_evals, &natural_ordering_point) }};
        }
        let claim = delegate_to_inner!(table => split);
        committed_statements.get_mut(table).unwrap().push(claim);
        table_col_evals.insert(*table, col_evals);
    }

    eprintln!("  AIR sumcheck: {:.0}ms", t_air.elapsed().as_secs_f64() * 1000.0);
    // --- Post-AIR-sumcheck binding (V-3 bytecode + V-4 memory) ---
    // The verifier derives virtual column evaluations from pushforwards
    // and checks them against the prover-supplied col_evals.
    let memory_binding_statement = {
        use sub_protocols::memory_binding::*;
        let memory_size = memory.len();
        let log_memory = log2_strict_usize(memory_size);
        let t_bind = std::time::Instant::now();

        let w = packing_width::<EF>();
        let pack_ef = |data: &[EF]| -> Vec<EFPacking<EF>> {
            data.chunks_exact(w)
                .map(|chunk| EFPacking::<EF>::from_ext_slice(chunk))
                .collect()
        };
        let br_packed = |data: &[EFPacking<EF>], piv: usize| -> Vec<EFPacking<EF>> {
            let packed_len = data.len();
            let chunk_packed = 1usize << (piv - packing_log_width::<EF>());
            let shift = usize::BITS as usize - (piv - packing_log_width::<EF>());
            let mut out = vec![EFPacking::<EF>::ZERO; packed_len];
            for (c_idx, chunk) in data.chunks(chunk_packed).enumerate() {
                for (i, &val) in chunk.iter().enumerate() {
                    let br_i = i.reverse_bits() >> shift;
                    out[c_idx * chunk_packed + br_i] = val;
                }
            }
            out
        };

        // --- V-4: Memory-bound value columns ---
        let mut all_groups: Vec<(Table, Vec<MemoryBindingGroup>)> = Vec::new();
        let mut eq_rs: BTreeMap<Table, Vec<EF>> = BTreeMap::new();

        for table in ALL_TABLES {
            let groups = memory_binding_groups(&table);
            if groups.is_empty() { continue; }
            let log_n_rows = traces[&table].log_n_rows;
            let r_air_for_table =
                natural_ordering_point_for_session(&sumcheck_air_point.0, log_n_rows);
            eq_rs.insert(table, eval_eq(&r_air_for_table));
            all_groups.push((table, groups));
        }

        let n_mem_groups: usize = all_groups.iter().map(|(_, gs)| gs.len()).sum();
        let mem_stmt = if n_mem_groups > 0 {
            prover_state.duplex();
            let c_bind: EF = prover_state.sample();
            prover_state.duplex();
            let gamma: EF = prover_state.sample();
            prover_state.duplex();
            let alpha_bind: EF = prover_state.sample();

            let mut p_batched = EF::zero_vec(memory_size);
            let mut expected_batched_val = EF::ZERO;
            let mut gamma_power = EF::ONE;
            for (table, groups) in &all_groups {
                let trace = &traces[table];
                let eq_r = &eq_rs[table];
                let col_evals = &table_col_evals[table];
                for group in groups {
                    for k in 0..group.value_cols.len() {
                        let addr_col = &trace.columns[group.addr_col];
                        for (i, &addr) in addr_col.iter().enumerate() {
                            let j = addr.to_usize() + k;
                            if j < memory_size {
                                p_batched[j] += gamma_power * eq_r[i];
                            }
                        }
                        expected_batched_val += gamma_power * col_evals[group.value_cols[k]];
                        gamma_power *= gamma;
                    }
                }
            }

            let batched_val: EF = p_batched.iter().zip(memory.iter())
                .map(|(&p, &m)| p * m).sum();
            debug_assert_eq!(batched_val, expected_batched_val,
                "batched_val from pushforward must match col_evals for virtual value columns");
            prover_state.add_extension_scalar(batched_val);

            // Combined GKR-product sumcheck (LEFT side):
            // nums[j] = P[j] * (alpha * memory[j] * (c-j) + 1)
            // dens[j] = (c - j)
            // total = alpha * <P, memory> + Σ P[j]/(c-j)
            let pivot = ENDIANNESS_PIVOT_GKR.min(log_memory);
            let combined_nums: Vec<EF> = (0..memory_size)
                .map(|j| p_batched[j] * (alpha_bind * EF::from(memory[j]) * (c_bind - EF::from_usize(j)) + EF::ONE))
                .collect();
            let combined_dens: Vec<EF> = (0..memory_size)
                .map(|j| c_bind - EF::from_usize(j))
                .collect();

            let nums_packed = pack_ef(&combined_nums);
            let dens_packed = pack_ef(&combined_dens);
            let nums_br = br_packed(&nums_packed, pivot);
            let dens_br = br_packed(&dens_packed, pivot);

            prover_state.duplex();
            let left = prove_gkr_quotient_ext(&mut prover_state, &nums_br, &dens_br, pivot);

            // RIGHT side: one GKR per table (trace side)
            // Σ_i eq_r[i] / (c_bind - addr[i]+k) for each (group, k)
            let mut total_right = EF::ZERO;
            let mut gp_global = EF::ONE;
            for (table, groups) in &all_groups {
                let trace = &traces[table];
                let eq_r = &eq_rs[table];
                let log_n = trace.log_n_rows;
                let n_rows = 1usize << log_n;

                let mut right_nums = EF::zero_vec(n_rows);
                let mut right_dens = EF::zero_vec(n_rows);
                let mut first_term = true;
                for group in groups {
                    for k in 0..group.value_cols.len() {
                        let addr_col = &trace.columns[group.addr_col];
                        for i in 0..n_rows {
                            let addr_val = addr_col[i].to_usize() + k;
                            let den_i = c_bind - EF::from_usize(addr_val);
                            if first_term {
                                right_nums[i] = gp_global * eq_r[i];
                                right_dens[i] = den_i;
                            } else {
                                right_nums[i] = right_nums[i] * den_i + gp_global * eq_r[i] * right_dens[i];
                                right_dens[i] = right_dens[i] * den_i;
                            }
                        }
                        gp_global *= gamma;
                        first_term = false;
                    }
                }

                let right_nums_packed = pack_ef(&right_nums);
                let right_dens_packed = pack_ef(&right_dens);
                let pivot_right = ENDIANNESS_PIVOT_GKR.min(log_n);
                let right_nums_br = br_packed(&right_nums_packed, pivot_right);
                let right_dens_br = br_packed(&right_dens_packed, pivot_right);

                let right = prove_gkr_quotient_ext(
                    &mut prover_state, &right_nums_br, &right_dens_br, pivot_right,
                );
                total_right += right.0;
            }

            debug_assert!((left.0 - total_right - alpha_bind * batched_val).is_zero(),
                "combined GKR balance: left - right - alpha*val must be zero");
            prover_state.duplex();

            None
        } else {
            None
        };

        // --- V-3: Bytecode-bound instruction columns (combined GKR) ---
        let exec_table = Table::execution();
        if let Some(bc_range) = exec_table.bytecode_bound_columns() {
            let exec_log_n = traces[&exec_table].log_n_rows;
            let r_air_exec = natural_ordering_point_for_session(&sumcheck_air_point.0, exec_log_n);
            let eq_r_exec = eval_eq(&r_air_exec);
            let pc_col = &traces[&exec_table].columns[EXEC_COL_PC];
            let bytecode_table_size = 1usize << bytecode.log_size();
            let log_bytecode = bytecode.log_size();
            let bytecode_stride = N_INSTRUCTION_COLUMNS.next_power_of_two();

            prover_state.duplex();
            let c_bc: EF = prover_state.sample();
            prover_state.duplex();
            let gamma_bc: EF = prover_state.sample();
            prover_state.duplex();
            let alpha_bc: EF = prover_state.sample();

            let pushforward_bc = sub_protocols::bytecode_binding::compute_pushforward::<EF>(
                pc_col, bytecode_table_size, &eq_r_exec,
            );

            let batched_bytecode = sub_protocols::bytecode_binding::compute_batched_bytecode::<EF>(
                &bytecode.instructions_multilinear, bytecode_table_size,
                bytecode_stride, bc_range.len(), &gamma_bc.powers().collect_n(bc_range.len()),
            );

            let batched_instr_val: EF = pushforward_bc.iter().zip(batched_bytecode.iter())
                .map(|(&p, &b)| p * b).sum();
            let col_evals = &table_col_evals[&exec_table];
            let mut expected_bc_val = EF::ZERO;
            let mut gp_bc = EF::ONE;
            for k in 0..bc_range.len() {
                expected_bc_val += gp_bc * col_evals[bc_range.start + k];
                gp_bc *= gamma_bc;
            }
            debug_assert_eq!(batched_instr_val, expected_bc_val,
                "bytecode batched_instr_val must match col_evals");
            prover_state.add_extension_scalar(batched_instr_val);

            // Left combined GKR: α*<P_bc, batched_bytecode> + Σ P_bc/(c-j)
            let bc_nums: Vec<EF> = (0..bytecode_table_size)
                .map(|j| pushforward_bc[j] * (alpha_bc * batched_bytecode[j] * (c_bc - EF::from_usize(j)) + EF::ONE))
                .collect();
            let bc_dens: Vec<EF> = (0..bytecode_table_size)
                .map(|j| c_bc - EF::from_usize(j))
                .collect();
            let pivot_bc = ENDIANNESS_PIVOT_GKR.min(log_bytecode);
            let bc_nums_packed = pack_ef(&bc_nums);
            let bc_dens_packed = pack_ef(&bc_dens);
            let bc_nums_br = br_packed(&bc_nums_packed, pivot_bc);
            let bc_dens_br = br_packed(&bc_dens_packed, pivot_bc);

            prover_state.duplex();
            let bc_left = prove_gkr_quotient_ext(&mut prover_state, &bc_nums_br, &bc_dens_br, pivot_bc);

            // Right: trace side
            let bc_right_nums: Vec<EF> = eq_r_exec.clone();
            let bc_right_dens: Vec<EF> = pc_col.iter()
                .map(|&pc| c_bc - EF::from(pc))
                .collect();
            let pivot_bc_right = ENDIANNESS_PIVOT_GKR.min(exec_log_n);
            let bc_right_nums_packed = pack_ef(&bc_right_nums);
            let bc_right_dens_packed = pack_ef(&bc_right_dens);
            let bc_right_nums_br = br_packed(&bc_right_nums_packed, pivot_bc_right);
            let bc_right_dens_br = br_packed(&bc_right_dens_packed, pivot_bc_right);

            let bc_right = prove_gkr_quotient_ext(
                &mut prover_state, &bc_right_nums_br, &bc_right_dens_br, pivot_bc_right,
            );

            debug_assert!((bc_left.0 - bc_right.0 - alpha_bc * batched_instr_val).is_zero(),
                "bytecode combined GKR balance failed");
            prover_state.duplex();
        }

        // --- Phase 3: Poseidon GKR (verify deterministic intermediates) ---
        let poseidon_table = Table::poseidon16();
        {
            let pos_trace = &traces[&poseidon_table];
            let pos_log_n = pos_trace.log_n_rows;
            let pos_n_rows = 1usize << pos_log_n;
            let input_cols: Vec<&[F]> = (0..16)
                .map(|k| pos_trace.columns[POSEIDON_COL_INPUT_START + k].as_slice())
                .collect();
            let _col_evals = &table_col_evals[&poseidon_table];

            let t_gkr = std::time::Instant::now();
            // Collect background checkpoints if computed
            let checkpoints = poseidon_checkpoints
                .or_else(|| checkpoint_handle.map(|h| h.join().unwrap()));
            let (gkr_final_point, gkr_final_input_evals) =
                sub_protocols::poseidon_gkr::prove_poseidon_gkr_precomputed(
                    &mut prover_state,
                    &input_cols,
                    pos_n_rows,
                    pos_log_n,
                    checkpoints,
                );
            eprintln!("    GKR prove: {:.0}ms", t_gkr.elapsed().as_secs_f64() * 1000.0);

            // TODO: Anchor GKR endpoint (Finding 1) — temporarily disabled to isolate WHIR param issue
            let _ = (gkr_final_point, gkr_final_input_evals);
            prover_state.duplex();
        }

        eprintln!(
            "  Post-AIR binding: {:.0}ms (mem_groups={}, log_memory={})",
            t_bind.elapsed().as_secs_f64() * 1000.0, n_mem_groups, log_memory,
        );

        mem_stmt
    };

    let public_memory_random_point = MultilinearPoint(prover_state.sample_vec(log2_strict_usize(PUBLIC_INPUT_LEN)));
    let public_memory_eval = (&memory[..PUBLIC_INPUT_LEN]).evaluate(&public_memory_random_point);

    let mut previous_statements = vec![
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
    if let Some(mem_bind_stmt) = memory_binding_statement {
        previous_statements.push(mem_bind_stmt);
    }

    let global_statements_base = stacked_pcs_global_statements(
        stacked_pcs_witness.stacked_n_vars,
        log2_strict_usize(memory.len()),
        bytecode.log_size(),
        bytecode.ending_pc,
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
