use std::collections::BTreeMap;

use crate::*;
use backend::ArenaVec;
use backend::ansi::Colorize;
use lean_vm::*;
use serde::{Deserialize, Serialize};
use sub_protocols::*;
use tracing::info_span;

fn pack_ef(data: &[EF]) -> Vec<EFPacking<EF>> {
    pack_extension(data)
}

fn br_packed(data: &[EFPacking<EF>], pivot: usize) -> Vec<EFPacking<EF>> {
    let w = packing_log_width::<EF>();
    let chunk_log = pivot - w;
    let chunk_size = 1usize << chunk_log;
    let mut out = data.to_vec();
    for chunk in out.chunks_exact_mut(chunk_size) {
        bit_reverse_permutation(chunk);
    }
    out
}

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
        &bytecode.instructions_multilinear,
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
        let extra_data = ExtraDataForBuses::new(&logup_alphas_eq_poly, alpha_slice);

        let mut flat_and_shift: Vec<&[PF<EF>]> = column_refs[idx].to_vec();
        flat_and_shift.extend(shifted_rows[idx].iter().map(|c| c.as_slice()));
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
    let mut eq_rs: BTreeMap<Table, Vec<EF>> = BTreeMap::new();
    for (idx, table) in ALL_TABLES.iter().enumerate() {
        let col_evals = sessions[idx].final_column_evals();
        prover_state.add_extension_scalars(&col_evals);

        let natural_ordering_point =
            natural_ordering_point_for_session(&sumcheck_air_point.0, traces[table].log_n_rows);
        let eq_r = eval_eq(&natural_ordering_point).to_vec();
        macro_rules! split {
            ($t:expr) => {{ columns_evals_flat_and_shift($t, &col_evals, &natural_ordering_point) }};
        }
        let claim = delegate_to_inner!(table => split);
        committed_statements.get_mut(table).unwrap().push(claim);
        table_col_evals.insert(*table, col_evals);
        eq_rs.insert(*table, eq_r);
    }

    // --- Post-AIR binding: Shout protocol (memory + bytecode) ---
    let log_memory = log2_strict_usize(memory.len());
    let memory_size = memory.len();
    let all_groups: Vec<(Table, Vec<sub_protocols::memory_binding::MemoryBindingGroup>)> = ALL_TABLES
        .iter()
        .filter_map(|t| {
            let g = sub_protocols::memory_binding::memory_binding_groups(t);
            if g.is_empty() { None } else { Some((*t, g)) }
        })
        .collect();
    let n_mem_groups: usize = all_groups.iter().map(|(_, gs)| gs.len()).sum();

    if n_mem_groups > 0 {
        use sub_protocols::shout_binding;

        prover_state.duplex();
        let gamma: EF = prover_state.sample();

        // Step 1: Build joint pushforward P_joint
        let mut p_joint = EF::zero_vec(memory_size);
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
                            p_joint[j] += gamma_power * eq_r[i];
                        }
                    }
                    expected_batched_val += gamma_power * col_evals[group.value_cols[k]];
                    gamma_power *= gamma;
                }
            }
        }

        // Step 2: Compute claimed_sum = <P_joint, memory>
        let claimed_sum: EF = parallel::map_reduce(
            memory_size,
            || EF::ZERO,
            |i| p_joint[i] * EF::from(memory[i]),
            |a, b| a + b,
        );
        debug_assert_eq!(claimed_sum, expected_batched_val);
        prover_state.add_extension_scalar(claimed_sum);

        // Step 3: Shout value sumcheck (degree 2, log_memory rounds)
        prover_state.duplex();
        let mut p_joint_fold = p_joint;
        let mut mem_ef: Vec<EF> = parallel::par_map_collect(memory.len(), |i| EF::from(memory[i]));
        mem_ef.resize(memory_size, EF::ZERO);
        let (_shout_endpoint, s_point) = shout_binding::prove_shout_value_sumcheck(
            &mut prover_state, &mut p_joint_fold, &mut mem_ef, claimed_sum,
        );

        // Step 4: Per-table tensor decomposition
        let half_bits = MEM_HALF_BITS.min(log_memory);
        let s_lo = &s_point[..half_bits];
        let s_hi = &s_point[half_bits..log_memory];

        let eq_table_hi = shout_binding::precompute_eq_table(s_hi, s_hi.len());
        let eq_table_lo = shout_binding::precompute_eq_table(s_lo, s_lo.len());

        struct TableDecomp<'a> {
            eq_row_prime: Vec<EF>,
            addr_col: &'a [F],
            n_rows: usize,
            half_bits: usize,
        }
        let mut mem_decomps: Vec<TableDecomp<'_>> = Vec::new();

        for (table, groups) in &all_groups {
            let trace = &traces[table];
            let eq_r = &eq_rs[table];
            let log_n = trace.log_n_rows;
            let n_rows = 1usize << log_n;

            let mut combined_eq_hi = EF::zero_vec(n_rows);
            let mut combined_eq_lo = EF::zero_vec(n_rows);
            let mut table_gamma_offset = EF::ONE;
            let mut gp_scan = EF::ONE;
            for (t, gs) in &all_groups {
                if *t == *table { table_gamma_offset = gp_scan; break; }
                for g in gs { for _ in 0..g.value_cols.len() { gp_scan *= gamma; } }
            }
            let mut gp = table_gamma_offset;

            let half_mask = (1usize << half_bits) - 1;
            for group in groups {
                let addr_col = &trace.columns[group.addr_col];
                let n_vals = group.value_cols.len();
                let mut gamma_powers = Vec::with_capacity(n_vals);
                let mut g = gp;
                for _ in 0..n_vals {
                    gamma_powers.push(g);
                    g *= gamma;
                }

                let partials: Vec<(EF, EF)> = parallel::par_map_collect(n_rows, |i| {
                    let base_addr = addr_col[i].to_usize();
                    let mut acc_hi = EF::ZERO;
                    let mut acc_lo = EF::ZERO;
                    for k in 0..n_vals {
                        let addr_val = base_addr + k;
                        acc_hi += gamma_powers[k] * eq_table_hi[addr_val >> half_bits];
                        acc_lo += gamma_powers[k] * eq_table_lo[addr_val & half_mask];
                    }
                    (acc_hi, acc_lo)
                });
                parallel::par_for_each_mut2(
                    &mut combined_eq_hi[..n_rows],
                    &mut combined_eq_lo[..n_rows],
                    |i, hi, lo| { *hi += partials[i].0; *lo += partials[i].1; },
                );
                gp = g;
            }

            let mut eq_r_fold = eq_r.clone();
            let mut eq_hi_fold = combined_eq_hi;
            let mut eq_lo_fold = combined_eq_lo;
            let table_contrib: EF = eq_r_fold.iter()
                .zip(eq_hi_fold.iter()).zip(eq_lo_fold.iter())
                .map(|((&e, &h), &l)| e * h * l).sum();
            prover_state.add_extension_scalar(table_contrib);

            let (_hi_eval, _lo_eval, row_point) = shout_binding::prove_tensor_decomp_sumcheck(
                &mut prover_state, &mut eq_r_fold, &mut eq_hi_fold, &mut eq_lo_fold, table_contrib,
            );

            mem_decomps.push(TableDecomp {
                eq_row_prime: eval_eq(&row_point).to_vec(),
                addr_col: &trace.columns[groups[0].addr_col],
                n_rows,
                half_bits,
            });
        }

        // --- Bytecode tensor decomp (before unified GKR) ---
        let exec_table = Table::execution();
        let bc_decomp = if let Some(bc_range) = exec_table.bytecode_bound_columns() {
            let exec_log_n = traces[&exec_table].log_n_rows;
            let r_air_exec = natural_ordering_point_for_session(&sumcheck_air_point.0, exec_log_n);
            let eq_r_exec = eval_eq(&r_air_exec).to_vec();
            let pc_col = &traces[&exec_table].columns[EXEC_COL_PC];
            let bytecode_table_size = 1usize << bytecode.log_size();
            let log_bytecode = bytecode.log_size();
            let bytecode_stride = N_INSTRUCTION_COLUMNS.next_power_of_two();
            let n_bc_cols = bc_range.len();

            prover_state.duplex();
            let gamma_bc: EF = prover_state.sample();

            let p_joint_bc = shout_binding::compute_joint_pushforward(pc_col, bytecode_table_size, &eq_r_exec);
            let batched_bc = sub_protocols::bytecode_binding::compute_batched_bytecode::<EF>(
                &bytecode.instructions_multilinear, bytecode_table_size,
                bytecode_stride, n_bc_cols, &gamma_bc.powers().collect_n(n_bc_cols),
            );

            let bc_len = p_joint_bc.len().min(batched_bc.len());
            let batched_instr_val: EF = parallel::map_reduce(
                bc_len,
                || EF::ZERO,
                |i| p_joint_bc[i] * batched_bc[i],
                |a, b| a + b,
            );
            let col_evals = &table_col_evals[&exec_table];
            let mut expected_bc_val = EF::ZERO;
            let mut gp_bc = EF::ONE;
            for k in 0..n_bc_cols {
                expected_bc_val += gp_bc * col_evals[bc_range.start + k];
                gp_bc *= gamma_bc;
            }
            debug_assert_eq!(batched_instr_val, expected_bc_val);
            prover_state.add_extension_scalar(batched_instr_val);

            prover_state.duplex();
            let mut p_bc_fold = p_joint_bc;
            let mut bc_mem_fold = batched_bc[..bytecode_table_size].to_vec();
            let (_bc_endpoint, bc_s_point) = shout_binding::prove_shout_value_sumcheck(
                &mut prover_state, &mut p_bc_fold, &mut bc_mem_fold, batched_instr_val,
            );

            let half_bits_bc = HALF_BITS_BC.min(log_bytecode);
            let bc_s_lo = &bc_s_point[..half_bits_bc];
            let bc_s_hi = &bc_s_point[half_bits_bc..log_bytecode];

            let pc_hi_col = &traces[&exec_table].columns[EXEC_COL_PC_HI];
            let pc_lo_col = &traces[&exec_table].columns[EXEC_COL_PC_LO];
            let n_exec_rows = 1usize << exec_log_n;

            let eq_table_bc_hi = shout_binding::precompute_eq_table(bc_s_hi, bc_s_hi.len());
            let eq_table_bc_lo = shout_binding::precompute_eq_table(bc_s_lo, bc_s_lo.len());
            let eq_hi_table: Vec<EF> = parallel::par_map_collect(n_exec_rows, |row| {
                eq_table_bc_hi[pc_hi_col[row].to_usize()]
            });
            let eq_lo_table: Vec<EF> = parallel::par_map_collect(n_exec_rows, |row| {
                eq_table_bc_lo[pc_lo_col[row].to_usize()]
            });

            let mut eq_r_bc_fold = eq_r_exec;
            let mut eq_hi_bc_fold = eq_hi_table;
            let mut eq_lo_bc_fold = eq_lo_table;
            let bc_pjoint_eval: EF = eq_r_bc_fold.iter()
                .zip(eq_hi_bc_fold.iter()).zip(eq_lo_bc_fold.iter())
                .map(|((&e, &h), &l)| e * h * l).sum();
            prover_state.add_extension_scalar(bc_pjoint_eval);

            let (_bc_hi_eval, _bc_lo_eval, bc_row_point) = shout_binding::prove_tensor_decomp_sumcheck(
                &mut prover_state, &mut eq_r_bc_fold, &mut eq_hi_bc_fold, &mut eq_lo_bc_fold, bc_pjoint_eval,
            );

            Some((eval_eq(&bc_row_point).to_vec(), pc_hi_col, pc_lo_col, n_exec_rows, half_bits_bc))
        } else {
            None
        };

        // --- Unified binding GKR: ONE alpha, ONE pushforward, ONE c_pf, ONE GKR ---
        prover_state.duplex();
        let alpha_sel: EF = prover_state.sample();
        let one_minus_alpha = EF::ONE - alpha_sel;

        let sqrt_k_mem = 1usize << half_bits;
        let sqrt_k_bc = bc_decomp.as_ref().map_or(0, |d| 1usize << d.4);
        let pf_total = sqrt_k_mem + sqrt_k_bc;

        let mut combined_pf = EF::zero_vec(pf_total);
        for decomp in &mem_decomps {
            let sqrt_k = 1usize << decomp.half_bits;
            for (row, &eq_val) in decomp.eq_row_prime.iter().enumerate() {
                let addr_val = decomp.addr_col[row].to_usize();
                let h = addr_val >> decomp.half_bits;
                let l = addr_val & ((1 << decomp.half_bits) - 1);
                if h < sqrt_k { combined_pf[h] += one_minus_alpha * eq_val; }
                if l < sqrt_k { combined_pf[l] += alpha_sel * eq_val; }
            }
        }
        if let Some((ref bc_eq_row, pc_hi_col, pc_lo_col, n_exec_rows, _hb_bc)) = bc_decomp {
            for row in 0..n_exec_rows {
                let eq_val = bc_eq_row[row];
                let h = pc_hi_col[row].to_usize();
                let l = pc_lo_col[row].to_usize();
                if h < sqrt_k_bc { combined_pf[sqrt_k_mem + h] += one_minus_alpha * eq_val; }
                if l < sqrt_k_bc { combined_pf[sqrt_k_mem + l] += alpha_sel * eq_val; }
            }
        }
        prover_state.add_extension_scalars(&combined_pf);
        let c_pf: EF = prover_state.sample();

        let mut total_trace_rows = 0usize;
        for d in &mem_decomps { total_trace_rows += 2 * d.n_rows; }
        if let Some((_, _, _, n_exec, _)) = &bc_decomp { total_trace_rows += 2 * n_exec; }
        let combined_size = (pf_total + total_trace_rows).next_power_of_two();
        let log_combined = log2_ceil_usize(combined_size);

        let mut combined_nums = EF::zero_vec(combined_size);
        let mut combined_dens = vec![EF::ONE; combined_size];

        // Pushforward entries (memory + bytecode) — parallel fill
        parallel::par_for_each_mut2(
            &mut combined_nums[..pf_total],
            &mut combined_dens[..pf_total],
            |j, num, den| {
                *num = -combined_pf[j];
                *den = c_pf - EF::from_usize(j);
            },
        );

        // Trace entries — parallel fill per section
        let mut offset = pf_total;
        for decomp in &mem_decomps {
            let o_hi = offset;
            let o_lo = offset + decomp.n_rows;
            parallel::par_for_each_mut2(
                &mut combined_nums[o_hi..o_hi + decomp.n_rows],
                &mut combined_dens[o_hi..o_hi + decomp.n_rows],
                |i, num, den| {
                    let addr_val = decomp.addr_col[i].to_usize();
                    *num = decomp.eq_row_prime[i] * one_minus_alpha;
                    *den = c_pf - EF::from_usize(addr_val >> decomp.half_bits);
                },
            );
            parallel::par_for_each_mut2(
                &mut combined_nums[o_lo..o_lo + decomp.n_rows],
                &mut combined_dens[o_lo..o_lo + decomp.n_rows],
                |i, num, den| {
                    let addr_val = decomp.addr_col[i].to_usize();
                    *num = decomp.eq_row_prime[i] * alpha_sel;
                    *den = c_pf - EF::from_usize(addr_val & ((1 << decomp.half_bits) - 1));
                },
            );
            offset += 2 * decomp.n_rows;
        }

        if let Some((ref bc_eq_row, pc_hi_col, pc_lo_col, n_exec_rows, _)) = bc_decomp {
            let o_hi = offset;
            let o_lo = offset + n_exec_rows;
            parallel::par_for_each_mut2(
                &mut combined_nums[o_hi..o_hi + n_exec_rows],
                &mut combined_dens[o_hi..o_hi + n_exec_rows],
                |i, num, den| {
                    *num = bc_eq_row[i] * one_minus_alpha;
                    *den = c_pf - EF::from_usize(sqrt_k_mem + pc_hi_col[i].to_usize());
                },
            );
            parallel::par_for_each_mut2(
                &mut combined_nums[o_lo..o_lo + n_exec_rows],
                &mut combined_dens[o_lo..o_lo + n_exec_rows],
                |i, num, den| {
                    *num = bc_eq_row[i] * alpha_sel;
                    *den = c_pf - EF::from_usize(sqrt_k_mem + pc_lo_col[i].to_usize());
                },
            );
        }

        let pivot = ENDIANNESS_PIVOT_GKR.min(log_combined);
        let comb_nums_packed = pack_ef(&combined_nums);
        let comb_dens_packed = pack_ef(&combined_dens);
        let comb_nums_br = br_packed(&comb_nums_packed, pivot);
        let comb_dens_br = br_packed(&comb_dens_packed, pivot);
        let gkr_result = prove_gkr_quotient_ext(&mut prover_state, &comb_nums_br, &comb_dens_br, pivot);
        debug_assert!(gkr_result.0.is_zero(), "Unified binding GKR: quotient must be zero");

        prover_state.duplex();
    }

    // --- Poseidon GKR (Finding 1) ---
    let poseidon_table = Table::poseidon16();
    {
        let pos_trace = &traces[&poseidon_table];
        let pos_log_n = pos_trace.log_n_rows;
        let pos_n_rows = 1usize << pos_log_n;
        let input_cols: Vec<&[F]> = (0..16)
            .map(|k| pos_trace.columns[POSEIDON_COL_INPUT_START + k].as_slice())
            .collect();

        let (gkr_final_point, gkr_final_input_evals) =
            sub_protocols::poseidon_gkr::prove_poseidon_gkr_precomputed(
                &mut prover_state, &input_cols, pos_n_rows, pos_log_n, None,
            );

        // Finding 1: anchor Poseidon GKR endpoint as WHIR claims
        let gkr_input_claim: BTreeMap<ColIndex, EF> = (0..16)
            .map(|k| (POSEIDON_COL_INPUT_START + k, gkr_final_input_evals[k]))
            .collect();
        committed_statements.get_mut(&poseidon_table).unwrap().push(
            (gkr_final_point, gkr_input_claim, BTreeMap::new())
        );
        prover_state.duplex();
    }

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
