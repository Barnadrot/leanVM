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
            use sub_protocols::shout_binding;

            prover_state.duplex();
            let gamma: EF = prover_state.sample();

            // Step 1: Build joint pushforward P_joint (same as before — scatter eq_r into memory domain)
            let mut p_joint = EF::zero_vec(memory_size);
            let mut expected_batched_val = EF::ZERO;
            let mut gamma_power = EF::ONE;
            let mut total_value_cols = 0usize;
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
                        total_value_cols += 1;
                    }
                }
            }

            // Step 2: Compute claimed_sum = <P_joint, memory>
            // P_joint already has gamma weighting, memory is raw
            let claimed_sum: EF = p_joint.iter().zip(memory.iter())
                .map(|(&p, &m)| p * EF::from(m)).sum();
            debug_assert_eq!(claimed_sum, expected_batched_val,
                "Shout: batched_val must match col_evals");
            prover_state.add_extension_scalar(claimed_sum);

            // Step 3: Shout value sumcheck (degree 1+1=2, log(K) rounds)
            // Evaluand: P_joint[k] * memory[k] summed over Boolean hypercube
            prover_state.duplex();
            let mut p_joint_fold = p_joint;
            let mut mem_ef: Vec<EF> = memory.iter().map(|&m| EF::from(m)).collect();
            mem_ef.resize(memory_size, EF::ZERO);
            let (_shout_endpoint, s_point) = shout_binding::prove_shout_value_sumcheck(
                &mut prover_state, &mut p_joint_fold, &mut mem_ef, claimed_sum,
            );

            // Step 4: Tensor decomposition sumcheck per table
            // Split s into (s_lo, s_hi) for d=2 decomposition
            let half_bits = log_memory / 2;
            let s_lo = &s_point[..half_bits];
            let s_hi = &s_point[half_bits..log_memory];

            // P_joint_tilde(s) = Σ_table table_contrib(s)
            // For each table, prove table_contrib via tensor decomp sumcheck
            prover_state.duplex();
            let c_pf: EF = prover_state.sample(); // challenge for pushforward GKR

            for (table, groups) in &all_groups {
                let trace = &traces[table];
                let eq_r = &eq_rs[table];
                let log_n = trace.log_n_rows;
                let n_rows = 1usize << log_n;

                // Build combined eq_addr tables weighted by gamma across groups
                let mut combined_eq_hi = EF::zero_vec(n_rows);
                let mut combined_eq_lo = EF::zero_vec(n_rows);
                let mut gp = EF::ONE; // track gamma power for this table's groups
                // Find this table's gamma offset
                let mut table_gamma_offset = EF::ONE;
                let mut found = false;
                let mut gp_scan = EF::ONE;
                for (t, gs) in &all_groups {
                    if *t == *table { table_gamma_offset = gp_scan; found = true; break; }
                    for g in gs { for _ in 0..g.value_cols.len() { gp_scan *= gamma; } }
                }
                assert!(found);
                gp = table_gamma_offset;

                for group in groups {
                    let addr_col = &trace.columns[group.addr_col];
                    // Find hi/lo columns for this group's addr
                    let memory_bounds = table.memory_bound_columns();
                    // addr_hi/lo columns depend on the table structure
                    // For extension: COL_IDX_X_HI/LO, for poseidon: POSEIDON_COL_NU_C_HI/LO
                    // We need to map group.addr_col to its hi/lo columns
                    // TODO: this mapping needs to come from the table trait

                    for k in 0..group.value_cols.len() {
                        for i in 0..n_rows {
                            let addr_val = addr_col[i].to_usize() + k;
                            let hi = addr_val >> half_bits;
                            let lo = addr_val & ((1 << half_bits) - 1);
                            let eq_hi = shout_binding::eq_bits_at_point(F::from_usize(hi), s_hi, half_bits);
                            let eq_lo = shout_binding::eq_bits_at_point(F::from_usize(lo), s_lo, half_bits);
                            combined_eq_hi[i] += gp * eq_hi;
                            combined_eq_lo[i] += gp * eq_lo;
                        }
                        gp *= gamma;
                    }
                }

                // Tensor decomp sumcheck for this table
                let mut eq_r_fold = eq_r.clone();
                let mut eq_hi_fold = combined_eq_hi;
                let mut eq_lo_fold = combined_eq_lo;
                let table_contrib: EF = eq_r_fold.iter()
                    .zip(eq_hi_fold.iter())
                    .zip(eq_lo_fold.iter())
                    .map(|((&e, &h), &l)| e * h * l)
                    .sum();
                prover_state.add_extension_scalar(table_contrib);

                let (_hi_eval, _lo_eval, row_point) = shout_binding::prove_tensor_decomp_sumcheck(
                    &mut prover_state, &mut eq_r_fold, &mut eq_hi_fold, &mut eq_lo_fold, table_contrib,
                );

                // d=2 pushforward at tensor decomp endpoint
                let eq_row_prime = eval_eq(&row_point);
                prover_state.duplex();
                let beta: EF = prover_state.sample();
                let addr_col = &trace.columns[groups[0].addr_col];
                let addr_hi_vals: Vec<F> = addr_col.iter().map(|&a| F::from_usize(a.to_usize() >> half_bits)).collect();
                let addr_lo_vals: Vec<F> = addr_col.iter().map(|&a| F::from_usize(a.to_usize() & ((1 << half_bits) - 1))).collect();
                let (p_hi, p_lo, p_batched_pf) = shout_binding::compute_d2_pushforward(
                    &addr_hi_vals, &addr_lo_vals, &eq_row_prime, half_bits, beta,
                );

                // Send pushforward to transcript
                prover_state.add_extension_scalars(&p_batched_pf);

                // GKR for pushforward well-formedness
                let min_gkr_log = N_VARS_TO_SEND_GKR_COEFFS + 1;
                let gkr_log = half_bits.max(min_gkr_log);
                let gkr_size = 1usize << gkr_log;

                // LEFT GKR: Σ_h P_batched[h]/(c_pf - h)
                let mut left_nums = EF::zero_vec(gkr_size);
                for (j, &p) in p_batched_pf.iter().enumerate() { left_nums[j] = -p; }
                let left_dens: Vec<EF> = (0..gkr_size).map(|h| c_pf - EF::from_usize(h)).collect();
                let pivot_left = ENDIANNESS_PIVOT_GKR.min(gkr_log);
                let left_nums_packed = pack_ef(&left_nums);
                let left_dens_packed = pack_ef(&left_dens);
                let left_nums_br = br_packed(&left_nums_packed, pivot_left);
                let left_dens_br = br_packed(&left_dens_packed, pivot_left);
                let left = prove_gkr_quotient_ext(&mut prover_state, &left_nums_br, &left_dens_br, pivot_left);

                // RIGHT GKR: Σ_row eq_row_prime[row]/(c_pf - addr_hi[row]) + beta * Σ_row eq_row_prime[row]/(c_pf - addr_lo[row])
                let right_nums: Vec<EF> = (0..n_rows).map(|i| {
                    let den_hi = c_pf - EF::from(addr_hi_vals[i]);
                    let den_lo = c_pf - EF::from(addr_lo_vals[i]);
                    eq_row_prime[i] * den_lo + beta * eq_row_prime[i] * den_hi
                }).collect();
                let right_dens: Vec<EF> = (0..n_rows).map(|i| {
                    (c_pf - EF::from(addr_hi_vals[i])) * (c_pf - EF::from(addr_lo_vals[i]))
                }).collect();
                let pivot_right = ENDIANNESS_PIVOT_GKR.min(log_n);
                let right_nums_packed = pack_ef(&right_nums);
                let right_dens_packed = pack_ef(&right_dens);
                let right_nums_br = br_packed(&right_nums_packed, pivot_right);
                let right_dens_br = br_packed(&right_dens_packed, pivot_right);
                let _right = prove_gkr_quotient_ext(&mut prover_state, &right_nums_br, &right_dens_br, pivot_right);

                debug_assert!((left.0 + _right.0).is_zero(),
                    "Shout pushforward GKR balance failed for table {}", table.name());
            }

            prover_state.duplex();
            None
        } else {
            None
        };

        // --- V-3: Bytecode-bound instruction columns (Shout protocol) ---
        let exec_table = Table::execution();
        if let Some(bc_range) = exec_table.bytecode_bound_columns() {
            use sub_protocols::shout_binding;

            let exec_log_n = traces[&exec_table].log_n_rows;
            let r_air_exec = natural_ordering_point_for_session(&sumcheck_air_point.0, exec_log_n);
            let eq_r_exec = eval_eq(&r_air_exec);
            let pc_col = &traces[&exec_table].columns[EXEC_COL_PC];
            let bytecode_table_size = 1usize << bytecode.log_size();
            let log_bytecode = bytecode.log_size();
            let bytecode_stride = N_INSTRUCTION_COLUMNS.next_power_of_two();
            let n_bc_cols = bc_range.len();

            prover_state.duplex();
            let gamma_bc: EF = prover_state.sample();

            // Step 1: Joint pushforward for bytecode
            let p_joint_bc = shout_binding::compute_joint_pushforward(pc_col, bytecode_table_size, &eq_r_exec);

            // Step 2: Batch bytecode columns
            let batched_bc = sub_protocols::bytecode_binding::compute_batched_bytecode::<EF>(
                &bytecode.instructions_multilinear, bytecode_table_size,
                bytecode_stride, n_bc_cols, &gamma_bc.powers().collect_n(n_bc_cols),
            );

            let batched_instr_val: EF = p_joint_bc.iter().zip(batched_bc.iter())
                .map(|(&p, &b)| p * b).sum();
            let col_evals = &table_col_evals[&exec_table];
            let mut expected_bc_val = EF::ZERO;
            let mut gp_bc = EF::ONE;
            for k in 0..n_bc_cols {
                expected_bc_val += gp_bc * col_evals[bc_range.start + k];
                gp_bc *= gamma_bc;
            }
            debug_assert_eq!(batched_instr_val, expected_bc_val,
                "Shout bytecode: batched_instr_val must match col_evals");
            prover_state.add_extension_scalar(batched_instr_val);

            // Step 3: Shout value sumcheck for bytecode
            prover_state.duplex();
            let mut p_bc_fold = p_joint_bc;
            let mut bc_mem_fold = batched_bc[..bytecode_table_size].to_vec();
            let (_bc_endpoint, bc_s_point) = shout_binding::prove_shout_value_sumcheck(
                &mut prover_state, &mut p_bc_fold, &mut bc_mem_fold, batched_instr_val,
            );

            // Step 4: Tensor decomposition for bytecode using PC_HI/PC_LO
            let half_bits_bc = log_bytecode / 2;
            let bc_s_lo = &bc_s_point[..half_bits_bc];
            let bc_s_hi = &bc_s_point[half_bits_bc..log_bytecode];

            let pc_hi_col = &traces[&exec_table].columns[EXEC_COL_PC_HI];
            let pc_lo_col = &traces[&exec_table].columns[EXEC_COL_PC_LO];
            let n_exec_rows = 1usize << exec_log_n;

            let (eq_hi_table, eq_lo_table) = shout_binding::build_eq_addr_tables(
                &pc_hi_col[..n_exec_rows], &pc_lo_col[..n_exec_rows],
                bc_s_hi, bc_s_lo, half_bits_bc,
            );

            let mut eq_r_bc_fold = eq_r_exec.clone();
            let mut eq_hi_bc_fold = eq_hi_table;
            let mut eq_lo_bc_fold = eq_lo_table;
            let bc_pjoint_eval: EF = eq_r_bc_fold.iter()
                .zip(eq_hi_bc_fold.iter()).zip(eq_lo_bc_fold.iter())
                .map(|((&e, &h), &l)| e * h * l).sum();
            prover_state.add_extension_scalar(bc_pjoint_eval);

            let (_bc_hi_eval, _bc_lo_eval, bc_row_point) = shout_binding::prove_tensor_decomp_sumcheck(
                &mut prover_state, &mut eq_r_bc_fold, &mut eq_hi_bc_fold, &mut eq_lo_bc_fold, bc_pjoint_eval,
            );

            // Step 5: d=2 pushforward + GKR for bytecode
            let eq_bc_row_prime = eval_eq(&bc_row_point);
            prover_state.duplex();
            let beta_bc: EF = prover_state.sample();
            let (p_bc_hi, p_bc_lo, p_bc_batched) = shout_binding::compute_d2_pushforward(
                &pc_hi_col[..n_exec_rows], &pc_lo_col[..n_exec_rows],
                &eq_bc_row_prime, half_bits_bc, beta_bc,
            );
            prover_state.add_extension_scalars(&p_bc_batched);

            // GKR for bytecode pushforward well-formedness
            let min_gkr_log = N_VARS_TO_SEND_GKR_COEFFS + 1;
            let gkr_log_bc = half_bits_bc.max(min_gkr_log);
            let gkr_size_bc = 1usize << gkr_log_bc;
            let bc_c_pf: EF = prover_state.sample();

            let mut bc_left_nums = EF::zero_vec(gkr_size_bc);
            for (j, &p) in p_bc_batched.iter().enumerate() { bc_left_nums[j] = -p; }
            let bc_left_dens: Vec<EF> = (0..gkr_size_bc).map(|h| bc_c_pf - EF::from_usize(h)).collect();
            let pivot_bc_left = ENDIANNESS_PIVOT_GKR.min(gkr_log_bc);
            let bc_left_nums_packed = pack_ef(&bc_left_nums);
            let bc_left_dens_packed = pack_ef(&bc_left_dens);
            let bc_left_nums_br = br_packed(&bc_left_nums_packed, pivot_bc_left);
            let bc_left_dens_br = br_packed(&bc_left_dens_packed, pivot_bc_left);
            let bc_left = prove_gkr_quotient_ext(&mut prover_state, &bc_left_nums_br, &bc_left_dens_br, pivot_bc_left);

            let bc_right_nums: Vec<EF> = (0..n_exec_rows).map(|i| {
                let den_hi = bc_c_pf - EF::from(pc_hi_col[i]);
                let den_lo = bc_c_pf - EF::from(pc_lo_col[i]);
                eq_bc_row_prime[i] * den_lo + beta_bc * eq_bc_row_prime[i] * den_hi
            }).collect();
            let bc_right_dens: Vec<EF> = (0..n_exec_rows).map(|i| {
                (bc_c_pf - EF::from(pc_hi_col[i])) * (bc_c_pf - EF::from(pc_lo_col[i]))
            }).collect();
            let pivot_bc_right = ENDIANNESS_PIVOT_GKR.min(exec_log_n);
            let bc_right_nums_packed = pack_ef(&bc_right_nums);
            let bc_right_dens_packed = pack_ef(&bc_right_dens);
            let bc_right_nums_br = br_packed(&bc_right_nums_packed, pivot_bc_right);
            let bc_right_dens_br = br_packed(&bc_right_dens_packed, pivot_bc_right);
            let bc_right = prove_gkr_quotient_ext(
                &mut prover_state, &bc_right_nums_br, &bc_right_dens_br, pivot_bc_right,
            );

            debug_assert!((bc_left.0 + bc_right.0).is_zero(),
                "Shout bytecode pushforward GKR balance failed");
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

            // Finding 1: anchor Poseidon GKR endpoint as WHIR claims
            let gkr_input_claim: BTreeMap<ColIndex, EF> = (0..16)
                .map(|k| (POSEIDON_COL_INPUT_START + k, gkr_final_input_evals[k]))
                .collect();
            committed_statements.get_mut(&poseidon_table).unwrap().push(
                (gkr_final_point, gkr_input_claim, BTreeMap::new())
            );
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
