use std::collections::BTreeMap;

use crate::*;
use backend::{Proof, RawProof, VerifierState};
use lean_vm::*;
use sub_protocols::*;
use utils::{ToUsize, from_end, get_poseidon16};

#[derive(Debug, Clone)]
pub struct ProofVerificationDetails {
    pub bytecode_evaluation: Evaluation<EF>,
    pub sorted_table_perm: Vec<usize>,
}

/// `bytecode` is trusted to be well-formed here (valid hash, valid instructions, etc)
pub fn verify_execution(
    bytecode: &Bytecode,
    public_input: &[F; PUBLIC_INPUT_LEN],
    proof: Proof<F>,
) -> Result<(ProofVerificationDetails, RawProof<F>), ProofError> {
    if bytecode.log_size() > MAX_BYTECODE_LOG_SIZE {
        return Err(ProofError::TooBigBytecode {
            current_log_size: bytecode.log_size(),
            max_log_size: MAX_BYTECODE_LOG_SIZE,
        });
    }
    let mut verifier_state =
        VerifierState::<EF, _>::new(proof, get_poseidon16().clone(), fiat_shamir_domain_sep(bytecode))?;
    verifier_state.observe_scalars(public_input);
    let dims = verifier_state
        .next_base_scalars_vec(2 + N_TABLES)?
        .into_iter()
        .map(|x| x.to_usize())
        .collect::<Vec<_>>();
    let log_inv_rate = dims[0];
    let log_memory = dims[1];
    let table_n_vars: BTreeMap<Table, VarCount> = (0..N_TABLES).map(|i| (ALL_TABLES[i], dims[i + 2])).collect();
    check_rate(log_inv_rate)?;
    let whir_config = default_whir_config(log_inv_rate);
    for (table, &log_n_rows) in &table_n_vars {
        if log_n_rows < MIN_LOG_N_ROWS_PER_TABLE {
            return Err(ProofError::InvalidProof);
        }
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
    // check memory is bigger than any other table
    if log_memory < (*table_n_vars.values().max().unwrap()).max(bytecode.log_size()) {
        return Err(ProofError::InvalidProof);
    }

    if !(MIN_LOG_MEMORY_SIZE..=MAX_LOG_MEMORY_SIZE).contains(&log_memory) {
        return Err(ProofError::InvalidProof);
    }

    if bytecode.log_size() < MIN_BYTECODE_LOG_SIZE {
        return Err(ProofError::InvalidProof);
    }

    let parsed_commitment = stacked_pcs_parse_commitment(
        &whir_config,
        &mut verifier_state,
        log_memory,
        bytecode.log_size(),
        &table_n_vars,
    )?;

    let logup_c = verifier_state.sample();

    verifier_state.duplex();
    let logup_alphas = verifier_state.sample_vec(LOG_MAX_BUS_WIDTH);
    let logup_alphas_eq_poly = eval_eq(&logup_alphas);

    let logup_statements = verify_generic_logup(
        &mut verifier_state,
        logup_c,
        &logup_alphas,
        &logup_alphas_eq_poly,
        log_memory,
        &bytecode.instructions_multilinear,
        &table_n_vars,
    )?;
    let gkr_point = &logup_statements.gkr_point;

    let mut committed_statements: CommittedStatements = Default::default();
    for table in ALL_TABLES {
        let log_n = table_n_vars[&table];
        committed_statements.insert(
            table,
            vec![(
                MultilinearPoint(from_end(gkr_point, log_n).to_vec()),
                logup_statements.columns_values[&table].clone(),
                BTreeMap::new(),
            )],
        );
    }

    let air_alpha = verifier_state.sample();
    let air_alpha_powers: Vec<EF> = air_alpha.powers().collect_n(total_air_constraints());

    struct TableVerifyData {
        table: Table,
        extra_data: ExtraDataForBuses<EF>,
    }
    let mut verify_data: Vec<TableVerifyData> = Vec::new();
    let mut initial_sum = EF::ZERO;
    let mut alpha_offset = 0;

    for table in ALL_TABLES {
        let n_constraints = table.n_constraints();
        let bus_numerator_value = logup_statements.bus_numerators_values[&table];
        let bus_denominator_value = logup_statements.bus_denominators_values[&table];
        let signed_numerator = bus_numerator_value
            * match table.bus_interactions()[0].direction {
                BusDirection::Pull => EF::NEG_ONE,
                BusDirection::Push => EF::ONE,
            };
        initial_sum += air_alpha_powers[alpha_offset] * signed_numerator
            + air_alpha_powers[alpha_offset + 1] * (logup_c - bus_denominator_value);

        let alpha_slice = air_alpha_powers[alpha_offset..alpha_offset + n_constraints].to_vec();
        verify_data.push(TableVerifyData {
            table,
            extra_data: ExtraDataForBuses::new(logup_alphas_eq_poly.clone(), alpha_slice),
        });

        alpha_offset += n_constraints;
    }

    let max_full_degree = ALL_TABLES.iter().map(|t| t.degree_air() + 1).max().unwrap();

    let n_max = *table_n_vars.values().max().unwrap();
    let Evaluation {
        point: sumcheck_air_point,
        value: claimed_air_final_value,
    } = sumcheck_verify(&mut verifier_state, n_max, max_full_degree, initial_sum, None)?;

    let mut my_air_final_value = EF::ZERO;
    let mut table_col_evals: BTreeMap<Table, Vec<EF>> = BTreeMap::new();
    for vd in &verify_data {
        let n_cols_total = vd.table.n_columns() + vd.table.n_shift_columns();
        let col_evals = verifier_state.next_extension_scalars_vec(n_cols_total)?;

        macro_rules! eval_constraint {
            ($t:expr) => {{ <_ as SumcheckComputation<EF>>::eval_extension($t, &col_evals, &vd.extra_data) }};
        }
        let constraint_eval = delegate_to_inner!(&vd.table => eval_constraint);

        let bus_point = from_end(gkr_point, table_n_vars[&vd.table]);
        let natural_ordering_point = natural_ordering_point_for_session(&sumcheck_air_point.0, table_n_vars[&vd.table]);
        my_air_final_value += back_loaded_table_contribution(
            bus_point,
            &sumcheck_air_point.0,
            &natural_ordering_point,
            constraint_eval,
        );

        macro_rules! split {
            ($t:expr) => {{ columns_evals_flat_and_shift($t, &col_evals, &natural_ordering_point) }};
        }
        let claim = delegate_to_inner!(&vd.table => split);

        committed_statements.get_mut(&vd.table).unwrap().push(claim);
        table_col_evals.insert(vd.table, col_evals);
    }

    if my_air_final_value != claimed_air_final_value {
        return Err(ProofError::InvalidProof);
    }

    // --- Post-AIR-sumcheck binding (V-3 bytecode + V-4 memory) via Shout protocol ---
    let memory_binding_statement = {
        use sub_protocols::memory_binding::*;
        use sub_protocols::shout_binding;
        let n_mem_groups = total_memory_binding_groups();
        let memory_size = 1usize << log_memory;

        // --- V-4: Memory-bound value columns (Shout protocol) ---
        let mem_stmt = if n_mem_groups > 0 {
            // Match prover FS: sample gamma
            verifier_state.duplex();
            let gamma: EF = verifier_state.sample();

            // Receive and check batched_val
            let batched_val = verifier_state.next_extension_scalar()?;
            let mut expected_batched_val = EF::ZERO;
            let mut gamma_power = EF::ONE;
            for table in ALL_TABLES {
                let groups = memory_binding_groups(&table);
                if groups.is_empty() { continue; }
                let col_evals = &table_col_evals[&table];
                for group in &groups {
                    for &val_col in &group.value_cols {
                        expected_batched_val += gamma_power * col_evals[val_col];
                        gamma_power *= gamma;
                    }
                }
            }
            if expected_batched_val != batched_val {
                return Err(ProofError::InvalidProof);
            }

            // Shout value sumcheck verification
            verifier_state.duplex();
            let shout_result = shout_binding::verify_shout_value_sumcheck(
                &mut verifier_state, log_memory, batched_val,
            )?;
            let s_point = shout_result.1;

            // Tensor decomposition per table (store results for unified GKR)
            let half_bits = (MAX_LOG_MEMORY_SIZE / 2).min(log_memory);
            let mut mem_table_n_rows: Vec<usize> = Vec::new();

            for table in ALL_TABLES {
                let groups = memory_binding_groups(&table);
                if groups.is_empty() { continue; }
                let log_n = table_n_vars[&table];

                let table_contrib = verifier_state.next_extension_scalar()?;
                let _td_result = shout_binding::verify_tensor_decomp_sumcheck(
                    &mut verifier_state, log_n, table_contrib,
                )?;
                mem_table_n_rows.push(1usize << log_n);
            }

            // --- V-3: Bytecode tensor decomp (before unified GKR) ---
            let exec_table = Table::execution();
            let bc_info = if let Some(bc_range) = exec_table.bytecode_bound_columns() {
                let log_bytecode_size = bytecode.log_size();
                let n_bc_cols = bc_range.len();

                verifier_state.duplex();
                let gamma_bc: EF = verifier_state.sample();

                let batched_instr_val = verifier_state.next_extension_scalar()?;
                let col_evals = &table_col_evals[&exec_table];
                let mut expected_bc_val = EF::ZERO;
                let mut gp_bc = EF::ONE;
                for k in 0..n_bc_cols {
                    expected_bc_val += gp_bc * col_evals[bc_range.start + k];
                    gp_bc *= gamma_bc;
                }
                if expected_bc_val != batched_instr_val {
                    return Err(ProofError::InvalidProof);
                }

                verifier_state.duplex();
                let _bc_shout = shout_binding::verify_shout_value_sumcheck(
                    &mut verifier_state, log_bytecode_size, batched_instr_val,
                )?;

                let bc_pjoint_eval = verifier_state.next_extension_scalar()?;
                let exec_log_n = table_n_vars[&exec_table];
                let _bc_td = shout_binding::verify_tensor_decomp_sumcheck(
                    &mut verifier_state, exec_log_n, bc_pjoint_eval,
                )?;

                let half_bits_bc = HALF_BITS_BC.min(log_bytecode_size);
                Some((1usize << exec_log_n, half_bits_bc))
            } else {
                None
            };

            // --- Unified binding GKR: ONE alpha, ONE pushforward, ONE c_pf, ONE GKR ---
            verifier_state.duplex();
            let _alpha_sel: EF = verifier_state.sample();

            let sqrt_k_mem = 1usize << half_bits;
            let sqrt_k_bc = bc_info.map_or(0, |(_, hb)| 1usize << hb);
            let pf_total = sqrt_k_mem + sqrt_k_bc;
            let _combined_pf = verifier_state.next_extension_scalars_vec(pf_total)?;
            let _c_pf: EF = verifier_state.sample();

            let mut total_trace_rows: usize = mem_table_n_rows.iter().map(|n| 2 * n).sum();
            if let Some((n_exec, _)) = bc_info { total_trace_rows += 2 * n_exec; }
            let combined_size = (pf_total + total_trace_rows).next_power_of_two();
            let log_combined = log2_ceil_usize(combined_size);
            let gkr_result = verify_gkr_quotient(&mut verifier_state, log_combined)?;
            if !gkr_result.0.is_zero() {
                return Err(ProofError::InvalidProof);
            }

            verifier_state.duplex();
            None
        } else {
            None
        };

        // --- Phase 3: Poseidon GKR (verify deterministic intermediates) ---
        {
            let poseidon_table = Table::poseidon16();
            let pos_log_n = table_n_vars[&poseidon_table];

            let (gkr_point, _gkr_claimed, gkr_input_evals) =
                sub_protocols::poseidon_gkr::verify_poseidon_gkr(
                    &mut verifier_state, pos_log_n,
                )?;

            // Finding 1: anchor Poseidon GKR endpoint as WHIR claims
            let gkr_input_claim: BTreeMap<ColIndex, EF> = (0..16)
                .map(|k| (POSEIDON_COL_INPUT_START + k, gkr_input_evals[k]))
                .collect();
            committed_statements.get_mut(&poseidon_table).unwrap().push(
                (gkr_point, gkr_input_claim, BTreeMap::new())
            );
            verifier_state.duplex();
        }

        mem_stmt
    };

    let public_memory_random_point = MultilinearPoint(verifier_state.sample_vec(log2_strict_usize(public_input.len())));
    let public_memory_eval = public_input.evaluate(&public_memory_random_point);

    let mut previous_statements = vec![
        SparseStatement::new(
            parsed_commitment.num_variables,
            logup_statements.memory_and_acc_point,
            vec![
                SparseValue::new(0, logup_statements.value_memory),
                SparseValue::new(1, logup_statements.value_memory_acc),
            ],
        ),
        SparseStatement::new(
            parsed_commitment.num_variables,
            public_memory_random_point,
            vec![SparseValue::new(0, public_memory_eval)],
        ),
        SparseStatement::new(
            parsed_commitment.num_variables,
            logup_statements.bytecode_and_acc_point,
            vec![SparseValue::new(
                (2 << log_memory) >> bytecode.log_size(),
                logup_statements.value_bytecode_acc,
            )],
        ),
    ];
    if let Some(mem_bind_stmt) = memory_binding_statement {
        previous_statements.push(mem_bind_stmt);
    }

    let global_statements_base = stacked_pcs_global_statements(
        parsed_commitment.num_variables,
        log_memory,
        bytecode.log_size(),
        bytecode.ending_pc,
        previous_statements,
        &table_n_vars,
        &committed_statements,
    );

    // sanity check (not necessary for soundness)
    let num_whir_statements = global_statements_base.iter().map(|s| s.values.len()).sum::<usize>();
    eprintln!("  WHIR: num_whir_statements={} expected={}", num_whir_statements, total_whir_statements());
    let expected = total_whir_statements();
    assert_eq!(num_whir_statements, expected);

    WhirConfig::new(&whir_config, parsed_commitment.num_variables).verify(
        &mut verifier_state,
        &parsed_commitment,
        global_statements_base,
    )?;

    let sorted_table_perm: Vec<usize> = sort_tables_by_height(&table_n_vars)
        .into_iter()
        .map(|(t, _)| t.index())
        .collect();
    verifier_state.check_fully_consumed()?;
    Ok((
        ProofVerificationDetails {
            bytecode_evaluation: logup_statements.bytecode_evaluation.unwrap(),
            sorted_table_perm,
        },
        verifier_state.into_raw_proof(),
    ))
}

fn back_loaded_table_contribution<EF: ExtensionField<PF<EF>>>(
    bus_point: &[EF],
    sumcheck_air_point: &[EF],
    natural_ordering_point: &[EF],
    constraint_eval: EF,
) -> EF {
    let n_t = bus_point.len();
    let n_max = sumcheck_air_point.len();
    let suffix_start = n_max - n_t;
    assert_eq!(natural_ordering_point.len(), n_t);
    let eq_val =
        MultilinearPoint(bus_point.to_vec()).eq_poly_outside(&MultilinearPoint(natural_ordering_point.to_vec()));
    let k_t: EF = sumcheck_air_point[..suffix_start].iter().copied().product();
    k_t * eq_val * constraint_eval
}
