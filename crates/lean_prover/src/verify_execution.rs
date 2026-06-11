use std::collections::BTreeMap;

use crate::*;
use backend::{Proof, RawProof, VerifierState, sub_block_lagrange_from_global, verify_univariate_skip_round};
use lean_vm::*;
use sub_protocols::*;

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
    if public_input.len() != PUBLIC_INPUT_LEN {
        return Err(ProofError::InvalidProof);
    }
    let mut verifier_state = VerifierState::<EF, _>::new(proof, *get_poseidon8(), fiat_shamir_domain_sep(bytecode))?;
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
        bytecode.instructions_multilinear(),
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
            extra_data: ExtraDataForBuses::new(&logup_alphas_eq_poly, alpha_slice),
        });

        alpha_offset += n_constraints;
    }

    let max_full_degree = ALL_TABLES.iter().map(|t| t.degree_air() + 1).max().unwrap();

    let n_max = *table_n_vars.values().max().unwrap();

    // Univariate skip round (plan_spec v2 §2.1 steps 1–2) followed by the
    // remaining n_max − k multivariate rounds (step 3).
    let k_skip = AIR_UNIVARIATE_SKIP;
    let n_uni_coeffs = air_skip_n_uni_coeffs(max_full_degree, k_skip);
    let skip = verify_univariate_skip_round(&mut verifier_state, k_skip, n_uni_coeffs, initial_sum)?;
    let Evaluation {
        point: c_post,
        value: claimed_air_final_value,
    } = sumcheck_verify(&mut verifier_state, n_max - k_skip, max_full_degree, skip.target, None)?;

    fn bit_reverse(x: usize, bits: usize) -> usize {
        if bits == 0 { 0 } else { x.reverse_bits() >> (usize::BITS as usize - bits) }
    }

    // AIR final check (§2.2): my_air_final_value = Σ_t κ_t·eq(β^x_t, x_nat_t)·C_t(fold_evals_t).
    let mut my_air_final_value = EF::ZERO;
    let mut fold_evals_all: Vec<Vec<EF>> = Vec::with_capacity(verify_data.len());
    for vd in &verify_data {
        let n_t = table_n_vars[&vd.table];
        let b_t = k_skip.saturating_sub(n_max - n_t);
        let n_cols_total = vd.table.n_columns() + vd.table.n_shift_columns();
        let col_evals = verifier_state.next_extension_scalars_vec(n_cols_total)?;

        macro_rules! eval_constraint {
            ($t:expr) => {{ <_ as SumcheckComputation<EF>>::eval_extension($t, &col_evals, &vd.extra_data) }};
        }
        let constraint_eval = delegate_to_inner!(&vd.table => eval_constraint);

        let bus_point = from_end(gkr_point, n_t);
        let x_nat = natural_ordering_point_for_session(&c_post.0, n_t - b_t);
        let eq_val = MultilinearPoint(bus_point[..n_t - b_t].to_vec()).eq_poly_outside(&MultilinearPoint(x_nat));
        let kappa = if b_t > 0 {
            // κ_t = A_t(r0) = Σ_j eq(β^blk)[j] · L^{(k)}[(2^k − 2^b) + rev_b(j)] (§1.1).
            let eq_blk = eval_eq(&bus_point[n_t - b_t..]);
            eq_blk
                .iter()
                .enumerate()
                .map(|(j, &w)| w * skip.lagrange_on_d[(1 << k_skip) - (1 << b_t) + bit_reverse(j, b_t)])
                .sum::<EF>()
        } else {
            // Late join: L^{(k)}[2^k − 1] · prefix-challenge product (§2.2).
            skip.lagrange_on_d[(1 << k_skip) - 1]
                * c_post.0[..(n_max - k_skip) - n_t].iter().copied().product::<EF>()
        };
        my_air_final_value += kappa * eq_val * constraint_eval;

        if b_t == 0 {
            // Tensor claim already (§2.3): pushed directly, no conversion.
            let natural_ordering_point = natural_ordering_point_for_session(&c_post.0, n_t);
            macro_rules! split {
                ($t:expr) => {{ columns_evals_flat_and_shift($t, &col_evals, &natural_ordering_point) }};
            }
            let claim = delegate_to_inner!(&vd.table => split);
            committed_statements.get_mut(&vd.table).unwrap().push(claim);
        }
        fold_evals_all.push(col_evals);
    }

    if my_air_final_value != claimed_air_final_value {
        return Err(ProofError::InvalidProof);
    }

    // Terminal Lagrange→tensor conversions (§2.1 steps 5–7, §2.3).
    let gamma: EF = verifier_state.sample();
    for (vd, col_evals) in verify_data.iter().zip(&fold_evals_all) {
        let n_t = table_n_vars[&vd.table];
        let b_t = k_skip.saturating_sub(n_max - n_t);
        if b_t == 0 {
            continue;
        }
        let m_t = col_evals.len();
        let gamma_pows: Vec<EF> = gamma.powers().collect_n(m_t);
        let v_t: EF = col_evals.iter().zip(&gamma_pows).map(|(&e, &g)| e * g).sum();
        let Evaluation {
            point: s_t,
            value: v_final,
        } = sumcheck_verify(&mut verifier_state, b_t, 2, v_t, None)?;
        let ghat = verifier_state.next_extension_scalars_vec(m_t)?;

        // w_t(s_t) = Σ_j ℓ_{t,j} · eq_j(reverse(s_t)); ℓ via coherence subset-sums.
        let sub = sub_block_lagrange_from_global(&skip.lagrange_on_d, b_t);
        let s_rev: Vec<EF> = s_t.0.iter().rev().copied().collect();
        let eq_s = eval_eq(&s_rev);
        let w_s: EF = (0..1usize << b_t).map(|j| sub[bit_reverse(j, b_t)] * eq_s[j]).sum();
        let ghat_gamma: EF = ghat.iter().zip(&gamma_pows).map(|(&g, &p)| g * p).sum();
        if w_s * ghat_gamma != v_final {
            return Err(ProofError::InvalidProof);
        }

        // §2.4 point splice: point_t = s_t ++ c_post (round order).
        let mut point_t = s_t.0;
        point_t.extend_from_slice(&c_post.0);
        let natural_ordering_point = natural_ordering_point_for_session(&point_t, n_t);
        macro_rules! split {
            ($t:expr) => {{ columns_evals_flat_and_shift($t, &ghat, &natural_ordering_point) }};
        }
        let claim = delegate_to_inner!(&vd.table => split);
        committed_statements.get_mut(&vd.table).unwrap().push(claim);
    }

    let public_memory_random_point = MultilinearPoint(verifier_state.sample_vec(log2_strict_usize(public_input.len())));
    let public_memory_eval = public_input.evaluate(&public_memory_random_point);

    let previous_statements = vec![
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

    let global_statements_base = stacked_pcs_global_statements(
        parsed_commitment.num_variables,
        log_memory,
        bytecode.log_size(),
        bytecode.ending_pc(),
        previous_statements,
        &table_n_vars,
        &committed_statements,
    );

    // sanity check (not necessary for soundness)
    let num_whir_statements = global_statements_base.iter().map(|s| s.values.len()).sum::<usize>();
    assert_eq!(num_whir_statements, total_whir_statements());

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
