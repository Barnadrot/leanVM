use crate::{EF, ExecutionTable, ExtraDataForBuses, LOGUP_MEMORY_DOMAINSEP, eval_bus_data_only, eval_bus_virtual};
use backend::*;

pub const N_RUNTIME_COLUMNS: usize = 5;
pub const N_INSTRUCTION_COLUMNS: usize = 10;
pub const N_TOTAL_EXECUTION_COLUMNS: usize = N_INSTRUCTION_COLUMNS + N_RUNTIME_COLUMNS;

// Committed columns (IMPORTANT: they must be the first columns)
// h9-A (iter 5): ADDR_A/B/C are no longer committed — they are exact low-degree
// polynomials of the columns below (the closed forms trace_gen.rs computes), so they
// live as temporary columns and the memory-bus claims are proven inside the batched
// AIR sumcheck (deferred-claim buses, plan_spec §3.A). Committed: 20 → 17 columns.
// h9-B (iter 5): the five addressing-mode booleans (FLAG_A/B/C, FLAG_C_FP, FLAG_AB_FP)
// merge into three ternary mode columns M_A/M_B/M_C ∈ {0,1,2} (0 = memory,
// 1 = immediate/constant, 2 = fp-relative; M_A = 2 ⟺ M_B = 2, the old flag_ab_fp).
// The boolean flags become degree-2 Lagrange decoders of the mode columns (the
// aux_1 flag_add/flag_deref precedent). Domain {0,1,2} is enforced for free by the
// bytecode lookup against the statically-validated public bytecode
// (Bytecode::validate_modes). Committed: 17 → 15 columns; exec degree_air 5 → 7.
pub const EXEC_COL_PC: usize = 0;
pub const EXEC_COL_FP: usize = 1;
pub const EXEC_COL_VALUE_A: usize = 2;
pub const EXEC_COL_VALUE_B: usize = 3;
pub const EXEC_COL_VALUE_C: usize = 4;

// Decoded instruction columns
pub const EXEC_COL_OPERAND_A: usize = 5;
pub const EXEC_COL_OPERAND_B: usize = 6;
pub const EXEC_COL_OPERAND_C: usize = 7;
pub const EXEC_COL_MODE_A: usize = 8;
pub const EXEC_COL_MODE_B: usize = 9;
pub const EXEC_COL_MODE_C: usize = 10;
pub const EXEC_COL_FLAG_MUL: usize = 11;
pub const EXEC_COL_FLAG_JUMP: usize = 12;
pub const EXEC_COL_AUX_1: usize = 13;
pub const EXEC_COL_AUX_2: usize = 14;

// Temporary columns (stored to avoid duplicate computations; NOT committed)
pub const N_TEMPORARY_EXEC_COLUMNS: usize = 7;
pub const EXEC_COL_ADDR_A: usize = 15;
pub const EXEC_COL_ADDR_B: usize = 16;
pub const EXEC_COL_ADDR_C: usize = 17;
pub const EXEC_COL_FLAG_PRECOMPILE: usize = 18;
pub const EXEC_COL_NU_A: usize = 19;
pub const EXEC_COL_NU_B: usize = 20;
pub const EXEC_COL_NU_C: usize = 21;

impl<const BUS: bool> Air for ExecutionTable<BUS> {
    type ExtraData = ExtraDataForBuses<EF>;

    fn n_columns(&self) -> usize {
        N_TOTAL_EXECUTION_COLUMNS
    }
    fn degree_air(&self) -> usize {
        // h9-B: mode decoders are quadratic in M_*, so nu_* are degree 3 and the
        // worst constraints (flag_mul·(nu_b − nu_a·nu_c); the jump family
        // (flag_jump·nu_a)·(… − nu_*)) reach degree 7. Global max_full_degree
        // stays 9 (poseidon declares 8) ⇒ sumcheck wire format unchanged.
        7
    }
    // C2 kill rule, measured iter-3 T3' (3x interleaved A/B, 1550-sig xmss):
    // exec class poly +1.5ms / fold +5.4ms = +6.9ms net REGRESSION — the cheap
    // 14-constraint eval (~235 ns/pair EF) does not amortize the table's
    // challenge-time extrapolation + cache traffic (plan_spec §5.1 thin case).
    // Poseidon (~3338 ns/pair, 106 constraints) nets -39.7ms and keeps C2.
    fn c2_table_profitable(&self) -> bool {
        false
    }
    fn n_shift_columns(&self) -> usize {
        2
    }
    fn n_constraints(&self) -> usize {
        // h9-A: 14 - 4 address-definition constraints (now definitional via the
        // virtual closed forms) + 3 encoded memory-bus asserts (deferred claims).
        13
    }

    #[inline]
    fn eval<AB: AirBuilder>(&self, builder: &mut AB, extra_data: &Self::ExtraData) {
        let flat = builder.flat();
        let shift = builder.shift();

        let pc_shift = shift[EXEC_COL_PC];
        let fp_shift = shift[EXEC_COL_FP];

        let (operand_a, operand_b, operand_c) = (
            flat[EXEC_COL_OPERAND_A],
            flat[EXEC_COL_OPERAND_B],
            flat[EXEC_COL_OPERAND_C],
        );
        let mode_a = flat[EXEC_COL_MODE_A];
        let mode_b = flat[EXEC_COL_MODE_B];
        let mode_c = flat[EXEC_COL_MODE_C];
        let flag_mul = flat[EXEC_COL_FLAG_MUL];
        let flag_jump = flat[EXEC_COL_FLAG_JUMP];
        let aux_1 = flat[EXEC_COL_AUX_1];
        let aux_2 = flat[EXEC_COL_AUX_2];

        let (value_a, value_b, value_c) = (flat[EXEC_COL_VALUE_A], flat[EXEC_COL_VALUE_B], flat[EXEC_COL_VALUE_C]);
        let pc = flat[EXEC_COL_PC];
        let fp = flat[EXEC_COL_FP];

        // h9-B Lagrange decoders over {0,1,2} (flag_add/flag_deref precedent):
        //   flag_x   = 1 iff M = 1:  M(2−M)        = 2M − M²
        //   fp_x     = 1 iff M = 2:  M(M−1)/2
        //   mem_x    = 1 iff M = 0:  1 − flag_x − fp_x  (= (M−1)(M−2)/2)
        let flag_a = mode_a * AB::F::TWO - mode_a * mode_a;
        let fab_a = (mode_a * (mode_a - AB::F::ONE)).halve();
        let flag_b = mode_b * AB::F::TWO - mode_b * mode_b;
        let fab_b = (mode_b * (mode_b - AB::F::ONE)).halve();
        let flag_c = mode_c * AB::F::TWO - mode_c * mode_c;
        let flag_c_fp = (mode_c * (mode_c - AB::F::ONE)).halve();

        let one_minus_flag_a_and_flag_ab_fp = -(flag_a + fab_a - AB::F::ONE);
        let one_minus_flag_b_and_flag_ab_fp = -(flag_b + fab_b - AB::F::ONE);
        let one_minus_flag_c_and_flag_c_fp = -(flag_c + flag_c_fp - AB::F::ONE);

        let nu_a = flag_a * operand_a + one_minus_flag_a_and_flag_ab_fp * value_a + fab_a * (fp + operand_a);
        let nu_b = flag_b * operand_b + one_minus_flag_b_and_flag_ab_fp * value_b + fab_b * (fp + operand_b);
        let nu_c = flag_c * operand_c + one_minus_flag_c_and_flag_c_fp * value_c + flag_c_fp * (fp + operand_c);

        let fp_plus_operand_a = fp + operand_a;
        let fp_plus_operand_b = fp + operand_b;
        let fp_plus_operand_c = fp + operand_c;
        let pc_plus_one = pc + AB::F::ONE;
        let nu_a_minus_one = nu_a - AB::F::ONE;

        let flag_add = aux_1 * AB::F::TWO - aux_1 * aux_1;
        let flag_deref = (aux_1 * (aux_1 - AB::F::ONE)).halve();
        let flag_precompile = -(flag_add + flag_mul + flag_deref + flag_jump - AB::F::ONE);

        // h9-A virtual addresses: the exact closed forms of trace_gen.rs (the old
        // committed columns satisfied these identically on every real row; the old
        // address-definition constraints are therefore definitional and deleted).
        // Degrees: addr_a/c 2, addr_b 3 (flag_deref is quadratic in aux_1).
        let addr_a = one_minus_flag_a_and_flag_ab_fp * fp_plus_operand_a;
        let addr_b = one_minus_flag_b_and_flag_ab_fp * fp_plus_operand_b + flag_deref * (value_a + operand_b);
        let addr_c = one_minus_flag_c_and_flag_c_fp * fp_plus_operand_c;

        if BUS {
            eval_bus_virtual::<AB, EF>(builder, extra_data, flag_precompile, aux_2, &[nu_a, nu_b, nu_c]);
            // h9-A deferred memory-bus claims: one encoded assert per memory lookup,
            // proven inside the batched AIR sumcheck (encoded degree <= 3+1 < degree_air).
            // Order matters: these occupy the constraint slots directly after the
            // precompile bus pair; the verifier's initial_sum loop matches this order.
            eval_bus_data_only::<AB, EF>(builder, extra_data, LOGUP_MEMORY_DOMAINSEP, &[addr_a, value_a]);
            eval_bus_data_only::<AB, EF>(builder, extra_data, LOGUP_MEMORY_DOMAINSEP, &[addr_b, value_b]);
            eval_bus_data_only::<AB, EF>(builder, extra_data, LOGUP_MEMORY_DOMAINSEP, &[addr_c, value_c]);
        } else {
            builder.declare_values(&[flag_precompile]);
            builder.declare_values(&[nu_a, nu_b, nu_c, aux_2]);
            builder.declare_values(&[addr_a, addr_b, addr_c]);
        }

        builder.assert_zero(flag_add * (nu_b - (nu_a + nu_c)));
        builder.assert_zero(flag_mul * (nu_b - nu_a * nu_c));

        // DEREF: result in value_B, compared to nu_C (the addr_B linkage is definitional now)
        builder.assert_zero(flag_deref * (value_b - nu_c));

        let jump_and_condition = flag_jump * nu_a;

        builder.assert_zero(jump_and_condition * nu_a_minus_one);
        builder.assert_zero(jump_and_condition * (pc_shift - nu_b));
        builder.assert_zero(jump_and_condition * (fp_shift - nu_c));
        let not_jump_and_condition = -(jump_and_condition - AB::F::ONE);
        builder.assert_zero(not_jump_and_condition * (pc_shift - pc_plus_one));
        builder.assert_zero(not_jump_and_condition * (fp_shift - fp));
    }
}

pub const fn instr_idx(col_index_in_air: usize) -> usize {
    col_index_in_air - N_RUNTIME_COLUMNS
}
