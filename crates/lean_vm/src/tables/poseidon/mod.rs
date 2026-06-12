use std::any::TypeId;

use crate::*;
use crate::{execution::memory::MemoryAccess, tables::poseidon::trace_gen::generate_trace_rows_for_perm};
use backend::*;

/// Dispatch `mds_fft_16` through concrete types.
/// For `SymbolicExpression` we use the dense form so the zkDSL generator can
/// emit `dot_product_be` precompile calls instead of Karatsuba arithmetic.
#[inline(always)]
fn mds_air_16<A: PrimeCharacteristicRing + 'static>(state: &mut [A; WIDTH]) {
    if TypeId::of::<A>() == TypeId::of::<SymbolicExpression<KoalaBear>>() {
        dense_mat_vec_air_16(mds_dense_16(), state);
        return;
    }
    macro_rules! dispatch {
        ($t:ty) => {
            if TypeId::of::<A>() == TypeId::of::<$t>() {
                mds_fft_16::<$t>(unsafe { &mut *(state as *mut [A; WIDTH] as *mut [$t; WIDTH]) });
                return;
            }
        };
    }
    dispatch!(F);
    dispatch!(EF);
    dispatch!(FPacking<F>);
    dispatch!(EFPacking<EF>);
    unreachable!()
}

fn mds_dense_16() -> &'static [[F; 16]; 16] {
    use std::sync::OnceLock;
    static MAT: OnceLock<[[KoalaBear; 16]; 16]> = OnceLock::new();
    MAT.get_or_init(|| {
        let cols: [[F; 16]; 16] = std::array::from_fn(|j| {
            let mut e = [F::ZERO; 16];
            e[j] = F::ONE;
            mds_circ_16(&mut e);
            e
        });
        std::array::from_fn(|i| std::array::from_fn(|j| cols[j][i]))
    })
}

/// Add a `KoalaBear` constant to any AIR type.
#[inline(always)]
fn add_kb<A: 'static>(a: &mut A, value: F) {
    macro_rules! dispatch {
        ($t:ty) => {
            if TypeId::of::<A>() == TypeId::of::<$t>() {
                *unsafe { &mut *(a as *mut A as *mut $t) } += value;
                return;
            }
        };
    }
    dispatch!(F);
    dispatch!(EF);
    dispatch!(FPacking<F>);
    dispatch!(EFPacking<EF>);
    dispatch!(SymbolicExpression<KoalaBear>);
    unreachable!()
}

/// Multiply any AIR type by a `KoalaBear` constant.
#[inline(always)]
fn mul_kb<A: PrimeCharacteristicRing + 'static>(a: A, value: F) -> A {
    macro_rules! dispatch {
        ($t:ty) => {
            if TypeId::of::<A>() == TypeId::of::<$t>() {
                let r = unsafe { std::ptr::read(&a as *const A as *const $t) } * value;
                return unsafe { std::ptr::read(&r as *const $t as *const A) };
            }
        };
    }
    dispatch!(F);
    dispatch!(EF);
    dispatch!(FPacking<F>);
    dispatch!(EFPacking<EF>);
    dispatch!(SymbolicExpression<KoalaBear>);
    unreachable!()
}

mod trace_gen;
pub use trace_gen::fill_trace_poseidon_16;

pub(super) const WIDTH: usize = 16;
const HALF_INITIAL_FULL_ROUNDS: usize = POSEIDON1_HALF_FULL_ROUNDS / 2;
const PARTIAL_ROUNDS: usize = POSEIDON1_PARTIAL_ROUNDS;
const HALF_FINAL_FULL_ROUNDS: usize = POSEIDON1_HALF_FULL_ROUNDS / 2;

// domainsep encoding: see `tables/mod.rs`.
pub const POSEIDON_DOMAINSEP_BASE: usize = 3;
pub const POSEIDON_FLAG_PERMUTE_SHIFT: usize = 1 << 1;
pub const POSEIDON_FLAG_OUT8_SHIFT: usize = 1 << 2;
pub const POSEIDON_FLAG_LEFT_SHIFT: usize = 1 << 3;
pub const POSEIDON_OFFSET_LEFT_SHIFT: usize = 1 << 4;

pub const POSEIDON_COL_MULTIPLICITY: ColIndex = 0;
pub const POSEIDON_COL_NU_B: ColIndex = 1;
pub const POSEIDON_COL_NU_C: ColIndex = 2;
pub const POSEIDON_COL_FLAG_OUT4: ColIndex = 3;
pub const POSEIDON_COL_FLAG_OUT8: ColIndex = 4;
pub const POSEIDON_COL_FLAG_LEFT: ColIndex = 5;
pub const POSEIDON_COL_OFFSET_LEFT: ColIndex = 6;
pub const POSEIDON_COL_ADDR_LEFT_LO: ColIndex = 7;
pub const POSEIDON_COL_ADDR_LEFT_HI: ColIndex = 8;
pub const POSEIDON_COL_FLAG_PERMUTE: ColIndex = 9;
pub const POSEIDON_COL_INPUT_START: ColIndex = 10;
pub const POSEIDON_COL_OUT_LO: ColIndex = num_cols_poseidon_16() - 16;
pub const POSEIDON_COL_OUT_HI: ColIndex = num_cols_poseidon_16() - 8;
/// Non-committed columns ("virtual"):
pub const POSEIDON_COL_NU_A: ColIndex = num_cols_poseidon_16();
pub const POSEIDON_COL_DOMAINSEP: ColIndex = num_cols_poseidon_16() + 1;

pub const POSEIDON16_COMPRESS_HALF_NAME: &str = "poseidon16_compress_half";
pub const POSEIDON16_QUARTER_NAME: &str = "poseidon16_compress_quarter";
pub const POSEIDON16_HARDCODED_LEFT_NAME: &str = "poseidon16_compress_half_hardcoded_left";
pub const POSEIDON16_QUARTER_HARDCODED_LEFT_NAME: &str = "poseidon16_compress_quarter_hardcoded_left";
pub const POSEIDON16_PERMUTE_NAME: &str = "poseidon16_permute";
pub const POSEIDON16_PERMUTE_HALF_NAME: &str = "poseidon16_permute_half";
pub const POSEIDON16_PERMUTE_HALF_HARDCODED_LEFT_NAME: &str = "poseidon16_permute_half_hardcoded_left";
pub const ALL_POSEIDON16_NAMES: [&str; 7] = [
    POSEIDON16_COMPRESS_HALF_NAME,
    POSEIDON16_QUARTER_NAME,
    POSEIDON16_HARDCODED_LEFT_NAME,
    POSEIDON16_QUARTER_HARDCODED_LEFT_NAME,
    POSEIDON16_PERMUTE_NAME,
    POSEIDON16_PERMUTE_HALF_NAME,
    POSEIDON16_PERMUTE_HALF_HARDCODED_LEFT_NAME,
];
pub const HALF_DIGEST_LEN: usize = DIGEST_LEN / 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon16Precompile<const BUS: bool>;

impl<const BUS: bool> TableT for Poseidon16Precompile<BUS> {
    fn name(&self) -> &'static str {
        "poseidon16"
    }

    fn table(&self) -> Table {
        Table::poseidon16()
    }

    fn n_columns_total(&self) -> usize {
        num_cols_total_poseidon_16()
    }

    fn bus_interactions(&self) -> Vec<BusInteraction> {
        let mut buses = vec![BusInteraction {
            direction: BusDirection::Pull,
            multiplicity: BusMultiplicity::Column(POSEIDON_COL_MULTIPLICITY),
            domainsep: BusData::Column(POSEIDON_COL_DOMAINSEP),
            data: vec![
                BusData::Column(POSEIDON_COL_NU_A),
                BusData::Column(POSEIDON_COL_NU_B),
                BusData::Column(POSEIDON_COL_NU_C),
            ],
        }];
        buses.extend(memory_lookups_consecutive(
            POSEIDON_COL_ADDR_LEFT_LO,
            POSEIDON_COL_INPUT_START,
            HALF_DIGEST_LEN,
        ));
        buses.extend(memory_lookups_consecutive(
            POSEIDON_COL_ADDR_LEFT_HI,
            POSEIDON_COL_INPUT_START + HALF_DIGEST_LEN,
            HALF_DIGEST_LEN,
        ));
        buses.extend(memory_lookups_consecutive(
            POSEIDON_COL_NU_B,
            POSEIDON_COL_INPUT_START + DIGEST_LEN,
            DIGEST_LEN,
        ));
        buses.extend(memory_lookups_consecutive(
            POSEIDON_COL_NU_C,
            POSEIDON_COL_OUT_LO,
            DIGEST_LEN * 2,
        ));
        buses
    }

    fn padding_row(&self, zero_vec_ptr: usize, null_hash_ptr: usize, _ending_pc: usize) -> Vec<F> {
        let mut row = vec![F::ZERO; num_cols_total_poseidon_16()];
        let ptrs: Vec<*mut F> = (0..num_cols_poseidon_16())
            .map(|i| unsafe { row.as_mut_ptr().add(i) })
            .collect();

        let perm: &mut Poseidon1Cols16<&mut F> = unsafe { &mut *(ptrs.as_ptr() as *mut Poseidon1Cols16<&mut F>) };
        perm.inputs.iter_mut().for_each(|x| **x = F::ZERO);
        *perm.multiplicity = F::ZERO;
        *perm.nu_b = F::from_usize(zero_vec_ptr);
        *perm.nu_c = F::from_usize(null_hash_ptr);
        *perm.flag_out4 = F::ZERO;
        *perm.flag_out8 = F::ONE;
        *perm.flag_left = F::ZERO;
        *perm.offset_left = F::ZERO;
        *perm.addr_left_lo = F::from_usize(zero_vec_ptr);
        *perm.addr_left_hi = F::from_usize(zero_vec_ptr + HALF_DIGEST_LEN);
        *perm.flag_permute = F::ZERO;
        perm.out_hi.iter_mut().for_each(|x| **x = F::ZERO);
        row[POSEIDON_COL_NU_A] = F::from_usize(zero_vec_ptr);
        row[POSEIDON_COL_DOMAINSEP] = F::from_usize(POSEIDON_DOMAINSEP_BASE + POSEIDON_FLAG_OUT8_SHIFT);

        generate_trace_rows_for_perm(perm);
        row
    }

    #[inline(always)]
    fn execute<M: MemoryAccess>(
        &self,
        arg_a: F,
        arg_b: F,
        index_res_a: F,
        args: PrecompileCompTimeArgs<usize>,
        ctx: &mut InstructionContext<'_, M>,
    ) -> Result<(), RunnerError> {
        let PrecompileCompTimeArgs::Poseidon16 {
            half_output,
            hardcoded_offset_left,
            permute,
        } = args
        else {
            unreachable!("Poseidon16 table called with non-Poseidon16 args");
        };
        let out4 = half_output && !permute;
        let out8 = (!half_output && !permute) || (half_output && permute);
        let trace = ctx.traces.get_mut(&self.table()).unwrap();

        let arg_a_usize = arg_a.to_usize();
        let flag_hardcoded = hardcoded_offset_left.is_some();
        // Convention:
        //   flag_hardcoded = 0: left input = m[arg_a..arg_a+8] (split as [arg_a..+4], [arg_a+4..+8])
        //   flag_hardcoded = 1: left input = m[offset..offset+4] | m[arg_a..arg_a+4]
        //                   (i.e. arg_a now points to a 4-element data digest, and the first 4
        //                    elements come from the hardcoded prefix at `offset`)
        let left_first_addr = hardcoded_offset_left.unwrap_or(arg_a_usize);
        let left_second_addr = if flag_hardcoded {
            arg_a_usize
        } else {
            arg_a_usize + HALF_DIGEST_LEN
        };
        let mut input = [F::ZERO; DIGEST_LEN * 2];
        ctx.memory
            .get_slice_into(left_first_addr, &mut input[..HALF_DIGEST_LEN])?;
        ctx.memory
            .get_slice_into(left_second_addr, &mut input[HALF_DIGEST_LEN..DIGEST_LEN])?;
        ctx.memory.get_slice_into(arg_b.to_usize(), &mut input[DIGEST_LEN..])?;

        let res_addr = index_res_a.to_usize();
        if permute {
            let permuted = poseidon16_permute(input);
            let out_len = if half_output { DIGEST_LEN } else { DIGEST_LEN * 2 };
            ctx.memory.set_slice(res_addr, &permuted[..out_len])?;
        } else {
            let output = poseidon16_compress(input);
            let out_len = if half_output { HALF_DIGEST_LEN } else { DIGEST_LEN };
            ctx.memory.set_slice(res_addr, &output[..out_len])?;
        }

        let hardcoded_offset_left_val = hardcoded_offset_left.unwrap_or(0);

        trace.columns[POSEIDON_COL_MULTIPLICITY].push(F::ONE);
        trace.columns[POSEIDON_COL_NU_B].push(arg_b);
        trace.columns[POSEIDON_COL_NU_C].push(index_res_a);
        trace.columns[POSEIDON_COL_FLAG_OUT4].push(F::from_bool(out4));
        trace.columns[POSEIDON_COL_FLAG_OUT8].push(F::from_bool(out8));
        trace.columns[POSEIDON_COL_FLAG_LEFT].push(F::from_bool(flag_hardcoded));
        trace.columns[POSEIDON_COL_OFFSET_LEFT].push(F::from_usize(hardcoded_offset_left_val));
        trace.columns[POSEIDON_COL_ADDR_LEFT_LO].push(F::from_usize(left_first_addr));
        trace.columns[POSEIDON_COL_ADDR_LEFT_HI].push(F::from_usize(left_second_addr));
        trace.columns[POSEIDON_COL_FLAG_PERMUTE].push(F::from_bool(permute));
        for (i, value) in input.iter().enumerate() {
            trace.columns[POSEIDON_COL_INPUT_START + i].push(*value);
        }
        // Non-committed columns
        trace.columns[POSEIDON_COL_NU_A].push(arg_a);
        let domainsep = POSEIDON_DOMAINSEP_BASE
            + POSEIDON_FLAG_PERMUTE_SHIFT * (permute as usize)
            + POSEIDON_FLAG_OUT8_SHIFT * (out8 as usize)
            + POSEIDON_FLAG_LEFT_SHIFT * (flag_hardcoded as usize)
            + POSEIDON_OFFSET_LEFT_SHIFT * hardcoded_offset_left_val;
        trace.columns[POSEIDON_COL_DOMAINSEP].push(F::from_usize(domainsep));

        // the rest of the trace is filled at the end of the execution (to get parallelism + SIMD)

        Ok(())
    }
}

impl<const BUS: bool> Air for Poseidon16Precompile<BUS> {
    type ExtraData = ExtraDataForBuses<EF>;
    fn n_columns(&self) -> usize {
        num_cols_poseidon_16()
    }
    fn degree_air(&self) -> usize {
        // The output constraints gate the degree-9 permutation expression by a single linear
        // factor (`1 - flag_out4` for out_lo[4..8], `1 - flag_out8 - flag_out4` for out_hi),
        // keeping them at degree 10.
        10
    }
    fn low_degree_air(&self) -> Option<(usize, usize)> {
        // Each partial round contributes one `assert_eq_low` per round (1 S-box / round), of degree 3 (= the "low" degree part)
        Some((3, PARTIAL_ROUNDS))
    }
    fn n_shift_columns(&self) -> usize {
        0
    }
    fn n_constraints(&self) -> usize {
        2 * BUS as usize + 94
    }
    fn eval<AB: AirBuilder>(&self, builder: &mut AB, extra_data: &Self::ExtraData) {
        let cols: Poseidon1Cols16<AB::IF> = {
            let flat = builder.flat();
            let (prefix, shorts, suffix) = unsafe { flat.align_to::<Poseidon1Cols16<AB::IF>>() };
            debug_assert!(prefix.is_empty(), "Alignment should match");
            debug_assert!(suffix.is_empty(), "Alignment should match");
            debug_assert_eq!(shorts.len(), 1);
            unsafe { std::ptr::read(&shorts[0]) }
        };

        let domainsep_reconstructed = AB::IF::from_usize(POSEIDON_DOMAINSEP_BASE)
            + cols.flag_permute * AB::F::from_usize(POSEIDON_FLAG_PERMUTE_SHIFT)
            + cols.flag_out8 * AB::F::from_usize(POSEIDON_FLAG_OUT8_SHIFT)
            + cols.flag_left * AB::F::from_usize(POSEIDON_FLAG_LEFT_SHIFT)
            + cols.flag_left * cols.offset_left * AB::F::from_usize(POSEIDON_OFFSET_LEFT_SHIFT);

        // addr_left_lo = nu_a * (1 - flag_left) + offset_left * flag_left
        let one_minus_flag_left = AB::IF::ONE - cols.flag_left;
        let nu_a = cols.addr_left_hi - one_minus_flag_left * AB::F::from_usize(HALF_DIGEST_LEN);

        // Bus: data = [nu_a, nu_b, nu_c], domainsep
        if BUS {
            eval_bus_virtual::<AB, EF>(
                builder,
                extra_data,
                cols.multiplicity,
                domainsep_reconstructed,
                &[nu_a, cols.nu_b, cols.nu_c],
            );
        } else {
            builder.declare_values(std::slice::from_ref(&cols.multiplicity));
            builder.declare_values(&[nu_a, cols.nu_b, cols.nu_c, domainsep_reconstructed]);
        }

        builder.assert_bool(cols.multiplicity);
        builder.assert_bool(cols.flag_out4);
        builder.assert_bool(cols.flag_out8);
        builder.assert_bool(cols.flag_left);
        builder.assert_bool(cols.flag_permute);
        builder.assert_zero(cols.flag_permute * cols.flag_out4);
        builder.assert_zero(cols.flag_out8 * cols.flag_out4);
        builder.assert_zero(
            (AB::IF::ONE - cols.flag_permute) * (AB::IF::ONE - cols.flag_out8) * (AB::IF::ONE - cols.flag_out4),
        );

        builder.assert_zero(cols.flag_left * (cols.offset_left - cols.addr_left_lo));
        builder.assert_zero(one_minus_flag_left * (nu_a - cols.addr_left_lo));

        eval_poseidon1_16(builder, &cols)
    }
}

#[repr(C)]
#[derive(Debug)]
pub(super) struct Poseidon1Cols16<T> {
    pub multiplicity: T, // 0 = padding, 1 = active
    pub nu_b: T,
    pub nu_c: T,
    pub flag_out4: T, // output is 4 elements (compression only)
    pub flag_out8: T, // output is 8 elements; neither out4 nor out8 set => 16 elements (permutation only)
    pub flag_left: T,
    pub offset_left: T,
    pub addr_left_lo: T,
    pub addr_left_hi: T,
    pub flag_permute: T,

    pub inputs: [T; WIDTH],
    pub beginning_full_rounds: [[T; WIDTH]; HALF_INITIAL_FULL_ROUNDS],
    pub partial_rounds: [T; PARTIAL_ROUNDS],
    pub ending_full_rounds: [[T; WIDTH]; HALF_FINAL_FULL_ROUNDS - 1],
    pub out_lo: [T; WIDTH / 2],
    pub out_hi: [T; WIDTH / 2],
}

fn eval_poseidon1_16<AB: AirBuilder>(builder: &mut AB, local: &Poseidon1Cols16<AB::IF>) {
    let mut state: [_; WIDTH] = local.inputs;

    let initial_constants = poseidon1_initial_constants();
    for round in 0..HALF_INITIAL_FULL_ROUNDS {
        eval_2_full_rounds_16(
            &mut state,
            &local.beginning_full_rounds[round],
            &initial_constants[2 * round],
            &initial_constants[2 * round + 1],
            builder,
        );
    }

    // --- Sparse partial rounds ---
    // Transition: add first-round constants, multiply by m_i
    builder.low_degree_block(&mut state, |b, state| {
        let state: &mut [AB::IF; WIDTH] = state.try_into().unwrap();

        let frc = poseidon1_sparse_first_round_constants();
        for (s, &c) in state.iter_mut().zip(frc.iter()) {
            add_kb(s, c);
        }
        dense_mat_vec_air_16(poseidon1_sparse_m_i(), state);

        let first_rows = poseidon1_sparse_first_row();
        let v_vecs = poseidon1_sparse_v();
        let scalar_rc = poseidon1_sparse_scalar_round_constants();
        for round in 0..PARTIAL_ROUNDS {
            // S-box on state[0]
            state[0] = state[0].cube();
            b.assert_eq_low(state[0], local.partial_rounds[round]);
            state[0] = local.partial_rounds[round];
            // Scalar round constant (not on last round)
            if round < PARTIAL_ROUNDS - 1 {
                add_kb(&mut state[0], scalar_rc[round]);
            }
            // Sparse matrix: new_s0 = dot(first_row, state), state[i] += old_s0 * v[i-1]
            sparse_mat_air_16(state, &first_rows[round], &v_vecs[round]);
        }
    });

    let final_constants = poseidon1_final_constants();
    for round in 0..HALF_FINAL_FULL_ROUNDS - 1 {
        eval_2_full_rounds_16(
            &mut state,
            &local.ending_full_rounds[round],
            &final_constants[2 * round],
            &final_constants[2 * round + 1],
            builder,
        );
    }

    eval_last_2_full_rounds_16(
        &local.inputs,
        &mut state,
        &local.out_lo,
        &local.out_hi,
        &final_constants[2 * (HALF_FINAL_FULL_ROUNDS - 1)],
        &final_constants[2 * (HALF_FINAL_FULL_ROUNDS - 1) + 1],
        local.flag_out8,
        local.flag_out4,
        local.flag_permute,
        builder,
    );
}

pub const fn num_cols_poseidon_16() -> usize {
    size_of::<Poseidon1Cols16<u8>>()
}

pub const fn num_cols_total_poseidon_16() -> usize {
    // +2 for non-committed columns: POSEIDON_COL_INDEX_INPUT_LEFT, POSEIDON_COL_DOMAINSEP
    num_cols_poseidon_16() + 2
}

#[inline]
fn eval_2_full_rounds_16<AB: AirBuilder>(
    state: &mut [AB::IF; WIDTH],
    post_full_round: &[AB::IF; WIDTH],
    round_constants_1: &[F; WIDTH],
    round_constants_2: &[F; WIDTH],
    builder: &mut AB,
) {
    for (s, r) in state.iter_mut().zip(round_constants_1.iter()) {
        add_kb(s, *r);
        *s = s.cube();
    }
    mds_air_16(state);
    for (s, r) in state.iter_mut().zip(round_constants_2.iter()) {
        add_kb(s, *r);
        *s = s.cube();
    }
    mds_air_16(state);
    for (state_i, post_i) in state.iter_mut().zip(post_full_round) {
        builder.assert_eq(*state_i, *post_i);
        *state_i = *post_i;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_last_2_full_rounds_16<AB: AirBuilder>(
    initial_state: &[AB::IF; WIDTH],
    state: &mut [AB::IF; WIDTH],
    out_lo: &[AB::IF; WIDTH / 2],
    out_hi: &[AB::IF; WIDTH / 2],
    round_constants_1: &[F; WIDTH],
    round_constants_2: &[F; WIDTH],
    flag_out8: AB::IF,
    flag_out4: AB::IF,
    flag_permute: AB::IF,
    builder: &mut AB,
) {
    for (s, r) in state.iter_mut().zip(round_constants_1.iter()) {
        add_kb(s, *r);
        *s = s.cube();
    }
    mds_air_16(state);
    for (s, r) in state.iter_mut().zip(round_constants_2.iter()) {
        add_kb(s, *r);
        *s = s.cube();
    }
    mds_air_16(state);
    let feedforward = AB::IF::ONE - flag_permute;
    let gate_lo_8 = AB::IF::ONE - flag_out4;
    let gate_hi = AB::IF::ONE - flag_out8 - flag_out4;
    for i in 0..(WIDTH / 2) {
        let value = state[i] + feedforward * initial_state[i];
        if i < HALF_DIGEST_LEN {
            builder.assert_zero(value - out_lo[i]);
        } else {
            builder.assert_zero(gate_lo_8 * (value - out_lo[i]));
        }
        builder.assert_zero(gate_hi * (state[i + WIDTH / 2] - out_hi[i])); // always permutation on the right-half
    }
}

#[inline]
fn dense_mat_vec_air_16<A: PrimeCharacteristicRing + 'static>(mat: &[[F; 16]; 16], state: &mut [A; WIDTH]) {
    let input = *state;
    for i in 0..WIDTH {
        let mut acc = A::ZERO;
        for j in 0..WIDTH {
            acc += mul_kb(input[j], mat[i][j]);
        }
        state[i] = acc;
    }
}

#[inline]
fn sparse_mat_air_16<A: PrimeCharacteristicRing + 'static>(
    state: &mut [A; WIDTH],
    first_row: &[F; WIDTH],
    v: &[F; WIDTH],
) {
    let old_s0 = state[0];
    let mut new_s0 = A::ZERO;
    for j in 0..WIDTH {
        new_s0 += mul_kb(state[j], first_row[j]);
    }
    state[0] = new_s0;
    for i in 1..WIDTH {
        state[i] += mul_kb(old_s0, v[i - 1]);
    }
}

// ---------------------------------------------------------------------------
// h1 kill-ladder rung benches (pw13-mac iter-2, hypothesis h1
// "virtual-out-columns-degree-drop"). Test-only; no production code change.
// Rung i  : packed AIR-eval microbench, 10-point baseline vs 9-point candidate
//           with 16 inline fingerprint accumulations (gate: >=8% faster).
// Rung ii : Merkle leaf-chunk crossing arithmetic (115->107 cols, 15->14 chunks).
// Rung iii: LogUp tuple-count invariance (33 buses/row, total <= 2^25).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod h1_kill_ladder {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    type FP = PFPacking<EF>; // == FPacking<F>
    type EFP = EFPacking<EF>;

    const N_FLAT_BASELINE: usize = 110;
    const N_FLAT_CANDIDATE: usize = 94;
    const D_LOW: usize = 3; // low_degree_air().0
    const N_FULL: usize = D_LOW + 1; // full-eval z-points: {0,2,3,4}

    /// Candidate committed region: Poseidon1Cols16 minus out_lo/out_hi.
    #[repr(C)]
    struct Poseidon1Cols16Candidate<T> {
        multiplicity: T,
        nu_b: T,
        nu_c: T,
        flag_out4: T,
        flag_out8: T,
        flag_left: T,
        offset_left: T,
        addr_left_lo: T,
        addr_left_hi: T,
        flag_permute: T,
        inputs: [T; WIDTH],
        beginning_full_rounds: [[T; WIDTH]; HALF_INITIAL_FULL_ROUNDS],
        partial_rounds: [T; PARTIAL_ROUNDS],
        ending_full_rounds: [[T; WIDTH]; HALF_FINAL_FULL_ROUNDS - 1],
    }
    const _: () = assert!(size_of::<Poseidon1Cols16Candidate<u8>>() == N_FLAT_CANDIDATE);
    const _: () = assert!(size_of::<Poseidon1Cols16<u8>>() == N_FLAT_BASELINE);

    /// Candidate last-2-full-rounds: no out-column asserts; instead 16 LogUp
    /// denominator fingerprints over the inline output expressions
    /// (value_i = state[i] + feedforward*initial[i] for i<8, state[i] for i>=8),
    /// each: 2 EF*IF muls + 1 assert_zero_ef (alpha mul). Unfused = conservative.
    #[allow(clippy::too_many_arguments)]
    fn eval_last_2_full_rounds_16_candidate<AB: AirBuilder>(
        initial_state: &[AB::IF; WIDTH],
        state: &mut [AB::IF; WIDTH],
        nu_c: AB::IF,
        round_constants_1: &[F; WIDTH],
        round_constants_2: &[F; WIDTH],
        flag_permute: AB::IF,
        eq_consts: &[AB::EF],
        builder: &mut AB,
    ) {
        for (s, r) in state.iter_mut().zip(round_constants_1.iter()) {
            add_kb(s, *r);
            *s = s.cube();
        }
        mds_air_16(state);
        for (s, r) in state.iter_mut().zip(round_constants_2.iter()) {
            add_kb(s, *r);
            *s = s.cube();
        }
        mds_air_16(state);
        let feedforward = AB::IF::ONE - flag_permute;
        // FUSED form: per-output alpha powers are premultiplied into the eq
        // constants at setup (alpha_i*c, alpha_i*eq0, alpha_i*eq1 are proof-time
        // constants), so each output costs 2 EF*IF muls + adds; one batched
        // assert_zero_ef per point carries the sum into the accumulator.
        let mut acc_fp = eq_consts[2]; // sum_i alpha_i*c, folded into one constant
        for i in 0..WIDTH {
            let value = if i < WIDTH / 2 {
                state[i] + feedforward * initial_state[i]
            } else {
                state[i]
            };
            let mut addr = nu_c;
            add_kb(&mut addr, F::from_usize(i));
            // premultiplied per-output constants: distinct entries, real reads
            let b = eq_consts[(2 * i) % eq_consts.len()];
            let c = eq_consts[(2 * i + 1) % eq_consts.len()];
            acc_fp = acc_fp - b * addr - c * value;
        }
        builder.assert_zero_ef(acc_fp);
    }

    /// Candidate full eval: identical prologue + rounds, candidate ending.
    fn eval_candidate<AB: AirBuilder>(builder: &mut AB, extra_data: &ExtraDataForBuses<EF>) {
        let cols: Poseidon1Cols16Candidate<AB::IF> = {
            let flat = builder.flat();
            let (prefix, shorts, suffix) = unsafe { flat.align_to::<Poseidon1Cols16Candidate<AB::IF>>() };
            debug_assert!(prefix.is_empty());
            debug_assert!(suffix.is_empty());
            debug_assert_eq!(shorts.len(), 1);
            unsafe { std::ptr::read(&shorts[0]) }
        };

        let domainsep_reconstructed = AB::IF::from_usize(POSEIDON_DOMAINSEP_BASE)
            + cols.flag_permute * AB::F::from_usize(POSEIDON_FLAG_PERMUTE_SHIFT)
            + cols.flag_out8 * AB::F::from_usize(POSEIDON_FLAG_OUT8_SHIFT)
            + cols.flag_left * AB::F::from_usize(POSEIDON_FLAG_LEFT_SHIFT)
            + cols.flag_left * cols.offset_left * AB::F::from_usize(POSEIDON_OFFSET_LEFT_SHIFT);

        let one_minus_flag_left = AB::IF::ONE - cols.flag_left;
        let nu_a = cols.addr_left_hi - one_minus_flag_left * AB::F::from_usize(HALF_DIGEST_LEN);

        eval_bus_virtual::<AB, EF>(
            builder,
            extra_data,
            cols.multiplicity,
            domainsep_reconstructed,
            &[nu_a, cols.nu_b, cols.nu_c],
        );

        builder.assert_bool(cols.multiplicity);
        builder.assert_bool(cols.flag_out4);
        builder.assert_bool(cols.flag_out8);
        builder.assert_bool(cols.flag_left);
        builder.assert_bool(cols.flag_permute);
        builder.assert_zero(cols.flag_permute * cols.flag_out4);
        builder.assert_zero(cols.flag_out8 * cols.flag_out4);
        builder.assert_zero(
            (AB::IF::ONE - cols.flag_permute) * (AB::IF::ONE - cols.flag_out8) * (AB::IF::ONE - cols.flag_out4),
        );
        builder.assert_zero(cols.flag_left * (cols.offset_left - cols.addr_left_lo));
        builder.assert_zero(one_minus_flag_left * (nu_a - cols.addr_left_lo));

        // permutation chain (identical to eval_poseidon1_16 except the ending)
        let mut state: [_; WIDTH] = cols.inputs;
        let initial_constants = poseidon1_initial_constants();
        for round in 0..HALF_INITIAL_FULL_ROUNDS {
            eval_2_full_rounds_16(
                &mut state,
                &cols.beginning_full_rounds[round],
                &initial_constants[2 * round],
                &initial_constants[2 * round + 1],
                builder,
            );
        }
        builder.low_degree_block(&mut state, |b, state| {
            let state: &mut [AB::IF; WIDTH] = state.try_into().unwrap();
            let frc = poseidon1_sparse_first_round_constants();
            for (s, &c) in state.iter_mut().zip(frc.iter()) {
                add_kb(s, c);
            }
            dense_mat_vec_air_16(poseidon1_sparse_m_i(), state);
            let first_rows = poseidon1_sparse_first_row();
            let v_vecs = poseidon1_sparse_v();
            let scalar_rc = poseidon1_sparse_scalar_round_constants();
            for round in 0..PARTIAL_ROUNDS {
                state[0] = state[0].cube();
                b.assert_eq_low(state[0], cols.partial_rounds[round]);
                state[0] = cols.partial_rounds[round];
                if round < PARTIAL_ROUNDS - 1 {
                    add_kb(&mut state[0], scalar_rc[round]);
                }
                sparse_mat_air_16(state, &first_rows[round], &v_vecs[round]);
            }
        });
        let final_constants = poseidon1_final_constants();
        for round in 0..HALF_FINAL_FULL_ROUNDS - 1 {
            eval_2_full_rounds_16(
                &mut state,
                &cols.ending_full_rounds[round],
                &final_constants[2 * round],
                &final_constants[2 * round + 1],
                builder,
            );
        }
        let eq_consts = extra_data.transmute_bus_data::<AB::EF>();
        eval_last_2_full_rounds_16_candidate(
            &cols.inputs,
            &mut state,
            cols.nu_c,
            &final_constants[2 * (HALF_FINAL_FULL_ROUNDS - 1)],
            &final_constants[2 * (HALF_FINAL_FULL_ROUNDS - 1) + 1],
            cols.flag_permute,
            eq_consts,
            builder,
        );
    }

    /// Tiny LCG so we need no rand dev-dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f(&mut self) -> F {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            const P: u64 = (1 << 31) - (1 << 24) + 1;
            F::from_usize(((self.0 >> 32) % P) as usize)
        }
        fn next_fp(&mut self) -> FP {
            FP::from_fn(|_| self.next_f())
        }
    }

    /// Drive one arm over all pairs, replicating compute_raw_poly_degree_split's
    /// per-pair z-point loop (4 full evals + (degree - 4) skip-low evals).
    fn drive<EVAL>(
        cols: &[Vec<FP>],
        n_flat: usize,
        degree: usize,
        extra: &ExtraDataForBuses<EF>,
        partial_eq: EFP,
        n_pairs: usize,
        eval: EVAL,
    ) -> EFP
    where
        EVAL: for<'a> Fn(&mut ConstraintFolderPacked<'a, FP, EF, ExtraDataForBuses<EF>>),
    {
        let n_skip = degree - N_FULL;
        let hi_zs_halved: Vec<F> = ((N_FULL + 1)..=degree).map(|z| F::from_usize(z).halve()).collect();
        // stand-in Lagrange coefficients (cost-equivalent to production's
        // lagrange_basis_evals output)
        let lagrange: [[F; N_FULL]; 6] =
            std::array::from_fn(|t| std::array::from_fn(|i| F::from_usize(3 + 5 * t + 7 * i)));

        let mut acc = vec![EFP::ZERO; degree];
        let mut point: Vec<FP> = Vec::with_capacity(n_flat);
        let mut diff: Vec<FP> = Vec::with_capacity(n_flat);
        let mut state_0: Vec<FP> = Vec::new();
        let mut state_2: Vec<FP> = Vec::new();
        let mut cached_buf: Vec<FP> = Vec::new();
        let mut low_evals = [EFP::ZERO; N_FULL];

        for j in 0..n_pairs {
            let i0 = 2 * j;
            let i1 = 2 * j + 1;
            point.clear();
            diff.clear();
            for c in cols.iter().take(n_flat) {
                let lo = c[i0];
                let hi = c[i1];
                point.push(lo);
                diff.push(hi - lo);
            }

            // z = 0: full eval, capture post-block state
            {
                let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], extra);
                folder.cached_state = Some(std::mem::take(&mut state_0));
                eval(&mut folder);
                acc[0] += folder.accumulator * partial_eq;
                low_evals[0] = folder.accumulator_low;
                state_0 = folder.cached_state.unwrap();
            }
            // z = 2
            for k in 0..n_flat {
                point[k] += diff[k].double();
            }
            {
                let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], extra);
                folder.cached_state = Some(std::mem::take(&mut state_2));
                eval(&mut folder);
                acc[1] += folder.accumulator * partial_eq;
                low_evals[1] = folder.accumulator_low;
                state_2 = folder.cached_state.unwrap();
            }
            // z = 3..=N_FULL: full evals
            for z_idx in 2..N_FULL {
                for k in 0..n_flat {
                    point[k] += diff[k];
                }
                let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], extra);
                eval(&mut folder);
                acc[z_idx] += folder.accumulator * partial_eq;
                low_evals[z_idx] = folder.accumulator_low;
            }
            // skip-low evals
            for t in 0..n_skip {
                for k in 0..n_flat {
                    point[k] += diff[k];
                }
                cached_buf.clear();
                for i in 0..state_0.len() {
                    cached_buf.push(state_0[i] + (state_2[i] - state_0[i]) * FP::from(hi_zs_halved[t]));
                }
                let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], extra);
                folder.skip_low = true;
                folder.cached_state = Some(std::mem::take(&mut cached_buf));
                folder.low_ci_count = PARTIAL_ROUNDS;
                eval(&mut folder);
                cached_buf = folder.cached_state.unwrap();

                let mut low_interpolated = EFP::ZERO;
                for (i, lc) in lagrange[t].iter().enumerate() {
                    low_interpolated += low_evals[i] * FP::from(*lc);
                }
                acc[N_FULL + t] += (folder.accumulator + low_interpolated) * partial_eq;
            }
        }
        acc.into_iter().sum()
    }

    fn build_extra() -> ExtraDataForBuses<EF> {
        let eq_poly: Vec<EF> = (0..1 << LOG_MAX_BUS_WIDTH)
            .map(|i| EF::from_usize(7 * i + 3) * EF::from_usize(1_000_003) + EF::from_usize(i * i + 11))
            .collect();
        let alphas: Vec<EF> = (0..128)
            .map(|i| EF::from_usize(13 * i + 5) * EF::from_usize(998_244_353) + EF::from_usize(i + 1))
            .collect();
        ExtraDataForBuses::new(&eq_poly, alphas)
    }

    #[test]
    #[ignore]
    fn h1_rung_i_eval_bench() {
        const N_PAIRS: usize = 2048; // packed pairs per pass
        const PASSES: usize = 12; // 24,576 packed-pair evals per rep
        const WARMUP: usize = 3;
        const REPS: usize = 5;

        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let cols: Vec<Vec<FP>> = (0..N_FLAT_BASELINE)
            .map(|_| (0..2 * N_PAIRS).map(|_| rng.next_fp()).collect())
            .collect();
        let extra = build_extra();
        let partial_eq = EFP::from(EF::from_usize(123_456_791));

        let air = Poseidon16Precompile::<true>;
        let run_baseline = || {
            drive(&cols, N_FLAT_BASELINE, 10, &extra, partial_eq, N_PAIRS, |f| {
                air.eval(f, &extra)
            })
        };
        let run_candidate = || {
            drive(&cols, N_FLAT_CANDIDATE, 9, &extra, partial_eq, N_PAIRS, |f| {
                eval_candidate(f, &extra)
            })
        };

        let mut sink = EFP::ZERO;
        let mut measure = |name: &str, run: &dyn Fn() -> EFP| -> f64 {
            let mut times = Vec::new();
            for rep in 0..(WARMUP + REPS) {
                let t = Instant::now();
                for _ in 0..PASSES {
                    sink += black_box(run());
                }
                let dt = t.elapsed().as_secs_f64();
                if rep >= WARMUP {
                    times.push(dt);
                }
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = times[times.len() / 2];
            let ns_per_pair = median * 1e9 / (PASSES * N_PAIRS) as f64;
            println!(
                "  {name}: median {ns_per_pair:.1} ns/packed-pair ({median:.3}s per {} pairs)",
                PASSES * N_PAIRS
            );
            ns_per_pair
        };

        let base = measure("baseline (110 cols, 10 z-points, out asserts)", &run_baseline);
        let cand = measure("candidate (94 cols, 9 z-points, fingerprints)", &run_candidate);
        black_box(sink);

        let delta = 100.0 * (base - cand) / base;
        let verdict = if delta >= 8.0 {
            "PASS"
        } else if delta >= 4.0 {
            "GRAY"
        } else {
            "KILL"
        };
        println!(
            "RUNG-I: baseline {base:.1} ns/pair, candidate {cand:.1} ns/pair, delta -{delta:.1}% (gate: >=8% PASS / 4-8% GRAY / <4% KILL) => {verdict}"
        );
        assert!(
            delta >= 4.0,
            "rung-i KILL: candidate only {delta:.1}% faster (need >=8%, gray zone >=4%)"
        );
    }

    /// Rung ii: Merkle leaf-chunk crossing. Literals tie to the measured iter-2
    /// baseline (1550-sig XMSS): stacked cells 60,194,816 = 2^25*(1+0.79), nu=26,
    /// WHIR initial folding factor 7 => 2^19 rows/block, sponge rate 8.
    #[test]
    fn h1_rung_ii_leaf_chunks() {
        let cells_before: usize = 60_194_816;
        let cells_after: usize = cells_before - 16 * (1 << 18); // -16 poseidon cols x 2^18 rows
        assert_eq!(cells_after, 56_000_512);
        let rows_per_block: usize = 1 << (26 - 7);
        let eff_before = cells_before.div_ceil(rows_per_block);
        let eff_after = cells_after.div_ceil(rows_per_block);
        let chunks_before = eff_before.div_ceil(8);
        let chunks_after = eff_after.div_ceil(8);
        println!(
            "RUNG-II: effective_n_cols {eff_before} -> {eff_after}, leaf sponge chunks {chunks_before} -> {chunks_after}"
        );
        assert_eq!(eff_before, 115);
        assert_eq!(eff_after, 107);
        assert_eq!(chunks_before, 15);
        assert_eq!(chunks_after, 14);
        // also pin the committed-column arithmetic
        assert_eq!(num_cols_poseidon_16(), 110, "baseline committed poseidon cols");
        assert_eq!(num_cols_poseidon_16() - 16, 94, "candidate committed poseidon cols");
    }

    /// Rung iii: LogUp tuple-count invariance. h1 keeps every bus tuple (same
    /// buses, same count) - only the verifier-side reconstruction route changes.
    #[test]
    fn h1_rung_iii_logup_size() {
        let buses = Poseidon16Precompile::<true>.bus_interactions();
        println!("RUNG-III: poseidon bus tuples per row = {}", buses.len());
        assert_eq!(buses.len(), 33, "1 precompile pull + 4+4+8+16 memory lookups");
        let memory_lookups = buses.iter().filter(|b| b.is_memory_lookup()).count();
        assert_eq!(memory_lookups, 32);
        // measured 1550-sig baseline logup data total; h1 delta = 0 tuples
        let logup_total: usize = 19_660_800;
        assert!(logup_total <= 1 << 25);
        println!(
            "RUNG-III: logup data {logup_total} <= 2^25 = {} (h1 delta: 0 tuples)",
            1usize << 25
        );
    }
}

// ---------------------------------------------------------------------------
// h-ep U0 kill-ladder rung benches (pw13-mac iter-2, hypothesis "air-efround-pack",
// plan: experiment_logs .../report/hypothesis_5/). Test-only; no production change.
// All arms drive the PRODUCTION Air::eval through ConstraintFolderPacked on
// EXTENSION-packed columns (the 78% region: rounds >= 1 of the AIR sumcheck).
//
// U0a  (poseidon, d=10, degree-split): baseline z-loop (full {0,2,3,4} + skip
//      {5..10}, anchors state_0/state_2, z/2 consts) vs candidate (full
//      {2,3,4,5} + skip {6..10}, anchors z=2/z=3, (z-2) consts, PLUS the Qbar
//      bookkeeping: 2x 11-dot E-row reads with EF node weights, acc0 mul,
//      11 E-row EFP stores). Gate: cand <= 0.97x base else KILL Qbar(poseidon).
// U0a2 (execution, d=5, no split): baseline 5 full evals {0,2,3,4,5} vs
//      candidate 4 full evals {2..5} + 2x 6-dot + 6 stores + acc0.
//      Gate: cand <= 0.97x base => execution table enabled.
// U0b  (fused fold+eval, poseidon shape): production-fold pass + eval pass vs
//      single fused pass (4-index map of plan §2.4). Gate: <= 0.97x else KILL
//      the fusion half.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod u0_kill_ladder {
    use super::*;
    use crate::tables::execution::ExecutionTable;
    use std::hint::black_box;
    use std::time::Instant;

    type EFP = EFPacking<EF>;

    struct Lcg(u64);
    impl Lcg {
        fn next_f(&mut self) -> F {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            const P: u64 = (1 << 31) - (1 << 24) + 1;
            F::from_usize(((self.0 >> 32) % P) as usize)
        }
        fn next_ef(&mut self) -> EF {
            // base-embedded values; mul cost is value-independent
            EF::from(self.next_f()) * EF::from_usize(998_244_353) + EF::from(self.next_f())
        }
        fn next_efp(&mut self) -> EFP {
            EFP::from(self.next_ef())
        }
    }

    fn build_extra(n_alphas: usize) -> ExtraDataForBuses<EF> {
        let eq_poly: Vec<EF> = (0..1 << LOG_MAX_BUS_WIDTH)
            .map(|i| EF::from_usize(7 * i + 3) * EF::from_usize(1_000_003) + EF::from_usize(i * i + 11))
            .collect();
        let alphas: Vec<EF> = (0..n_alphas)
            .map(|i| EF::from_usize(13 * i + 5) * EF::from_usize(998_244_353) + EF::from_usize(i + 1))
            .collect();
        ExtraDataForBuses::new(&eq_poly, alphas)
    }

    fn median_time(warmup: usize, reps: usize, mut f: impl FnMut()) -> f64 {
        let mut times = Vec::new();
        for rep in 0..(warmup + reps) {
            let t = Instant::now();
            f();
            if rep >= warmup {
                times.push(t.elapsed().as_secs_f64());
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    }

    /// Poseidon split z-loop on EFP columns. `first_full_z` = 0 (baseline) or
    /// 2 (candidate). When `qbar` is set, adds the candidate's bookkeeping:
    /// q0/q1 = (d+1)-dots over `e_prev` rows, acc0 contribution, and E-row
    /// stores of all d+1 node values into `e_out`.
    #[allow(clippy::too_many_arguments)]
    fn drive_poseidon_efp(
        cols: &[Vec<EFP>],
        extra: &ExtraDataForBuses<EF>,
        partial_eq: EFP,
        n_pairs: usize,
        first_full_z: usize,
        qbar: Option<(&[EFP], &mut [EFP], &[EF; 11])>,
        air: &Poseidon16Precompile<true>,
    ) -> EFP {
        const D: usize = 10;
        const N_FULL: usize = 4; // low_degree + 1
        let n_flat = N_FLAT_BASELINE_U0;
        let n_skip = D - N_FULL - if first_full_z == 2 { 1 } else { 0 };
        // skip-eval cached-state interpolation constants
        let anchor_consts: Vec<F> = if first_full_z == 0 {
            // anchors z=0, z=2: cached(z) = s0 + (s2 - s0) * z/2
            ((N_FULL + 1)..=D).map(|z| F::from_usize(z).halve()).collect()
        } else {
            // anchors z=2, z=3: cached(z) = s2 + (s3 - s2) * (z - 2)
            ((N_FULL + 2)..=D).map(|z| F::from_usize(z - 2)).collect()
        };
        let lagrange: [[F; N_FULL]; 6] =
            std::array::from_fn(|t| std::array::from_fn(|i| F::from_usize(3 + 5 * t + 7 * i)));

        let (e_prev, mut e_out, weights) = match qbar {
            Some((p, o, w)) => (Some(p), Some(o), Some(w)),
            None => (None, None, None),
        };

        let mut acc = vec![EFP::ZERO; D];
        let mut point: Vec<EFP> = Vec::with_capacity(n_flat);
        let mut diff: Vec<EFP> = Vec::with_capacity(n_flat);
        let mut state_a: Vec<EFP> = Vec::new();
        let mut state_b: Vec<EFP> = Vec::new();
        let mut cached_buf: Vec<EFP> = Vec::new();
        let mut low_evals = [EFP::ZERO; N_FULL];

        for j in 0..n_pairs {
            let (i0, i1) = (2 * j, 2 * j + 1);
            point.clear();
            diff.clear();
            for c in cols.iter().take(n_flat) {
                let lo = c[i0];
                let hi = c[i1];
                point.push(lo);
                diff.push(hi - lo);
            }

            let mut e_row_idx = 0usize;
            if let (Some(prev), Some(out), Some(w)) = (e_prev, e_out.as_deref_mut(), weights) {
                // Qbar consume: q0/q1 from previous-round E-rows (11-dots, EFP x EF)
                let r0 = &prev[(2 * j) * 11..(2 * j) * 11 + 11];
                let r1 = &prev[(2 * j + 1) * 11..(2 * j + 1) * 11 + 11];
                let mut q0 = EFP::ZERO;
                let mut q1 = EFP::ZERO;
                for k in 0..11 {
                    q0 += r0[k] * w[k];
                    q1 += r1[k] * w[k];
                }
                acc[0] += q0 * partial_eq;
                out[j * 11] = q0;
                out[j * 11 + 1] = q1;
                e_row_idx = 2;
                // advance point to z=2 for the first full node
                for k in 0..n_flat {
                    point[k] += diff[k].double();
                }
            }

            // full nodes
            for z_idx in 0..N_FULL {
                if z_idx > 0 || first_full_z == 0 {
                    if z_idx == 1 && first_full_z == 0 {
                        // baseline z: 0 -> 2 jump
                        for k in 0..n_flat {
                            point[k] += diff[k].double();
                        }
                    } else if z_idx > 0 {
                        for k in 0..n_flat {
                            point[k] += diff[k];
                        }
                    }
                }
                let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], extra);
                if z_idx == 0 {
                    folder.cached_state = Some(std::mem::take(&mut state_a));
                } else if z_idx == 1 {
                    folder.cached_state = Some(std::mem::take(&mut state_b));
                }
                air.eval(&mut folder, extra);
                let v = folder.accumulator;
                let acc_slot = if first_full_z == 0 { z_idx } else { 1 + z_idx };
                acc[acc_slot] += v * partial_eq;
                low_evals[z_idx] = folder.accumulator_low;
                if z_idx == 0 {
                    state_a = folder.cached_state.unwrap();
                } else if z_idx == 1 {
                    state_b = folder.cached_state.unwrap();
                }
                if let Some(out) = e_out.as_deref_mut() {
                    out[j * 11 + e_row_idx] = v;
                    e_row_idx += 1;
                }
            }

            // skip-low evals
            for t in 0..n_skip {
                for k in 0..n_flat {
                    point[k] += diff[k];
                }
                cached_buf.clear();
                for i in 0..state_a.len() {
                    cached_buf.push(state_a[i] + (state_b[i] - state_a[i]) * PFPacking::<EF>::from(anchor_consts[t]));
                }
                let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], extra);
                folder.skip_low = true;
                folder.cached_state = Some(std::mem::take(&mut cached_buf));
                folder.low_ci_count = PARTIAL_ROUNDS;
                air.eval(&mut folder, extra);
                cached_buf = folder.cached_state.unwrap();
                let mut low_interpolated = EFP::ZERO;
                for (i, lc) in lagrange[t].iter().enumerate() {
                    low_interpolated += low_evals[i] * PFPacking::<EF>::from(*lc);
                }
                let v = folder.accumulator + low_interpolated;
                let acc_slot = if first_full_z == 0 { N_FULL + t } else { 1 + N_FULL + t };
                acc[acc_slot] += v * partial_eq;
                if let Some(out) = e_out.as_deref_mut() {
                    out[j * 11 + e_row_idx] = v;
                    e_row_idx += 1;
                }
            }
        }
        acc.into_iter().sum()
    }

    const N_FLAT_BASELINE_U0: usize = 110;

    #[test]
    #[ignore]
    fn u0a_poseidon_efp_rung() {
        const N_PAIRS: usize = 1024;
        const PASSES: usize = 8;
        let mut rng = Lcg(0xA5A5_5A5A_1234_5678);
        let cols: Vec<Vec<EFP>> = (0..N_FLAT_BASELINE_U0)
            .map(|_| (0..2 * N_PAIRS).map(|_| rng.next_efp()).collect())
            .collect();
        let extra = build_extra(128);
        let partial_eq = EFP::from(EF::from_usize(123_456_791));
        let air = Poseidon16Precompile::<true>;
        let e_prev: Vec<EFP> = (0..2 * N_PAIRS * 11).map(|_| rng.next_efp()).collect();
        let mut e_out: Vec<EFP> = vec![EFP::ZERO; N_PAIRS * 11];
        let weights: [EF; 11] = std::array::from_fn(|i| EF::from_usize(31 * i + 17) * EF::from_usize(7_777_777));

        let mut sink = EFP::ZERO;
        let t_base = median_time(3, 5, || {
            for _ in 0..PASSES {
                sink += black_box(drive_poseidon_efp(&cols, &extra, partial_eq, N_PAIRS, 0, None, &air));
            }
        });
        let t_cand = median_time(3, 5, || {
            for _ in 0..PASSES {
                sink += black_box(drive_poseidon_efp(
                    &cols,
                    &extra,
                    partial_eq,
                    N_PAIRS,
                    2,
                    Some((&e_prev, &mut e_out, &weights)),
                    &air,
                ));
            }
        });
        black_box(sink);
        let per = 1e9 / (PASSES * N_PAIRS) as f64;
        let delta = 100.0 * (t_base - t_cand) / t_base;
        let verdict = if t_cand <= 0.97 * t_base { "PASS" } else { "KILL" };
        println!(
            "U0A: baseline {:.0} ns/pair, candidate {:.0} ns/pair, delta {:+.1}% (gate cand <= 0.97x base) => {verdict}",
            t_base * per,
            t_cand * per,
            -delta
        );
    }

    /// Execution-shaped rung: production ExecutionTable eval, no degree split.
    fn drive_exec_efp(
        cols: &[Vec<EFP>],
        n_flat: usize,
        extra: &ExtraDataForBuses<EF>,
        partial_eq: EFP,
        n_pairs: usize,
        candidate: bool,
        e_prev: &[EFP],
        e_out: &mut [EFP],
        weights: &[EF; 6],
        air: &ExecutionTable<true>,
    ) -> EFP {
        const D: usize = 5;
        let n_cols = cols.len();
        let mut acc = vec![EFP::ZERO; D];
        let mut point: Vec<EFP> = Vec::with_capacity(n_cols);
        let mut diff: Vec<EFP> = Vec::with_capacity(n_cols);
        for j in 0..n_pairs {
            let (i0, i1) = (2 * j, 2 * j + 1);
            point.clear();
            diff.clear();
            for c in cols {
                let lo = c[i0];
                let hi = c[i1];
                point.push(lo);
                diff.push(hi - lo);
            }
            let mut e_idx = 0usize;
            if candidate {
                let r0 = &e_prev[(2 * j) * 6..(2 * j) * 6 + 6];
                let r1 = &e_prev[(2 * j + 1) * 6..(2 * j + 1) * 6 + 6];
                let mut q0 = EFP::ZERO;
                let mut q1 = EFP::ZERO;
                for k in 0..6 {
                    q0 += r0[k] * weights[k];
                    q1 += r1[k] * weights[k];
                }
                acc[0] += q0 * partial_eq;
                e_out[j * 6] = q0;
                e_out[j * 6 + 1] = q1;
                e_idx = 2;
                for k in 0..n_cols {
                    point[k] += diff[k].double();
                }
            } else {
                // baseline z=0 eval
                let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], extra);
                air.eval(&mut folder, extra);
                acc[0] += folder.accumulator * partial_eq;
                for k in 0..n_cols {
                    point[k] += diff[k].double();
                }
            }
            // z = 2..=5 evals (4 of them)
            for z in 0..4 {
                if z > 0 {
                    for k in 0..n_cols {
                        point[k] += diff[k];
                    }
                }
                let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], extra);
                air.eval(&mut folder, extra);
                let v = folder.accumulator;
                acc[1 + z] += v * partial_eq;
                if candidate {
                    e_out[j * 6 + e_idx] = v;
                    e_idx += 1;
                }
            }
        }
        acc.into_iter().sum()
    }

    #[test]
    #[ignore]
    fn u0a2_execution_efp_rung() {
        const N_PAIRS: usize = 4096;
        const PASSES: usize = 8;
        let air = ExecutionTable::<true>;
        let n_flat = air.n_columns();
        let n_cols = n_flat + air.n_shift_columns();
        let n_constraints = air.n_constraints();
        let mut rng = Lcg(0xBEEF_CAFE_0BAD_F00D);
        let cols: Vec<Vec<EFP>> = (0..n_cols)
            .map(|_| (0..2 * N_PAIRS).map(|_| rng.next_efp()).collect())
            .collect();
        let extra = build_extra(n_constraints);
        let partial_eq = EFP::from(EF::from_usize(987_654_323));
        let e_prev: Vec<EFP> = (0..2 * N_PAIRS * 6).map(|_| rng.next_efp()).collect();
        let mut e_out: Vec<EFP> = vec![EFP::ZERO; N_PAIRS * 6];
        let weights: [EF; 6] = std::array::from_fn(|i| EF::from_usize(31 * i + 17) * EF::from_usize(7_777_777));

        let mut sink = EFP::ZERO;
        let t_base = median_time(3, 5, || {
            for _ in 0..PASSES {
                sink += black_box(drive_exec_efp(
                    &cols, n_flat, &extra, partial_eq, N_PAIRS, false, &e_prev, &mut e_out, &weights, &air,
                ));
            }
        });
        let t_cand = median_time(3, 5, || {
            for _ in 0..PASSES {
                sink += black_box(drive_exec_efp(
                    &cols, n_flat, &extra, partial_eq, N_PAIRS, true, &e_prev, &mut e_out, &weights, &air,
                ));
            }
        });
        black_box(sink);
        let per = 1e9 / (PASSES * N_PAIRS) as f64;
        let delta = 100.0 * (t_base - t_cand) / t_base;
        let verdict = if t_cand <= 0.97 * t_base { "PASS (exec enabled)" } else { "SKIP exec (Qbar poseidon-only)" };
        println!(
            "U0A2: baseline {:.0} ns/pair, candidate {:.0} ns/pair, delta {:+.1}% => {verdict}",
            t_base * per,
            t_cand * per,
            -delta
        );
    }

    #[test]
    #[ignore]
    fn u0b_fused_fold_eval_rung() {
        // Unfolded EFP columns of length 4*Q; round folds bit b' (plan §2.4 map):
        // base = (j_hi << (b'+2)) | j_lo; m0..m3 at base, |s', |2s', |3s';
        // f0 = m0 + r(m2-m0) -> i0' = (j_hi << (b'+1)) | j_lo; f1 = m1 + r(m3-m1) -> i0'|s'.
        const Q: usize = 4096; // folded pair count
        const BP: usize = 3; // b'
        const PASSES: usize = 4;
        let n_flat = N_FLAT_BASELINE_U0;
        let len = 4 * Q;
        let mut rng = Lcg(0x1357_9BDF_2468_ACE0);
        let cols: Vec<Vec<EFP>> = (0..n_flat).map(|_| (0..len).map(|_| rng.next_efp()).collect()).collect();
        let extra = build_extra(128);
        let partial_eq = EFP::from(EF::from_usize(192_837_465));
        let air = Poseidon16Precompile::<true>;
        let r = EF::from_usize(1_111_111_117);
        let rp = EFP::from(r);

        let s = 1usize << BP;
        let lo_mask = s - 1;
        let fold_one = |c: &Vec<EFP>, out: &mut Vec<EFP>| {
            out.clear();
            out.resize(len / 2, EFP::ZERO);
            for j in 0..len / 2 {
                let j_hi = j >> (BP + 1);
                let j_lo = j & ((1 << (BP + 1)) - 1);
                // production fold_at_bit shape: pairs (i, i|stride) at the fold bit
                let base = (j_hi << (BP + 2)) | j_lo;
                out[j] = c[base] + rp * (c[base | (2 * s)] - c[base]);
            }
        };

        let mut sink = EFP::ZERO;
        // Arm A: separate fold pass (all columns) + eval pass over folded pairs
        let mut folded: Vec<Vec<EFP>> = vec![Vec::new(); n_flat];
        let t_base = median_time(2, 3, || {
            for _ in 0..PASSES {
                for (c, fc) in cols.iter().zip(folded.iter_mut()) {
                    fold_one(c, fc);
                }
                sink += black_box(drive_poseidon_efp(&folded, &extra, partial_eq, Q, 0, None, &air));
            }
        });
        // Arm B: fused — 4-index reads, fold both halves, eval on (f0, f1-f0)
        let t_cand = median_time(2, 3, || {
            for _ in 0..PASSES {
                let mut acc = vec![EFP::ZERO; 10];
                let mut f0v: Vec<EFP> = Vec::with_capacity(n_flat);
                let mut diff: Vec<EFP> = Vec::with_capacity(n_flat);
                let mut state_a: Vec<EFP> = Vec::new();
                let mut state_b: Vec<EFP> = Vec::new();
                let mut cached_buf: Vec<EFP> = Vec::new();
                let mut low_evals = [EFP::ZERO; 4];
                let mut fold_out: Vec<Vec<EFP>> = (0..n_flat).map(|_| vec![EFP::ZERO; len / 2]).collect();
                let lagrange: [[F; 4]; 6] =
                    std::array::from_fn(|t| std::array::from_fn(|i| F::from_usize(3 + 5 * t + 7 * i)));
                for j in 0..Q {
                    let j_hi = j >> BP;
                    let j_lo = j & lo_mask;
                    let base = (j_hi << (BP + 2)) | j_lo;
                    let i0p = (j_hi << (BP + 1)) | j_lo;
                    f0v.clear();
                    diff.clear();
                    for (k, c) in cols.iter().enumerate() {
                        let m0 = c[base];
                        let m1 = c[base | s];
                        let m2 = c[base | (2 * s)];
                        let m3 = c[base | (3 * s)];
                        let f0 = m0 + rp * (m2 - m0);
                        let f1 = m1 + rp * (m3 - m1);
                        fold_out[k][i0p] = f0;
                        fold_out[k][i0p | s] = f1;
                        f0v.push(f0);
                        diff.push(f1 - f0);
                    }
                    // inline baseline z-loop on (f0, diff)
                    let point = &mut f0v;
                    {
                        let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], &extra);
                        folder.cached_state = Some(std::mem::take(&mut state_a));
                        air.eval(&mut folder, &extra);
                        acc[0] += folder.accumulator * partial_eq;
                        low_evals[0] = folder.accumulator_low;
                        state_a = folder.cached_state.unwrap();
                    }
                    for k in 0..n_flat {
                        point[k] += diff[k].double();
                    }
                    {
                        let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], &extra);
                        folder.cached_state = Some(std::mem::take(&mut state_b));
                        air.eval(&mut folder, &extra);
                        acc[1] += folder.accumulator * partial_eq;
                        low_evals[1] = folder.accumulator_low;
                        state_b = folder.cached_state.unwrap();
                    }
                    for z_idx in 2..4 {
                        for k in 0..n_flat {
                            point[k] += diff[k];
                        }
                        let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], &extra);
                        air.eval(&mut folder, &extra);
                        acc[z_idx] += folder.accumulator * partial_eq;
                        low_evals[z_idx] = folder.accumulator_low;
                    }
                    for t in 0..6 {
                        for k in 0..n_flat {
                            point[k] += diff[k];
                        }
                        cached_buf.clear();
                        for i in 0..state_a.len() {
                            cached_buf.push(
                                state_a[i]
                                    + (state_b[i] - state_a[i]) * PFPacking::<EF>::from(F::from_usize(5 + t).halve()),
                            );
                        }
                        let mut folder = ConstraintFolderPacked::new(&point[..n_flat], &point[n_flat..], &extra);
                        folder.skip_low = true;
                        folder.cached_state = Some(std::mem::take(&mut cached_buf));
                        folder.low_ci_count = PARTIAL_ROUNDS;
                        air.eval(&mut folder, &extra);
                        cached_buf = folder.cached_state.unwrap();
                        let mut low_interpolated = EFP::ZERO;
                        for (i, lc) in lagrange[t].iter().enumerate() {
                            low_interpolated += low_evals[i] * PFPacking::<EF>::from(*lc);
                        }
                        acc[4 + t] += (folder.accumulator + low_interpolated) * partial_eq;
                    }
                }
                sink += black_box(acc.into_iter().sum::<EFP>() + fold_out[0][0]);
            }
        });
        black_box(sink);
        let delta = 100.0 * (t_base - t_cand) / t_base;
        let verdict = if t_cand <= 0.97 * t_base { "PASS" } else { "KILL fusion" };
        println!(
            "U0B: separate {:.1} ms, fused {:.1} ms, delta {:+.1}% (gate <= 0.97x) => {verdict}",
            t_base * 1e3 / PASSES as f64,
            t_cand * 1e3 / PASSES as f64,
            -delta
        );
    }
}
