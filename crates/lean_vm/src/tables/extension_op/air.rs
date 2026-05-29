use crate::{
    EF, EXT_OP_FLAG_ADD, EXT_OP_FLAG_BE, EXT_OP_FLAG_DOT_PRODUCT, EXT_OP_FLAG_EQ, ExtraDataForBuses, eval_bus_virtual,
    tables::extension_op::{EXT_OP_LEN_MULTIPLIER, ExtensionOpPrecompile},
};
use backend::*;

// Shift columns first, in positions 0..13 (see `n_shift_columns` below).
// Flat-only columns follow.
pub(super) const COL_FLAG_BE: usize = 0;
pub(super) const COL_FLAG_START: usize = 1;
pub(super) const COL_LEN: usize = 2;
pub(super) const COL_FLAG_ADD: usize = 3;
pub(super) const COL_FLAG_DOT_PRODUCT: usize = 4;
pub(super) const COL_FLAG_EQ: usize = 5;
pub(super) const COL_IDX_A: usize = 6;
pub(super) const COL_IDX_B: usize = 7;
/// acc (running accumulator) coordinates (5 columns).
pub(super) const COL_ACC: usize = 8;
// --- flat-only columns ---
pub(super) const COL_IDX_RES: usize = 13;
// d=2 Shout address decomposition (3 groups: idx_a, idx_b, idx_res)
pub const N_EXTENSION_MEMORY_GROUPS: usize = 3;
pub(super) const COL_SHOUT_LO_EXT_START: usize = 14;
pub(super) const COL_SHOUT_HI_EXT_START: usize = 14 + N_EXTENSION_MEMORY_GROUPS;
const N_COMMITTED_EXTENSION: usize = COL_SHOUT_HI_EXT_START + N_EXTENSION_MEMORY_GROUPS; // = 20
/// v_A coordinates (5 columns).
pub(super) const COL_V_A: usize = N_COMMITTED_EXTENSION;
/// v_B coordinates (5 columns).
pub(super) const COL_V_B: usize = N_COMMITTED_EXTENSION + 5;
/// res coordinates (5 columns).
pub(super) const COL_RES: usize = N_COMMITTED_EXTENSION + 10;

// Virtual columns (not explicitely in AIR)
pub(super) const COL_MULTIPLICITY_EXTENSION_OP: usize = N_COMMITTED_EXTENSION + 15;
pub(super) const COL_DOMAINSEP_EXTENSION_OP: usize = N_COMMITTED_EXTENSION + 16;

use backend::quintic_extension::extension::quintic_mul;

#[inline]
fn quintic_mul_air<T: PrimeCharacteristicRing>(a: &[T; 5], b: &[T; 5]) -> [T; 5] {
    quintic_mul(a, b, |x, y| {
        x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3] + x[4] * y[4]
    })
}

impl<const BUS: bool> Air for ExtensionOpPrecompile<BUS> {
    type ExtraData = ExtraDataForBuses<EF>;

    fn n_columns(&self) -> usize {
        N_COMMITTED_EXTENSION + 15 // 20 committed + 15 virtual (v_a + v_b + res)
    }
    fn n_committed_columns(&self) -> usize {
        N_COMMITTED_EXTENSION
    }
    fn memory_bound_columns(&self) -> Vec<(usize, std::ops::Range<usize>)> {
        vec![
            (COL_IDX_A, COL_V_A..COL_V_A + crate::DIMENSION),
            (COL_IDX_B, COL_V_B..COL_V_B + crate::DIMENSION),
            (COL_IDX_RES, COL_RES..COL_RES + crate::DIMENSION),
        ]
    }
    fn memory_shout_columns(&self) -> Vec<(usize, usize)> {
        (0..N_EXTENSION_MEMORY_GROUPS)
            .map(|g| (COL_SHOUT_LO_EXT_START + g, COL_SHOUT_HI_EXT_START + g))
            .collect()
    }
    fn degree_air(&self) -> usize {
        6
    }
    fn n_constraints(&self) -> usize {
        35
    }
    fn n_shift_columns(&self) -> usize {
        COL_ACC + 5
    }

    #[inline]
    fn eval<AB: AirBuilder>(&self, builder: &mut AB, extra_data: &Self::ExtraData) {
        let flat = builder.flat();
        let shift = builder.shift();

        let flag_be = flat[COL_FLAG_BE];
        let flag_start = flat[COL_FLAG_START];
        let flag_add = flat[COL_FLAG_ADD];
        let flag_dot_product = flat[COL_FLAG_DOT_PRODUCT];
        let flag_eq = flat[COL_FLAG_EQ];
        let len = flat[COL_LEN];
        let idx_a = flat[COL_IDX_A];
        let idx_b = flat[COL_IDX_B];

        let v_a: [AB::IF; 5] = std::array::from_fn(|k| flat[COL_V_A + k]);
        let v_b: [AB::IF; 5] = std::array::from_fn(|k| flat[COL_V_B + k]);
        let res: [AB::IF; 5] = std::array::from_fn(|k| flat[COL_RES + k]);
        let acc: [AB::IF; 5] = std::array::from_fn(|k| flat[COL_ACC + k]);

        // Shift columns map 1:1 onto the first 13 columns by convention.
        let flag_be_shift = shift[COL_FLAG_BE];
        let flag_start_shift = shift[COL_FLAG_START];
        let len_shift = shift[COL_LEN];
        let flag_add_shift = shift[COL_FLAG_ADD];
        let flag_dot_product_shift = shift[COL_FLAG_DOT_PRODUCT];
        let flag_eq_shift = shift[COL_FLAG_EQ];
        let idx_a_shift = shift[COL_IDX_A];
        let idx_b_shift = shift[COL_IDX_B];
        let acc_shift: [AB::IF; 5] = std::array::from_fn(|k| shift[COL_ACC + k]);

        let active = flag_add + flag_dot_product + flag_eq;
        let multiplicity = flag_start * active;

        let aux_2 = flag_be * AB::F::from_usize(EXT_OP_FLAG_BE)
            + flag_add * AB::F::from_usize(EXT_OP_FLAG_ADD)
            + flag_dot_product * AB::F::from_usize(EXT_OP_FLAG_DOT_PRODUCT)
            + flag_eq * AB::F::from_usize(EXT_OP_FLAG_EQ)
            + len * AB::F::from_usize(EXT_OP_LEN_MULTIPLIER);

        let idx_r = flat[COL_IDX_RES];

        if BUS {
            eval_bus_virtual::<AB, EF>(builder, extra_data, multiplicity, aux_2, &[idx_a, idx_b, idx_r]);
        } else {
            builder.declare_values(&[multiplicity]);
            builder.declare_values(&[idx_a, idx_b, idx_r, aux_2]);
        }

        let is_ee = -(flag_be - AB::F::ONE);
        let not_start_shift = -(flag_start_shift - AB::F::ONE);

        let v_a_tilde: [AB::IF; 5] = std::array::from_fn(|k| if k == 0 { v_a[0] } else { v_a[k] * is_ee });

        let acc_tail: [AB::IF; 5] = std::array::from_fn(|k| acc_shift[k] * not_start_shift);

        builder.assert_bool(flag_be);
        builder.assert_bool(flag_start);
        builder.assert_bool(flag_add);
        builder.assert_bool(flag_dot_product);
        builder.assert_bool(flag_eq);

        for k in 0..5 {
            builder.assert_zero((acc[k] - (v_a_tilde[k] + v_b[k] + acc_tail[k])) * flag_add);
        }

        let v_a_times_v_b = quintic_mul_air(&v_a_tilde, &v_b);

        for k in 0..5 {
            builder.assert_zero((acc[k] - (v_a_times_v_b[k] + acc_tail[k])) * flag_dot_product);
        }

        let e_eq: [AB::IF; 5] = std::array::from_fn(|k| {
            let base = v_a_times_v_b[k].double() - v_a_tilde[k] - v_b[k];
            if k == 0 { base + AB::F::ONE } else { base }
        });
        let acc_tail_or_one: [AB::IF; 5] = std::array::from_fn(|k| {
            if k == 0 {
                acc_shift[0] * not_start_shift + flag_start_shift
            } else {
                acc_shift[k] * not_start_shift
            }
        });
        let eq_result = quintic_mul_air(&e_eq, &acc_tail_or_one);
        for k in 0..5 {
            builder.assert_zero((acc[k] - eq_result[k]) * flag_eq);
        }

        for k in 0..5 {
            builder.assert_zero((acc[k] - res[k]) * flag_start);
        }

        builder.assert_zero(not_start_shift * (len - len_shift - AB::F::ONE));
        builder.assert_zero(not_start_shift * (flag_be - flag_be_shift));
        builder.assert_zero(not_start_shift * (flag_add - flag_add_shift));
        builder.assert_zero(not_start_shift * (flag_dot_product - flag_dot_product_shift));
        builder.assert_zero(not_start_shift * (flag_eq - flag_eq_shift));
        let a_increment = flag_be + is_ee * AB::F::from_usize(crate::DIMENSION);
        builder.assert_zero(not_start_shift * (idx_a_shift - idx_a - a_increment));
        builder.assert_zero(not_start_shift * (idx_b_shift - idx_b - AB::F::from_usize(crate::DIMENSION)));

        builder.assert_zero(flag_start_shift * (len - AB::F::ONE));
    }
}
