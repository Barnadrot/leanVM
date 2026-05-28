//! Bytecode binding sumcheck (LOGUP* approach, eprint 2025/946).
//!
//! Proves that instruction column evaluations at a random point r are
//! consistent with the committed PC column and the known bytecode table.
//!
//! The pushforward P[j] = Σ_{i: PC[i]=j} eq_r[i] is a vector of m (bytecode
//! table size) extension field elements. It satisfies:
//!
//!   instruction_col_k(r) = Σ_j P[j] × bytecode[j, k]  for each k
//!
//! Well-formedness of P (that it's correctly derived from the committed PC
//! column) is proved via a GKR-style sumcheck.

use backend::*;
use lean_vm::{EF, F};
use rayon::prelude::*;
use utils::ToUsize;

/// Compute the pushforward: P[j] = Σ_{i: PC[i]=j} eq_r[i]
/// where eq_r[i] = Π_bit (r_bit * i_bit + (1-r_bit)*(1-i_bit))
pub fn compute_pushforward<EF: ExtensionField<PF<EF>>>(
    pc_column: &[PF<EF>],
    bytecode_table_size: usize,
    eq_evals: &[EF],
) -> Vec<EF> {
    assert_eq!(pc_column.len(), eq_evals.len());
    let mut pushforward = EF::zero_vec(bytecode_table_size);
    for (i, &pc_val) in pc_column.iter().enumerate() {
        let j = pc_val.to_usize();
        if j < bytecode_table_size {
            pushforward[j] += eq_evals[i];
        }
    }
    pushforward
}

/// Derive instruction column evaluations from pushforward + bytecode table.
/// instruction_col_k(r) = Σ_j P[j] × bytecode[j, k]
pub fn derive_instruction_evals<EF: ExtensionField<PF<EF>>>(
    pushforward: &[EF],
    bytecode_multilinear: &[PF<EF>],
    n_instruction_columns: usize,
    bytecode_stride: usize,
) -> Vec<EF> {
    let m = pushforward.len();
    (0..n_instruction_columns)
        .map(|k| {
            let mut sum = EF::ZERO;
            for j in 0..m {
                sum += pushforward[j] * bytecode_multilinear[j * bytecode_stride + k];
            }
            sum
        })
        .collect()
}

/// Prove the LOGUP* binding equation via GKR:
///   Σ_i eq_r[i] / (c - PC[i]) = Σ_j P[j] / (c - j)
///
/// Uses the existing quotient GKR infrastructure with extension-field numerators.
/// The left side (execution table) and right side (bytecode table) are proved
/// as separate GKR quotients. The verifier checks their sum is zero.
///
/// Returns the two GKR quotients and their evaluation points.
pub fn prove_logup_star_binding(
    prover_state: &mut impl FSProver<EF>,
    eq_r: &[EF],
    pc_column: &[F],
    pushforward: &[EF],
    c: EF,
) -> ((EF, MultilinearPoint<EF>), (EF, MultilinearPoint<EF>)) {
    let n = eq_r.len();
    let m = pushforward.len();
    assert!(n.is_power_of_two());
    assert!(m.is_power_of_two());

    // Build left-side numerators/denominators (execution table)
    let left_nums: Vec<EF> = eq_r.to_vec();
    let left_dens: Vec<EF> = pc_column.iter().map(|&pc| c - EF::from(pc)).collect();

    // Build right-side numerators/denominators (bytecode table)
    let right_nums: Vec<EF> = pushforward.iter().map(|&p| -p).collect();
    let right_dens: Vec<EF> = (0..m).map(|j| c - EF::from(PF::<EF>::from_usize(j))).collect();

    // Pack into EFPacking for the GKR (manual packing since EFPacking doesn't impl PackedValue)
    let w = packing_width::<EF>();
    let pack_ef = |data: &[EF]| -> Vec<EFPacking<EF>> {
        data.chunks_exact(w)
            .map(|chunk| {
                let mut acc = EFPacking::<EF>::ZERO;
                for (lane, &val) in chunk.iter().enumerate() {
                    let mut basis = [F::ZERO; 4];
                    basis[lane] = F::ONE;
                    acc += EFPacking::<EF>::from(val) * EFPacking::<EF>::from(PFPacking::<EF>::from_fn(|l| basis[l]));
                }
                acc
            })
            .collect()
    };
    let left_nums_packed = pack_ef(&left_nums);
    let left_dens_packed = pack_ef(&left_dens);
    let right_nums_packed = pack_ef(&right_nums);
    let right_dens_packed = pack_ef(&right_dens);

    let pivot_left = crate::ENDIANNESS_PIVOT_GKR.min(log2_strict_usize(n));
    let pivot_right = crate::ENDIANNESS_PIVOT_GKR.min(log2_strict_usize(m));

    // Bit-reverse within chunks for the GKR
    let br_packed = |data: &[EFPacking<EF>], n_vars: usize, pivot: usize| -> Vec<EFPacking<EF>> {
        let packed_len = data.len();
        let chunk_packed = 1usize << (pivot - packing_log_width::<EF>());
        let shift = usize::BITS as usize - (pivot - packing_log_width::<EF>());
        let mut out = vec![EFPacking::<EF>::ZERO; packed_len];
        for (c_idx, chunk) in data.chunks(chunk_packed).enumerate() {
            for (i, &val) in chunk.iter().enumerate() {
                let br_i = i.reverse_bits() >> shift;
                out[c_idx * chunk_packed + br_i] = val;
            }
        }
        out
    };

    let left_nums_br = br_packed(&left_nums_packed, log2_strict_usize(n), pivot_left);
    let left_dens_br = br_packed(&left_dens_packed, log2_strict_usize(n), pivot_left);
    let right_nums_br = br_packed(&right_nums_packed, log2_strict_usize(m), pivot_right);
    let right_dens_br = br_packed(&right_dens_packed, log2_strict_usize(m), pivot_right);

    let left = crate::prove_gkr_quotient_ext(prover_state, &left_nums_br, &left_dens_br, pivot_left);
    let right = crate::prove_gkr_quotient_ext(prover_state, &right_nums_br, &right_dens_br, pivot_right);

    debug_assert!(
        (left.0 + right.0).is_zero(),
        "LOGUP* binding: quotient balance failed: left={:?} right={:?}", left.0, right.0
    );

    (left, right)
}

/// Compute the batched bytecode column: Q[j] = Σ_k γ^k × bytecode[j, k]
pub fn compute_batched_bytecode<EF: ExtensionField<PF<EF>>>(
    bytecode_multilinear: &[PF<EF>],
    bytecode_table_size: usize,
    bytecode_stride: usize,
    n_instruction_columns: usize,
    gamma_powers: &[EF],
) -> Vec<EF> {
    (0..bytecode_table_size)
        .into_par_iter()
        .map(|j| {
            let mut sum = EF::ZERO;
            for k in 0..n_instruction_columns {
                sum += gamma_powers[k] * bytecode_multilinear[j * bytecode_stride + k];
            }
            sum
        })
        .collect()
}
