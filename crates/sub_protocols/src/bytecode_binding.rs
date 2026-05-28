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
