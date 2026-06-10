use backend::*;

pub fn compute_batched_bytecode<EF: ExtensionField<PF<EF>>>(
    bytecode_multilinear: &[PF<EF>],
    bytecode_table_size: usize,
    bytecode_stride: usize,
    n_instruction_columns: usize,
    gamma_powers: &[EF],
) -> Vec<EF> {
    parallel::par_map_collect(bytecode_table_size, |j| {
        let mut sum = EF::ZERO;
        for k in 0..n_instruction_columns {
            sum += gamma_powers[k] * bytecode_multilinear[j * bytecode_stride + k];
        }
        sum
    })
}
