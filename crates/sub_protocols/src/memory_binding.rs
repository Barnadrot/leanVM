use backend::*;
use lean_vm::{EF, F, Table, TableT, memory_lookup_groups};
use rayon::prelude::*;
use std::collections::BTreeMap;
use utils::ToUsize;

pub struct MemoryBindingGroup {
    pub addr_col: usize,
    pub value_cols: Vec<usize>,
}

pub fn memory_binding_groups(table: &Table) -> Vec<MemoryBindingGroup> {
    let n_committed = table.n_committed_columns();
    let buses = table.bus_interactions();
    let groups = memory_lookup_groups(&buses);
    groups
        .into_iter()
        .filter(|g| g.value_cols.iter().any(|&c| c >= n_committed))
        .map(|g| MemoryBindingGroup {
            addr_col: g.idx_col,
            value_cols: g.value_cols,
        })
        .collect()
}

pub fn compute_pushforward_hi(
    addr_col: &[F],
    half_bits: usize,
    eq_r: &[EF],
) -> Vec<EF> {
    let s = 1usize << half_bits;
    let mut p_hi = EF::zero_vec(s);
    for (i, &addr) in addr_col.iter().enumerate() {
        let h = addr.to_usize() >> half_bits;
        if h < s {
            p_hi[h] += eq_r[i];
        }
    }
    p_hi
}

pub fn fold_memory_slice(
    memory: &[F],
    eq_s_hi: &[EF],
    half_bits: usize,
) -> Vec<EF> {
    let s = 1usize << half_bits;
    assert!(memory.len() >= s * s);
    assert_eq!(eq_s_hi.len(), s);
    (0..s)
        .into_par_iter()
        .map(|l| {
            let mut acc = EF::ZERO;
            for h in 0..s {
                acc += eq_s_hi[h] * memory[h * s + l];
            }
            acc
        })
        .collect()
}

pub fn compute_q_lo_d2(
    all_groups: &[(Table, Vec<MemoryBindingGroup>)],
    traces: &BTreeMap<Table, lean_vm::TableTrace>,
    eq_rs: &BTreeMap<Table, Vec<EF>>,
    eq_s_hi: &[EF],
    gamma: EF,
    half_bits: usize,
) -> Vec<EF> {
    let s = 1usize << half_bits;
    let mut q_lo = EF::zero_vec(s);
    let mut gamma_power = EF::ONE;
    for (table, groups) in all_groups {
        let trace = &traces[table];
        let eq_r = &eq_rs[table];
        for group in groups {
            let addr_col = &trace.columns[group.addr_col];
            let n_values = group.value_cols.len();
            let weighted: Vec<EF> = addr_col
                .par_iter()
                .enumerate()
                .map(|(i, &addr)| {
                    let h = addr.to_usize() >> half_bits;
                    if h < s { eq_r[i] * eq_s_hi[h] } else { EF::ZERO }
                })
                .collect();
            let mut gamma_powers_k = Vec::with_capacity(n_values);
            for _ in 0..n_values {
                gamma_powers_k.push(gamma_power);
                gamma_power *= gamma;
            }
            let chunk_size = 4096.max(addr_col.len() / rayon::current_num_threads());
            let partial_q_los: Vec<Vec<EF>> = addr_col
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let mut local_q = EF::zero_vec(s);
                    let base = chunk_idx * chunk_size;
                    for (ci, &addr) in chunk.iter().enumerate() {
                        let i = base + ci;
                        let w = weighted[i];
                        if w.is_zero() { continue; }
                        let lo = addr.to_usize() & (s - 1);
                        for k in 0..n_values {
                            let target = lo + k;
                            if target < s {
                                local_q[target] += gamma_powers_k[k] * w;
                            }
                        }
                    }
                    local_q
                })
                .collect();
            for partial in &partial_q_los {
                for j in 0..s {
                    q_lo[j] += partial[j];
                }
            }
        }
    }
    q_lo
}

pub fn compute_weighted_batched_val(
    q_lo: &[EF],
    m_slice: &[EF],
) -> EF {
    assert_eq!(q_lo.len(), m_slice.len());
    q_lo.iter().zip(m_slice.iter()).map(|(&q, &m)| q * m).sum()
}

pub fn total_memory_binding_groups() -> usize {
    lean_vm::ALL_TABLES
        .iter()
        .map(|t| memory_binding_groups(t).len())
        .sum()
}

pub fn total_memory_bound_value_cols() -> usize {
    lean_vm::ALL_TABLES
        .iter()
        .flat_map(|t| memory_binding_groups(t))
        .map(|g| g.value_cols.len())
        .sum()
}

pub fn compute_shout_pushforward(
    shout_col: &[F],
    half_bits: usize,
    eq_r: &[EF],
) -> Vec<EF> {
    let s = 1usize << half_bits;
    let mut p = EF::zero_vec(s);
    for (i, &v) in shout_col.iter().enumerate() {
        let j = v.to_usize();
        if j < s {
            p[j] += eq_r[i];
        }
    }
    p
}

pub fn compute_batched_mem(
    memory: &[F],
    gamma: EF,
    n_values: usize,
) -> Vec<EF> {
    let k_size = memory.len();
    (0..k_size)
        .into_par_iter()
        .map(|j| {
            let mut acc = EF::ZERO;
            let mut gp = EF::ONE;
            for k in 0..n_values {
                if j + k < k_size {
                    acc += gp * memory[j + k];
                }
                gp *= gamma;
            }
            acc
        })
        .collect()
}
