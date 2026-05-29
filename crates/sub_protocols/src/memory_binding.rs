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
            let gamma_base = gamma_power;
            let mut gamma_powers_k = Vec::with_capacity(n_values);
            for _ in 0..n_values {
                gamma_powers_k.push(gamma_power);
                gamma_power *= gamma;
            }
            for (i, &addr) in addr_col.iter().enumerate() {
                if weighted[i].is_zero() { continue; }
                let lo = addr.to_usize() & (s - 1);
                let w = weighted[i];
                for k in 0..n_values {
                    let target = lo + k;
                    if target < s {
                        q_lo[target] += gamma_powers_k[k] * w;
                    }
                }
            }
            let _ = gamma_base;
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
