use backend::*;
use lean_vm::{EF, F, Table, TableT, memory_lookup_groups};
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

pub fn compute_memory_pushforward(
    addr_col: &[F],
    memory_size: usize,
    eq_r: &[EF],
) -> Vec<EF> {
    assert_eq!(addr_col.len(), eq_r.len());
    let mut pushforward = EF::zero_vec(memory_size);
    for (i, &addr) in addr_col.iter().enumerate() {
        let j = addr.to_usize();
        if j < memory_size {
            pushforward[j] += eq_r[i];
        }
    }
    pushforward
}

pub struct MemoryBindingData {
    pub groups: Vec<(Table, Vec<MemoryBindingGroup>)>,
    pub pushforwards: Vec<Vec<EF>>,
    pub eq_rs: BTreeMap<Table, Vec<EF>>,
}

pub fn compute_batched_q(
    pushforwards: &[Vec<EF>],
    group_value_counts: &[usize],
    gamma: EF,
    memory_size: usize,
) -> Vec<EF> {
    let mut q = EF::zero_vec(memory_size);
    let mut gamma_power = EF::ONE;
    for (pf_idx, pushforward) in pushforwards.iter().enumerate() {
        let n_values = group_value_counts[pf_idx];
        for k in 0..n_values {
            for (j, &pf_val) in pushforward.iter().enumerate() {
                if j + k < memory_size && !pf_val.is_zero() {
                    q[j + k] += gamma_power * pf_val;
                }
            }
            gamma_power *= gamma;
        }
    }
    q
}

pub fn compute_batched_val(
    columns_values: &BTreeMap<Table, BTreeMap<usize, EF>>,
    groups: &[(Table, Vec<MemoryBindingGroup>)],
    gamma: EF,
) -> EF {
    let mut batched = EF::ZERO;
    let mut gamma_power = EF::ONE;
    for (table, table_groups) in groups {
        let table_vals = &columns_values[table];
        for group in table_groups {
            for &val_col in &group.value_cols {
                batched += gamma_power * table_vals[&val_col];
                gamma_power *= gamma;
            }
        }
    }
    batched
}
