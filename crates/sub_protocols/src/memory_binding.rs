use backend::*;
use lean_vm::{EF, F, DIGEST_LEN, Table, TableT, memory_lookup_groups};
use std::collections::BTreeMap;
use utils::{ToUsize, poseidon16_compress_pair};

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

pub fn hash_extension_vec(data: &[EF]) -> [F; DIGEST_LEN] {
    use rayon::prelude::*;
    let base_data = flatten_scalars_to_base(data);
    let chunk_size = DIGEST_LEN;
    let mut leaves: Vec<[F; DIGEST_LEN]> = base_data
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut arr = [F::ZERO; DIGEST_LEN];
            for (i, &v) in chunk.iter().enumerate() {
                arr[i] = v;
            }
            arr
        })
        .collect();
    while leaves.len() > 1 {
        let next_len = (leaves.len() + 1) / 2;
        let mut next: Vec<[F; DIGEST_LEN]> = Vec::with_capacity(next_len);
        for i in 0..leaves.len() / 2 {
            next.push(poseidon16_compress_pair(&leaves[2 * i], &leaves[2 * i + 1]));
        }
        if leaves.len() % 2 == 1 {
            next.push(poseidon16_compress_pair(leaves.last().unwrap(), &[F::ZERO; DIGEST_LEN]));
        }
        leaves = next;
    }
    if leaves.is_empty() { [F::ZERO; DIGEST_LEN] } else { leaves[0] }
}

pub fn compute_batched_q_from_traces(
    all_groups: &[(Table, Vec<MemoryBindingGroup>)],
    traces: &BTreeMap<Table, lean_vm::TableTrace>,
    eq_rs: &BTreeMap<Table, Vec<EF>>,
    gamma: EF,
    memory_size: usize,
) -> Vec<EF> {
    let mut q = EF::zero_vec(memory_size);
    let mut gamma_power = EF::ONE;
    for (table, groups) in all_groups {
        let trace = &traces[table];
        let eq_r = &eq_rs[table];
        for group in groups {
            let addr_col = &trace.columns[group.addr_col];
            let n_values = group.value_cols.len();
            for k in 0..n_values {
                for (i, &addr) in addr_col.iter().enumerate() {
                    let j = addr.to_usize();
                    if j + k < memory_size {
                        q[j + k] += gamma_power * eq_r[i];
                    }
                }
                gamma_power *= gamma;
            }
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
