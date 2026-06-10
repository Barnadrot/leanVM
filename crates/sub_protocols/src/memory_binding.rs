use backend::Air;
use lean_vm::{Table, TableT, memory_lookup_groups};

#[derive(Debug)]
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

pub fn total_memory_binding_groups() -> usize {
    lean_vm::ALL_TABLES
        .iter()
        .map(|t| memory_binding_groups(t).len())
        .sum()
}

pub fn total_memory_bound_value_cols() -> usize {
    lean_vm::ALL_TABLES
        .iter()
        .flat_map(memory_binding_groups)
        .map(|g| g.value_cols.len())
        .sum()
}
