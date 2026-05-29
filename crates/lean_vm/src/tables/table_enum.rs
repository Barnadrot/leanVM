use backend::*;

use crate::execution::memory::MemoryAccess;
use crate::*;

pub const N_TABLES: usize = 3;
pub const ALL_TABLES: [Table; N_TABLES] = [Table::execution(), Table::extension_op(), Table::poseidon16()];
pub const MAX_BUS_WIDTH: usize = N_INSTRUCTION_COLUMNS + 2; // + 1 for PC, + 1 for domainsep
pub const LOG_MAX_BUS_WIDTH: usize = log2_ceil_usize(MAX_BUS_WIDTH);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(usize)]
pub enum Table {
    Execution(ExecutionTable<true>),
    ExtensionOp(ExtensionOpPrecompile<true>),
    Poseidon16(Poseidon16Precompile<true>),
}

#[macro_export]
macro_rules! delegate_to_inner {
    // Existing pattern for method calls
    ($self:expr, $method:ident $(, $($arg:expr),*)?) => {
        match $self {
            Self::ExtensionOp(p) => p.$method($($($arg),*)?),
            Self::Poseidon16(p) => p.$method($($($arg),*)?),
            Self::Execution(p) => p.$method($($($arg),*)?),
        }
    };
    // New pattern for applying a macro to the inner value
    ($self:expr => $macro_name:ident) => {
        match $self {
            Table::ExtensionOp(p) => $macro_name!(p),
            Table::Poseidon16(p) => $macro_name!(p),
            Table::Execution(p) => $macro_name!(p),
        }
    };
}

impl Table {
    pub const fn execution() -> Self {
        Self::Execution(ExecutionTable)
    }
    pub const fn extension_op() -> Self {
        Self::ExtensionOp(ExtensionOpPrecompile)
    }
    pub const fn poseidon16() -> Self {
        Self::Poseidon16(Poseidon16Precompile)
    }
    pub fn embed<PF: PrimeCharacteristicRing>(&self) -> PF {
        PF::from_usize(self.index())
    }
    pub const fn index(&self) -> usize {
        unsafe { *(self as *const Self as *const usize) }
    }
}

impl TableT for Table {
    fn name(&self) -> &'static str {
        delegate_to_inner!(self, name)
    }
    fn table(&self) -> Table {
        delegate_to_inner!(self, table)
    }
    fn is_execution_table(&self) -> bool {
        delegate_to_inner!(self, is_execution_table)
    }
    fn bus_interactions(&self) -> Vec<BusInteraction> {
        delegate_to_inner!(self, bus_interactions)
    }
    fn padding_row(&self, zero_vec_ptr: usize, null_hash_ptr: usize, ending_pc: usize) -> Vec<PF<EF>> {
        delegate_to_inner!(self, padding_row, zero_vec_ptr, null_hash_ptr, ending_pc)
    }
    fn execute<M: MemoryAccess>(
        &self,
        arg_a: F,
        arg_b: F,
        arg_c: F,
        args: PrecompileCompTimeArgs<usize>,
        ctx: &mut InstructionContext<'_, M>,
    ) -> Result<(), RunnerError> {
        delegate_to_inner!(self, execute, arg_a, arg_b, arg_c, args, ctx)
    }
    fn n_columns_total(&self) -> usize {
        delegate_to_inner!(self, n_columns_total)
    }
}

impl Air for Table {
    type ExtraData = ();
    fn degree_air(&self) -> usize {
        delegate_to_inner!(self, degree_air)
    }
    fn n_columns(&self) -> usize {
        delegate_to_inner!(self, n_columns)
    }
    fn n_committed_columns(&self) -> usize {
        delegate_to_inner!(self, n_committed_columns)
    }
    fn bytecode_bound_columns(&self) -> Option<std::ops::Range<usize>> {
        delegate_to_inner!(self, bytecode_bound_columns)
    }
    fn memory_bound_columns(&self) -> Vec<(usize, std::ops::Range<usize>)> {
        delegate_to_inner!(self, memory_bound_columns)
    }
    fn memory_shout_columns(&self) -> Vec<(usize, usize)> {
        delegate_to_inner!(self, memory_shout_columns)
    }
    fn n_constraints(&self) -> usize {
        delegate_to_inner!(self, n_constraints)
    }
    fn n_shift_columns(&self) -> usize {
        delegate_to_inner!(self, n_shift_columns)
    }
    fn eval<AB: AirBuilder>(&self, _: &mut AB, _: &Self::ExtraData) {
        unreachable!()
    }
}

pub fn total_air_constraints() -> usize {
    ALL_TABLES.iter().map(|table| table.n_constraints()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_indices() {
        for (i, table) in ALL_TABLES.iter().enumerate() {
            assert_eq!(table.index(), i);
        }
    }

    #[test]
    fn test_max_bus_width() {
        let expected_max_bus_width = ALL_TABLES
            .iter()
            .flat_map(|table| table.bus_interactions())
            .map(|bus| bus.data.len() + 1)
            .max()
            .unwrap();
        assert_eq!(MAX_BUS_WIDTH, expected_max_bus_width);
    }

    #[test]
    fn committed_columns_cover_bus_referenced_columns() {
        for table in ALL_TABLES {
            let n_committed = table.n_committed_columns();
            let n_total = table.n_columns();
            let n_total_with_virtual = table.n_columns_total();

            assert!(
                n_committed <= n_total,
                "table {}: n_committed_columns ({}) > n_columns ({})",
                table.name(),
                n_committed,
                n_total,
            );
            assert!(
                n_total <= n_total_with_virtual,
                "table {}: n_columns ({}) > n_columns_total ({})",
                table.name(),
                n_total,
                n_total_with_virtual,
            );

            for bus in table.bus_interactions() {
                let bus_cols: Vec<ColIndex> = std::iter::once(&bus.domainsep)
                    .chain(bus.data.iter())
                    .filter_map(|entry| entry.column())
                    .collect();

                match &bus.multiplicity {
                    BusMultiplicity::Column(mult_col) => {
                        assert!(
                            *mult_col < n_total_with_virtual,
                            "table {}: Multiplicity::Column bus references multiplicity col {} \
                             but n_columns_total = {}",
                            table.name(),
                            mult_col,
                            n_total_with_virtual,
                        );
                        for &col in &bus_cols {
                            assert!(
                                col < n_total_with_virtual,
                                "table {}: Multiplicity::Column bus references col {} \
                                 but n_columns_total = {}",
                                table.name(),
                                col,
                                n_total_with_virtual,
                            );
                        }
                    }
                    BusMultiplicity::One => {
                        let bc_bound = table.bytecode_bound_columns();
                        let mem_bound = table.memory_bound_columns();
                        for &col in &bus_cols {
                            let is_bytecode_bound =
                                bc_bound.as_ref().is_some_and(|range| range.contains(&col));
                            let is_memory_bound =
                                mem_bound.iter().any(|(_, range)| range.contains(&col));
                            assert!(
                                col < n_committed || is_bytecode_bound || is_memory_bound,
                                "SOUNDNESS: table {}: Multiplicity::One bus references col {} \
                                 which is outside the committed range [0, {}) and not \
                                 bytecode-bound or memory-bound.",
                                table.name(),
                                col,
                                n_committed,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shift_columns_are_committed() {
        for table in ALL_TABLES {
            let n_shift = table.n_shift_columns();
            let n_committed = table.n_committed_columns();
            assert!(
                n_shift <= n_committed,
                "table {}: n_shift_columns ({}) > n_committed_columns ({}). \
                 Shift columns must be committed for WHIR opening at the next-row point.",
                table.name(),
                n_shift,
                n_committed,
            );
        }
    }
}
