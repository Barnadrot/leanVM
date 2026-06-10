mod air_sumcheck;

pub use air_sumcheck::*;

mod logup;
pub use logup::*;

mod stacked_pcs;
pub use stacked_pcs::*;

mod quotient_gkr;
pub use quotient_gkr::*;

pub mod shout_binding;
pub mod memory_binding;
pub mod bytecode_binding;
#[allow(clippy::needless_range_loop, clippy::if_same_then_else)]
pub mod poseidon_gkr;

pub(crate) const MIN_VARS_FOR_PACKING: usize = 8;
pub const N_VARS_TO_SEND_GKR_COEFFS: usize = 5;
