mod air_sumcheck;

pub use air_sumcheck::*;

mod logup;
pub use logup::*;

mod stacked_pcs;
pub use stacked_pcs::*;

mod quotient_gkr;
pub use quotient_gkr::*;

mod skip_round;
pub use skip_round::*;

pub(crate) const MIN_VARS_FOR_PACKING: usize = 8;
pub const N_VARS_TO_SEND_GKR_COEFFS: usize = 5;

/// Univariate-skip parameter `k` for the batched AIR sumcheck (plan_spec v2
/// §1): the first `k` sumcheck rounds are replaced by ONE univariate round
/// over the multiplicative subgroup of order `2^k`. Chosen at the T3' abort
/// checkpoint (k=5: saved 145.8 ms; k=4 fails the 80 ms bar).
/// Always applicable: `MIN_LOG_N_ROWS_PER_TABLE = 8 >= k`.
pub const AIR_UNIVARIATE_SKIP: usize = 5;

/// Wire length of the skip-round message: `N + 1` coefficients with
/// `N = max_full_degree·(2^k − 1)` (plan_spec v2 §1.1) — height-independent,
/// shared by prover, verifier and the recursion circuit.
#[must_use]
pub const fn air_skip_n_uni_coeffs(max_full_degree: usize, k: usize) -> usize {
    max_full_degree * ((1usize << k) - 1) + 1
}
