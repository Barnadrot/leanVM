//! T3' ABORT-CHECKPOINT bench (plan_spec v2 §8): the skip-round engine +
//! g-pass at benchmark heights with the REAL AIRs (constraint-eval cost is
//! data-independent, so random column data gives faithful timings).
//!
//! Run (idle box, no other compute):
//!   RUSTFLAGS="-C target-cpu=native" cargo test -p sub_protocols --release \
//!     --test skip_engine_bench -- --ignored --nocapture
//! Single-thread variant: prefix with `taskset -c 0`.
//!
//! Comparison windows (T0, report/iter1_T0_measurements.md):
//!   k=4 replaces rounds 0..3 = 498.5 ms wall; k=5 replaces rounds 0..4 = 624.5 ms.
//! Decision rule: KILL if best-k saving < 80 ms wall.

use backend::*;
use lean_vm::{EF, ExecutionTable, ExtraDataForBuses, F, LOG_MAX_BUS_WIDTH, Poseidon8Precompile};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::time::Instant;
use sub_protocols::{
    SkipAir, SkipComputation, SkipTableInput, compute_shifted_columns, fold_columns_at_x_point,
    prove_air_univariate_skip,
};

const EXEC_LOG_N: usize = 20;
const POS_LOG_N: usize = 18;
const MAX_AIR_DEGREE: usize = 8;

struct BenchTable {
    flat: Vec<ArenaVec<F>>,
    shifted: Vec<ArenaVec<F>>,
    eq_factor: Vec<EF>,
}

impl BenchTable {
    fn random(rng: &mut StdRng, log_n_rows: usize, n_flat: usize, n_shift: usize) -> Self {
        let n_rows = 1usize << log_n_rows;
        let flat: Vec<ArenaVec<F>> =
            (0..n_flat).map(|_| ArenaVec::from_iter((0..n_rows).map(|_| rng.random::<F>()))).collect();
        let refs: Vec<&[F]> = flat.iter().map(|c| c.as_slice()).collect();
        let shifted = compute_shifted_columns(n_shift, &refs);
        let eq_factor: Vec<EF> = (0..log_n_rows).map(|_| rng.random()).collect();
        Self { flat, shifted, eq_factor }
    }

    fn columns(&self) -> Vec<&[F]> {
        self.flat
            .iter()
            .map(|c| c.as_slice())
            .chain(self.shifted.iter().map(|c| c.as_slice()))
            .collect()
    }
}

fn time_engine(tables: &[SkipTableInput<'_, EF>], k: usize, reps: usize) -> f64 {
    let n_uni = (MAX_AIR_DEGREE + 1) * ((1usize << k) - 1) + 1;
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let mut ps = ProverState::<EF, _>::new(*get_poseidon8(), Default::default());
        let start = Instant::now();
        let out = prove_air_univariate_skip(&mut ps, tables, k, n_uni);
        let dt = start.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        best = best.min(dt);
    }
    best * 1e3
}

fn time_gpass(columns: &[&[F]], n: usize, b: usize, rng: &mut StdRng, reps: usize) -> f64 {
    let x_nat: Vec<EF> = (0..n - b).map(|_| rng.random()).collect();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let start = Instant::now();
        let g = fold_columns_at_x_point::<EF>(columns, &x_nat, b);
        let dt = start.elapsed().as_secs_f64();
        std::hint::black_box(&g);
        best = best.min(dt);
    }
    best * 1e3
}

#[test]
#[ignore = "perf checkpoint — run explicitly on an idle box"]
fn skip_engine_checkpoint() {
    let mut rng = StdRng::seed_from_u64(42);

    let exec_air = ExecutionTable::<true>;
    let pos_air = Poseidon8Precompile::<true>;
    let logup_alphas: Vec<EF> = (0..LOG_MAX_BUS_WIDTH).map(|_| rng.random()).collect();
    let logup_eq: ArenaVec<EF> = eval_eq(&logup_alphas);
    let alpha: EF = rng.random();
    let exec_extra = ExtraDataForBuses::new(&logup_eq, alpha.powers().collect_n(exec_air.n_constraints()));
    let pos_extra = ExtraDataForBuses::new(&logup_eq, alpha.powers().collect_n(pos_air.n_constraints()));

    eprintln!("building random benchmark tables (exec 2^{EXEC_LOG_N} x22, poseidon 2^{POS_LOG_N} x111)...");
    let exec_t = BenchTable::random(&mut rng, EXEC_LOG_N, exec_air.n_columns(), exec_air.n_shift_columns());
    let pos_t = BenchTable::random(&mut rng, POS_LOG_N, pos_air.n_columns(), pos_air.n_shift_columns());

    let skip_exec = SkipAir::<EF, _>::new(&exec_air, &exec_extra);
    let skip_pos = SkipAir::<EF, _>::new(&pos_air, &pos_extra);

    let exec_input = || SkipTableInput::<EF> {
        columns: exec_t.columns(),
        eq_factor: exec_t.eq_factor.clone(),
        computation: &skip_exec as &dyn SkipComputation<EF>,
        sum: EF::ZERO,
        non_padded_n_rows: 1 << EXEC_LOG_N,
    };
    let pos_input = || SkipTableInput::<EF> {
        columns: pos_t.columns(),
        eq_factor: pos_t.eq_factor.clone(),
        computation: &skip_pos as &dyn SkipComputation<EF>,
        sum: EF::ZERO,
        non_padded_n_rows: 1 << POS_LOG_N,
    };

    eprintln!("k | exec_engine_ms | both_engine_ms | poseidon_attrib_ms | gpass_exec_ms | gpass_pos_ms | total_ms | window_ms | saved_ms");
    for k in [4usize, 5] {
        // Warmup.
        let _ = time_engine(&[exec_input()], k, 1);

        let t_exec = time_engine(&[exec_input()], k, 3);
        let t_both = time_engine(&[exec_input(), pos_input()], k, 3);
        let t_pos = (t_both - t_exec).max(0.0);

        let g_exec = time_gpass(&exec_t.columns(), EXEC_LOG_N, k, &mut rng, 3);
        let b_pos = k - 2; // p = n_max - n = 2
        let g_pos = time_gpass(&pos_t.columns(), POS_LOG_N, b_pos, &mut rng, 3);

        let total = t_both + g_exec + g_pos;
        let window = if k == 4 { 498.5 } else { 624.5 };
        eprintln!(
            "{k} | {t_exec:.1} | {t_both:.1} | {t_pos:.1} | {g_exec:.1} | {g_pos:.1} | {total:.1} | {window:.1} | {:.1}",
            window - total
        );
    }
}
