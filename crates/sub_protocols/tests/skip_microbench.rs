//! Phase-A microbenchmarks for the univariate-skip hypothesis (plan_spec §1.A3).
//!
//! Measures, on this machine:
//!   - tau_t  = ns per ROW of `eval_packed_base` for the execution / poseidon8 AIRs
//!   - rho_t  = eval_packed_extension / eval_packed_base time ratio
//!   - lambda = ns per base mul of an LDE-shaped butterfly kernel (mul + add, packed)
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo test -p sub_protocols --release \
//!     --test skip_microbench -- --ignored --nocapture
//!
//! All numbers are single-threaded; convert to wall-clock with the measured
//! parallel scaling factor of the prover (~6.6x on 8 cores).

use std::hint::black_box;
use std::time::Instant;

use backend::*;
use lean_vm::{EF, ExecutionTable, ExtraDataForBuses, Poseidon8Precompile};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const LOG_ROWS: usize = 16;

fn rand_packed_base(rng: &mut StdRng) -> PFPacking<EF> {
    PFPacking::<EF>::from_fn(|_| rng.random())
}

fn rand_packed_ext(rng: &mut StdRng) -> EFPacking<EF> {
    EFPacking::<EF>::from_basis_coefficients_fn(|_| rand_packed_base(rng))
}

fn bench_air<A>(name: &str, air: &A, rng: &mut StdRng) -> (f64, f64, f64)
where
    A: Air + SumcheckComputation<EF, ExtraData = ExtraDataForBuses<EF>>,
{
    let w = packing_width::<EF>();
    let n_packed = (1usize << LOG_ROWS) / w;
    let n_cols_tot = air.n_columns() + air.n_shift_columns();

    let logup_alphas: Vec<EF> = (0..4).map(|_| rng.random()).collect();
    let eq_poly = eval_eq(&logup_alphas);
    let alpha: EF = rng.random();
    let alpha_powers: Vec<EF> = alpha.powers().collect_n(air.n_constraints());
    let extra = ExtraDataForBuses::new(&eq_poly, alpha_powers);

    // ---- packed base ----
    let pts_base: Vec<Vec<PFPacking<EF>>> = (0..n_packed)
        .map(|_| (0..n_cols_tot).map(|_| rand_packed_base(rng)).collect())
        .collect();
    let reps = 8usize;
    let mut acc = EFPacking::<EF>::ZERO;
    // warmup
    for p in &pts_base {
        acc += air.eval_packed_base(p, &extra);
    }
    let t = Instant::now();
    for _ in 0..reps {
        for p in &pts_base {
            acc += air.eval_packed_base(p, &extra);
        }
    }
    let tau_base = t.elapsed().as_nanos() as f64 / (reps * n_packed * w) as f64;
    black_box(acc);

    // ---- packed extension ----
    let pts_ext: Vec<Vec<EFPacking<EF>>> = (0..n_packed)
        .map(|_| (0..n_cols_tot).map(|_| rand_packed_ext(rng)).collect())
        .collect();
    let mut acc = EFPacking::<EF>::ZERO;
    for p in &pts_ext {
        acc += air.eval_packed_extension(p, &extra);
    }
    let t = Instant::now();
    for _ in 0..reps {
        for p in &pts_ext {
            acc += air.eval_packed_extension(p, &extra);
        }
    }
    let tau_ext = t.elapsed().as_nanos() as f64 / (reps * n_packed * w) as f64;
    black_box(acc);

    let rho = tau_ext / tau_base;
    println!(
        "{name}: cols={n_cols_tot} constraints={} degree={} | tau_base={tau_base:.2} ns/row  \
         tau_ext={tau_ext:.2} ns/row  rho={rho:.2}",
        air.n_constraints(),
        air.degree_air(),
    );
    (tau_base, tau_ext, rho)
}

#[test]
#[ignore]
fn skip_microbench() {
    let mut rng = StdRng::seed_from_u64(42);

    println!("packing width = {}", packing_width::<EF>());

    let exec = ExecutionTable::<true>;
    let (tau_e, _, rho_e) = bench_air("execution", &exec, &mut rng);

    let pos = Poseidon8Precompile::<true>;
    let (tau_p, _, rho_p) = bench_air("poseidon8", &pos, &mut rng);

    // ---- LDE-shaped butterfly kernel: out[i] = a[i] * twiddle + b[i] over
    // 22 columns x 2^20 base elements (packed). lambda = ns per base mul. ----
    let w = packing_width::<EF>();
    let n_cols = 22usize;
    let n_packed = (1usize << 20) / w;
    let cols_a: Vec<Vec<PFPacking<EF>>> = (0..n_cols)
        .map(|_| (0..n_packed).map(|_| rand_packed_base(&mut rng)).collect())
        .collect();
    let cols_b: Vec<Vec<PFPacking<EF>>> = (0..n_cols)
        .map(|_| (0..n_packed).map(|_| rand_packed_base(&mut rng)).collect())
        .collect();
    let twiddle = rand_packed_base(&mut rng);
    let mut out: Vec<PFPacking<EF>> = vec![PFPacking::<EF>::ZERO; n_packed];

    // warmup
    for (a, b) in cols_a.iter().zip(&cols_b) {
        for i in 0..n_packed {
            out[i] = a[i] * twiddle + b[i];
        }
        black_box(&out);
    }
    let reps = 5usize;
    let t = Instant::now();
    for _ in 0..reps {
        for (a, b) in cols_a.iter().zip(&cols_b) {
            for i in 0..n_packed {
                out[i] = a[i] * twiddle + b[i];
            }
            black_box(&out);
        }
    }
    let lambda = t.elapsed().as_nanos() as f64 / (reps * n_cols * n_packed * w) as f64;
    println!("lambda (LDE butterfly, mul+add): {lambda:.3} ns/mul");

    println!();
    println!("SUMMARY: tau_e={tau_e:.2} rho_e={rho_e:.2} tau_p={tau_p:.2} rho_p={rho_p:.2} lambda={lambda:.3}");
}
