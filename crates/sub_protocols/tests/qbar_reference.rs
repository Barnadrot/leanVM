//! h-ep U0c — transcript-identity / math-reference harness (pw13-mac iter 2).
//!
//! Validates the three load-bearing identities of the "air-efround-pack"
//! hypothesis (report/hypothesis_5/) against the PRODUCTION AirSumcheckSession,
//! on a real Poseidon16 trace (BUS=true alpha shape) with a padded tail:
//!
//! 1. Row-restriction identity: on every row y of a constraint-valid trace,
//!    the composed accumulator C(y) equals its bus-only part Bus(y) — checked
//!    WITHOUT reimplementing the bus, by evaluating the production
//!    `eval_extension` with the full alpha vector vs with alphas zeroed beyond
//!    the two bus slots. Equality row-wise proves every polynomial constraint
//!    vanishes (Gruen-Q precondition, corrected for leanVM's bus batching).
//! 2. Footnote-12 interpolation (Gruen 2024/108 §4.2 fn.12): per pair x, the
//!    univariate q_x(z) = C(r_{<i}, z, x) of degree <= d is reproduced at ANY
//!    point from its values at nodes {0..d} via Lagrange weights M_j(r) —
//!    i.e. next-round q values are exact linear images of the previous round's
//!    raw eval rows.
//! 3. p_evals[0] identity: the production session's reported bare round
//!    polynomial evaluated at 0 matches the reference model's eq-weighted sum
//!    that uses the row-restricted (bus-only) values for z = 0.
//!
//! Any mismatch kills the Qbar half of h-ep at the cheapest rung.

use backend::*;
use lean_vm::{
    EF, ExtraDataForBuses, F, POSEIDON_COL_ADDR_LEFT_HI, POSEIDON_COL_ADDR_LEFT_LO, POSEIDON_COL_FLAG_OUT8,
    POSEIDON_COL_INPUT_START, POSEIDON_COL_MULTIPLICITY, POSEIDON_COL_NU_B, POSEIDON_COL_NU_C, Poseidon16Precompile,
    fill_trace_poseidon_16, num_cols_poseidon_16,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sub_protocols::{AirSumcheckSession, OuterSumcheckSession};

const WIDTH: usize = 16;

fn build_valid_trace(log_n_rows: usize, rng: &mut StdRng) -> Vec<ArenaVec<F>> {
    let n_rows = 1 << log_n_rows;
    let n_cols = num_cols_poseidon_16();
    let mut trace: Vec<ArenaVec<F>> = (0..n_cols).map(|_| ArenaVec::filled(F::ZERO, n_rows)).collect();
    for t in trace.iter_mut().skip(POSEIDON_COL_INPUT_START).take(WIDTH) {
        *t = ArenaVec::from_iter((0..n_rows).map(|_| rng.random()));
    }
    trace[POSEIDON_COL_MULTIPLICITY] = ArenaVec::filled(F::ONE, n_rows);
    trace[POSEIDON_COL_FLAG_OUT8] = ArenaVec::filled(F::ONE, n_rows);
    trace[POSEIDON_COL_ADDR_LEFT_LO] = ArenaVec::filled(F::ZERO, n_rows);
    trace[POSEIDON_COL_ADDR_LEFT_HI] = ArenaVec::filled(F::from_usize(4), n_rows);
    trace[POSEIDON_COL_NU_B] = ArenaVec::from_iter((0..n_rows).map(F::from_usize));
    trace[POSEIDON_COL_NU_C] = ArenaVec::from_iter((0..n_rows).map(|i| F::from_usize(i + 7)));
    fill_trace_poseidon_16(&mut trace);
    trace
}

/// Scalar evaluation of the composed accumulator at one point (row of EF column
/// values), through the PRODUCTION eval path.
fn c_at(air: &Poseidon16Precompile<true>, point: &[EF], extra: &ExtraDataForBuses<EF>) -> EF {
    <Poseidon16Precompile<true> as SumcheckComputation<EF>>::eval_extension(air, point, extra)
}

#[test]
fn u0c_row_restriction_and_footnote12() {
    let log_n_rows: usize = 10; // small enough for O(2^n * d) scalar reference
    let n_rows = 1usize << log_n_rows;
    let n_cols = num_cols_poseidon_16();
    let mut rng = StdRng::seed_from_u64(42);
    let trace = build_valid_trace(log_n_rows, &mut rng);

    let air = Poseidon16Precompile::<true>;
    let n_constraints = air.n_constraints();
    let d = air.degree_air();

    // BUS=true alpha shape: logup_alphas_eq_poly (16 slots) + alpha powers.
    let logup_alphas: Vec<EF> = (0..4).map(|_| rng.random()).collect();
    let logup_alphas_eq = eval_eq(&logup_alphas);
    let alpha: EF = rng.random();
    let alpha_powers: Vec<EF> = alpha.powers().collect_n(n_constraints);
    let extra_full = ExtraDataForBuses::new(&logup_alphas_eq, alpha_powers.clone());
    // Bus-only alphas: zero beyond the two bus slots (indices 0, 1).
    let mut alpha_bus_only = vec![EF::ZERO; n_constraints];
    alpha_bus_only[0] = alpha_powers[0];
    alpha_bus_only[1] = alpha_powers[1];
    let extra_bus_only = ExtraDataForBuses::new(&logup_alphas_eq, alpha_bus_only);

    // --- Identity 1: C(y) == Bus(y) on EVERY row of the valid trace ---
    for y in 0..n_rows {
        let row: Vec<EF> = (0..n_cols).map(|c| EF::from(trace[c][y])).collect();
        let full = c_at(&air, &row, &extra_full);
        let bus_only = c_at(&air, &row, &extra_bus_only);
        assert_eq!(
            full, bus_only,
            "U0c KILL: row {y} has a non-vanishing polynomial constraint (C != Bus)"
        );
    }
    println!("U0C-1 PASS: row restriction C(y) == Bus(y) on all {n_rows} rows");

    // --- Identities 2+3: drive the production session vs a scalar reference ---
    let eq_factor: Vec<EF> = (0..log_n_rows).map(|_| rng.random()).collect();

    // Reference model state: scalar EF columns, folded round by round (top-bit
    // pairing in NATURAL row order — fold_at_bit on the bit-reversed packed
    // storage is exactly this in row space).
    let mut ref_cols: Vec<Vec<EF>> = (0..n_cols)
        .map(|c| (0..n_rows).map(|y| EF::from(trace[c][y])).collect())
        .collect();

    // Production session (sum = brute-force eq-weighted total of C).
    let eq_table = eval_eq(&eq_factor);
    let mut sum = EF::ZERO;
    for y in 0..n_rows {
        let row: Vec<EF> = (0..n_cols).map(|c| ref_cols[c][y]).collect();
        sum += eq_table[y] * c_at(&air, &row, &extra_full);
    }
    let column_refs: Vec<&[F]> = trace.iter().map(|c| c.as_slice()).collect();
    let packed = MleGroupRef::<EF>::Base(column_refs).pack();
    let mut session = AirSumcheckSession::new(
        packed,
        eq_factor.clone(),
        sum,
        air,
        ExtraDataForBuses::new(&logup_alphas_eq, alpha_powers.clone()),
        n_rows,
    );

    let mut eq_left = eq_factor.clone(); // entries consumed from the END per round
    let mut ref_missing = EF::ONE; // mirrors the session's missing_mul_factor
    for round in 0..3 {
        let bare = session.compute_bare_round_poly();

        // Reference: p_bare(z) = missing * sum over pairs x of partial_eq(x) * q_x(z),
        // with the LAST eq coordinate divided out (Gruen bare form). The session
        // folds the LAST variable (LSB): pairs are ADJACENT rows (2x, 2x+1).
        let m = ref_cols[0].len();
        let half = m / 2;
        let eq_rest = eval_eq(&eq_left[..eq_left.len() - 1]);
        // z = 0 via the ROW-RESTRICTED (bus-only) evaluation — the Qbar claim.
        let mut p0_restricted = EF::ZERO;
        // and the general q_x(z) node table for footnote-12.
        let mut node_vals: Vec<Vec<EF>> = vec![Vec::with_capacity(half); d + 1];
        for x in 0..half {
            let lo: Vec<EF> = (0..n_cols).map(|c| ref_cols[c][2 * x]).collect();
            let hi: Vec<EF> = (0..n_cols).map(|c| ref_cols[c][2 * x + 1]).collect();
            p0_restricted += eq_rest[x] * c_at(&air, &lo, &extra_bus_only);
            for z in 0..=d {
                let zf = EF::from_usize(z);
                let pt: Vec<EF> = lo.iter().zip(&hi).map(|(&l, &h)| l + (h - l) * zf).collect();
                node_vals[z].push(c_at(&air, &pt, &extra_full));
            }
        }
        // Identity 3: bare(0) == eq-weighted z=0 sum, and the z=0 sum computed
        // from FULL evals equals the bus-only (restricted) one in round 0.
        let mut p0_full = EF::ZERO;
        for x in 0..half {
            p0_full += eq_rest[x] * node_vals[0][x];
        }
        assert_eq!(
            bare.evaluate(EF::ZERO),
            p0_full * ref_missing,
            "U0c KILL (round {round}): session bare(0) != reference eq-weighted z=0 sum"
        );
        if round == 0 {
            assert_eq!(
                p0_full, p0_restricted,
                "U0c KILL: z=0 full eval != row-restricted eval at round 0"
            );
        }

        // Identity 2 (footnote-12): q_x(r) from the node table equals the
        // direct evaluation on folded columns, for the actual challenge.
        let r: EF = rng.random();
        // Lagrange node weights M_j(r) over nodes {0..d}.
        let nodes: Vec<EF> = (0..=d).map(EF::from_usize).collect();
        let weights: Vec<EF> = (0..=d)
            .map(|j| {
                let mut w = EF::ONE;
                for k in 0..=d {
                    if k != j {
                        w *= (r - nodes[k]) * (nodes[j] - nodes[k]).try_inverse().unwrap();
                    }
                }
                w
            })
            .collect();
        for x in (0..half).step_by((half / 8).max(1)) {
            let interp: EF = (0..=d).map(|j| weights[j] * node_vals[j][x]).sum();
            let folded: Vec<EF> = (0..n_cols)
                .map(|c| ref_cols[c][2 * x] + (ref_cols[c][2 * x + 1] - ref_cols[c][2 * x]) * r)
                .collect();
            let direct = c_at(&air, &folded, &extra_full);
            assert_eq!(
                interp, direct,
                "U0c KILL (round {round}, pair {x}): footnote-12 interpolation != direct eval"
            );
        }

        // Advance both sides with the same challenge.
        session.process_challenge(r, &bare);
        for c in ref_cols.iter_mut() {
            let folded: Vec<EF> = (0..half).map(|x| c[2 * x] + (c[2 * x + 1] - c[2 * x]) * r).collect();
            *c = folded;
        }
        let a = *eq_left.last().unwrap();
        ref_missing *= (EF::ONE - a) * (EF::ONE - r) + a * r;
        eq_left.pop();
    }
    println!("U0C-2/3 PASS: footnote-12 interpolation + p(0) identities hold over 3 production rounds");
}
