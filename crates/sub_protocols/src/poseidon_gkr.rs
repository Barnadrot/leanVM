//! Poseidon GKR with SplitEq optimization.
//!
//! Uses SplitEq to separate the eq linear factor from the sumcheck polynomial,
//! reducing evaluation points from `degree` to `degree-1` per round (25% fewer
//! for degree-4 transitions). The bare polynomial is sent via add_sumcheck_polynomial
//! with eq_alpha, and the verifier reconstructs the full polynomial.

use backend::*;
use lean_vm::{EF, F};
use rayon::prelude::*;

const WIDTH: usize = 16;
const N_TRANSITIONS: usize = 29;

#[allow(clippy::upper_case_acronyms)]
type PEF = EFPacking<EF>;
const PACK_WIDTH: usize = <<F as Field>::Packing as PackedValue>::WIDTH;
const RAYON_CHUNK: usize = 256;
#[allow(clippy::upper_case_acronyms)]
type PBF = <F as Field>::Packing;
const BF_PACK_WIDTH: usize = <PBF as PackedValue>::WIDTH;

struct PoseidonConstants {
    initial_rc: &'static [[F; WIDTH]],
    final_rc: &'static [[F; WIDTH]],
    frc: &'static [F; WIDTH],
    m_i: &'static [[F; WIDTH]; WIDTH],
    first_rows: &'static Vec<[F; WIDTH]>,
    v_vecs: &'static Vec<[F; WIDTH]>,
    scalar_rc: &'static Vec<F>,
}
unsafe impl Send for PoseidonConstants {}
unsafe impl Sync for PoseidonConstants {}

fn poseidon_constants() -> PoseidonConstants {
    PoseidonConstants {
        initial_rc: poseidon1_initial_constants(),
        final_rc: poseidon1_final_constants(),
        frc: poseidon1_sparse_first_round_constants(),
        m_i: poseidon1_sparse_m_i(),
        first_rows: poseidon1_sparse_first_row(),
        v_vecs: poseidon1_sparse_v(),
        scalar_rc: poseidon1_sparse_scalar_round_constants(),
    }
}

#[inline(always)]
fn hsum_pef(p: PEF) -> EF {
    let coeffs: &[FPacking<F>] = p.as_basis_coefficients_slice();
    let mut sum = EF::ZERO;
    for lane in 0..PACK_WIDTH {
        let c: [F; 5] = std::array::from_fn(|d| coeffs[d].as_slice()[lane]);
        sum += EF::from_basis_coefficients_fn(|d| c[d]);
    }
    sum
}

pub fn compute_checkpoints_from_inputs(input_cols: &[&[F]], n_rows: usize) -> Vec<Vec<[F; WIDTH]>> {
    compute_checkpoint_states_base(input_cols, n_rows)
}

#[allow(clippy::uninit_vec)]
fn compute_checkpoint_states_base(input_cols: &[&[F]], n_rows: usize) -> Vec<Vec<[F; WIDTH]>> {
    let c = poseidon_constants();
    let mut checkpoints: Vec<Vec<[F; WIDTH]>> = (0..N_TRANSITIONS + 1)
        .map(|_| unsafe { let mut v = Vec::with_capacity(n_rows); v.set_len(n_rows); v })
        .collect();
    let cp0 = &mut checkpoints[0];
    for i in 0..n_rows { cp0[i] = std::array::from_fn(|k| input_cols[k][i]); }
    (0..n_rows).into_par_iter().for_each(|i| {
        let mut state = checkpoints[0][i];
        for r in 0..4 {
            for k in 0..WIDTH { state[k] += c.initial_rc[r][k]; state[k] = state[k].cube(); }
            mds_circ_16(&mut state);
            unsafe { *checkpoints.get_unchecked(1 + r).as_ptr().add(i).cast_mut() = state; }
        }
        for (s, &rc) in state.iter_mut().zip(c.frc.iter()) { *s += rc; }
        let inp = state;
        for k in 0..WIDTH { state[k] = F::ZERO; for j in 0..WIDTH { state[k] += inp[j] * c.m_i[k][j]; } }
        unsafe { *checkpoints.get_unchecked(5).as_ptr().add(i).cast_mut() = state; }
        for r in 0..20 {
            state[0] = state[0].cube();
            if r < 19 { state[0] += c.scalar_rc[r]; }
            let old_s0 = state[0];
            let mut new_s0 = F::ZERO;
            for j in 0..WIDTH { new_s0 += state[j] * c.first_rows[r][j]; }
            state[0] = new_s0;
            for j in 1..WIDTH { state[j] += old_s0 * c.v_vecs[r][j - 1]; }
            unsafe { *checkpoints.get_unchecked(6 + r).as_ptr().add(i).cast_mut() = state; }
        }
        for r in 0..4 {
            for k in 0..WIDTH { state[k] += c.final_rc[r][k]; state[k] = state[k].cube(); }
            mds_circ_16(&mut state);
            unsafe { *checkpoints.get_unchecked(26 + r).as_ptr().add(i).cast_mut() = state; }
        }
    });
    checkpoints
}

fn apply_transition_to_evals(t: usize, input: &[EF], c: &PoseidonConstants) -> [EF; WIDTH] {
    let mut state: [EF; WIDTH] = std::array::from_fn(|k| input[k]);
    match t {
        0..=3 => { let rc = &c.initial_rc[t]; for k in 0..WIDTH { state[k] += rc[k]; state[k] = state[k].cube(); } mds_circ_16(&mut state); }
        4 => { for (s, &rc) in state.iter_mut().zip(c.frc.iter()) { *s += rc; } let inp = state; for k in 0..WIDTH { state[k] = EF::ZERO; for j in 0..WIDTH { state[k] += inp[j] * c.m_i[k][j]; } } }
        5..=24 => { let r = t - 5; state[0] = state[0].cube(); if r < 19 { state[0] += c.scalar_rc[r]; } let old_s0 = state[0]; let mut new_s0 = EF::ZERO; for j in 0..WIDTH { new_s0 += state[j] * c.first_rows[r][j]; } state[0] = new_s0; for j in 1..WIDTH { state[j] += old_s0 * c.v_vecs[r][j - 1]; } }
        25..=28 => { let rc = &c.final_rc[t - 25]; for k in 0..WIDTH { state[k] += rc[k]; state[k] = state[k].cube(); } mds_circ_16(&mut state); }
        _ => unreachable!(),
    }
    state
}

fn sumcheck_degree(t: usize) -> usize { match t { 4 => 2, _ => 4 } }
fn bare_degree(t: usize) -> usize { sumcheck_degree(t) - 1 }

struct TransitionPrecomp {
    eq_el: [PEF; WIDTH],
    cw: PEF,
    weights: [PEF; WIDTH],
    mds_t_eq: [PEF; WIDTH],
    mi_t_eq: [PEF; WIDTH],
}

impl TransitionPrecomp {
    fn new(eq_p_el: &[EF], t: usize, c: &PoseidonConstants) -> Self {
        let eq_el: [PEF; WIDTH] = std::array::from_fn(|k| PEF::from(eq_p_el[k]));
        let (cw, weights) = if (5..=24).contains(&t) {
            let r = t - 5;
            let mut cw = eq_el[0] * c.first_rows[r][0];
            for k in 1..WIDTH { cw += eq_el[k] * c.v_vecs[r][k - 1]; }
            let weights: [PEF; WIDTH] = std::array::from_fn(|k| if k == 0 { PEF::default() } else { eq_el[k] + eq_el[0] * c.first_rows[r][k] });
            (cw, weights)
        } else { (PEF::default(), [PEF::default(); WIDTH]) };
        const MDS_COL: [F; WIDTH] = F::new_array([1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1]);
        let mds_t_eq: [PEF; WIDTH] = std::array::from_fn(|k| {
            let mut acc = PEF::default();
            for j in 0..WIDTH { acc += eq_el[j] * MDS_COL[(j + WIDTH - k) % WIDTH]; }
            acc
        });
        let mi_t_eq: [PEF; WIDTH] = if t == 4 {
            let mi = poseidon1_sparse_m_i();
            std::array::from_fn(|k| { let mut acc = PEF::default(); for j in 0..WIDTH { acc += eq_el[j] * mi[j][k]; } acc })
        } else { [PEF::default(); WIDTH] };
        Self { eq_el, cw, weights, mds_t_eq, mi_t_eq }
    }
}

/// Bare eval for base-field round with pre-packed eq_remaining
#[inline(always)]
fn bare_eval_base_direct(prev: &[[F; WIDTH]], eq_rem_p: PEF, pre: &TransitionPrecomp, start: usize, t: usize, n_evals: usize, c: &PoseidonConstants) -> [PEF; 5] {
    let mut a_bf = [PBF::default(); WIDTH]; let mut d_bf = [PBF::default(); WIDTH];
    for k in 0..WIDTH {
        a_bf[k] = *PBF::from_slice(&std::array::from_fn::<F, BF_PACK_WIDTH, _>(|i| prev[2 * (start + i)][k]));
        d_bf[k] = *PBF::from_slice(&std::array::from_fn::<F, BF_PACK_WIDTH, _>(|i| prev[2 * (start + i) + 1][k])) - a_bf[k];
    }
    let mut result = [PEF::default(); 5];
    if (5..=24).contains(&t) {
        let r = t - 5;
        let mut lc = PEF::default(); let mut ls = PEF::default();
        for k in 1..WIDTH { lc += pre.weights[k] * a_bf[k]; ls += pre.weights[k] * d_bf[k]; }
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let s0_bf = a_bf[0] + d_bf[0] * pt;
            let cubed_bf = s0_bf * s0_bf * s0_bf;
            let mut cubed_pef = PEF::from(cubed_bf);
            if r < 19 { cubed_pef += c.scalar_rc[r]; }
            result[idx] = eq_rem_p * ((lc + ls * pt) + pre.cw * cubed_pef);
        }
    } else if t <= 3 || t >= 25 {
        let rc = match t { 0..=3 => &c.initial_rc[t], 25..=28 => &c.final_rc[t - 25], _ => unreachable!() };
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let mut w = PEF::default();
            for k in 0..WIDTH { let s = a_bf[k] + d_bf[k] * pt + rc[k]; w += pre.mds_t_eq[k] * (s * s * s); }
            result[idx] = eq_rem_p * w;
        }
    } else {
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let mut w = PEF::default();
            for k in 0..WIDTH { w += pre.mi_t_eq[k] * (a_bf[k] + d_bf[k] * pt + c.frc[k]); }
            result[idx] = eq_rem_p * w;
        }
    }
    result
}

/// Bare polynomial evaluation (WITHOUT eq linear factor) for EF round.
#[inline(always)]
fn bare_eval_ef(folded_prev: &[[EF; WIDTH]], eq_table: &[EF], pre: &TransitionPrecomp, start: usize, t: usize, n_evals: usize, c: &PoseidonConstants) -> [PEF; 5] {
    let eq_rem_p = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i|
        eq_table[2 * (start + i)] + eq_table[2 * (start + i) + 1]
    ));
    let mut a_p = [PEF::default(); WIDTH]; let mut d_p = [PEF::default(); WIDTH];
    for k in 0..WIDTH {
        a_p[k] = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| folded_prev[2 * (start + i)][k]));
        d_p[k] = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| folded_prev[2 * (start + i) + 1][k])) - a_p[k];
    }
    let mut result = [PEF::default(); 5];
    if (5..=24).contains(&t) {
        let r = t - 5;
        let mut lc = PEF::default(); let mut ls = PEF::default();
        for k in 1..WIDTH { lc += pre.weights[k] * a_p[k]; ls += pre.weights[k] * d_p[k]; }
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let s0 = a_p[0] + d_p[0] * pt;
            let mut cubed = s0 * s0 * s0;
            if r < 19 { cubed += c.scalar_rc[r]; }
            result[idx] = eq_rem_p * ((lc + ls * pt) + pre.cw * cubed);
        }
    } else if t <= 3 || t >= 25 {
        let rc = match t { 0..=3 => &c.initial_rc[t], 25..=28 => &c.final_rc[t - 25], _ => unreachable!() };
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let mut w = PEF::default();
            for k in 0..WIDTH { let s = a_p[k] + d_p[k] * pt + rc[k]; w += pre.mds_t_eq[k] * (s * s * s); }
            result[idx] = eq_rem_p * w;
        }
    } else {
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let mut w = PEF::default();
            for k in 0..WIDTH { w += pre.mi_t_eq[k] * (a_p[k] + d_p[k] * pt + c.frc[k]); }
            result[idx] = eq_rem_p * w;
        }
    }
    result
}

/// Scalar fallback for bare evaluation
fn bare_eval_scalar(a: &[EF; WIDTH], b: &[EF; WIDTH], eq_rem: EF, eq_p_el: &[EF], t: usize, n_evals: usize, c: &PoseidonConstants) -> [EF; 5] {
    let mut diffs = [EF::ZERO; WIDTH];
    for k in 0..WIDTH { diffs[k] = b[k] - a[k]; }
    let mut result = [EF::ZERO; 5];
    for idx in 0..n_evals {
        let point = if idx == 0 { 0 } else { idx + 1 };
        let pt = F::from_usize(point);
        let prev_interp: [EF; WIDTH] = std::array::from_fn(|k| a[k] + diffs[k] * pt);
        let out = apply_transition_to_evals(t, &prev_interp, c);
        let mut w = EF::ZERO;
        for k in 0..WIDTH { w += eq_p_el[k] * out[k]; }
        result[idx] = eq_rem * w;
    }
    result
}

fn evals_to_coeffs(evals: &[EF]) -> Vec<EF> {
    let n = evals.len();
    let mut coeffs = vec![EF::ZERO; n];
    for i in 0..n {
        let mut basis_coeffs = vec![EF::ZERO; n];
        basis_coeffs[0] = EF::ONE;
        let mut denom = EF::ONE;
        for j in 0..n { if j == i { continue; } denom *= EF::from_usize(i) - EF::from_usize(j); let j_f = EF::from_usize(j); for k in (1..n).rev() { basis_coeffs[k] = basis_coeffs[k - 1] - j_f * basis_coeffs[k]; } basis_coeffs[0] = -j_f * basis_coeffs[0]; }
        let inv_denom = EF::ONE / denom;
        for k in 0..n { coeffs[k] += evals[i] * basis_coeffs[k] * inv_denom; }
    }
    coeffs
}

fn build_bare_from_coeffs(c0_raw: EF, c2_raw: EF, eq_alpha: EF, sum: EF, mmf: EF) -> Vec<EF> {
    let c0_mmf = c0_raw * mmf;
    let c2_mmf = c2_raw * mmf;
    let h1_mmf = (sum - (EF::ONE - eq_alpha) * c0_mmf) / eq_alpha;
    let c1_mmf = h1_mmf - c0_mmf - c2_mmf;
    vec![c0_mmf, c1_mmf, c2_mmf]
}

pub fn prove_poseidon_gkr(prover_state: &mut impl FSProver<EF>, input_cols: &[&[F]], n_rows: usize, log_n_rows: usize) -> (MultilinearPoint<EF>, Vec<EF>) {
    prove_poseidon_gkr_precomputed(prover_state, input_cols, n_rows, log_n_rows, None)
}

pub fn prove_poseidon_gkr_precomputed(prover_state: &mut impl FSProver<EF>, input_cols: &[&[F]], n_rows: usize, log_n_rows: usize, precomputed_checkpoints: Option<Vec<Vec<[F; WIDTH]>>>) -> (MultilinearPoint<EF>, Vec<EF>) {
    let c = poseidon_constants();
    let checkpoints = precomputed_checkpoints.unwrap_or_else(|| compute_checkpoint_states_base(input_cols, n_rows));

    prover_state.duplex();
    let p_el: Vec<EF> = prover_state.sample_vec(4);
    let mut eq_p_el = eval_eq(&p_el);
    prover_state.duplex();
    let mut p_row: Vec<EF> = prover_state.sample_vec(log_n_rows);

    let eq_p_row = eval_eq(&p_row);
    let last_cp = N_TRANSITIONS;
    let mut current_claim: EF = (0..n_rows).into_par_iter()
        .map(|i| { let mut rv = EF::ZERO; for k in 0..WIDTH { rv += eq_p_el[k] * checkpoints[last_cp][i][k]; } eq_p_row[i] * rv })
        .sum();
    prover_state.add_extension_scalar(current_claim);

    for t in (0..N_TRANSITIONS).rev() {
        let bd = bare_degree(t);
        let prev = &checkpoints[t];
        let n = 1usize << log_n_rows;
        if t < N_TRANSITIONS - 1 { prover_state.add_extension_scalar(current_claim); }

        // Compute eq_remaining directly (half size): eq_remaining[h] = eval_eq(p_row[0..n-1])[h]
        // This equals eval_eq(p_row)[2h] + eval_eq(p_row)[2h+1], saving 50% of eval_eq work
        let mut eq_table: Vec<EF> = eval_eq(&p_row[..log_n_rows - 1]).to_vec();
        let mut challenges = Vec::with_capacity(log_n_rows);
        let pre = TransitionPrecomp::new(&eq_p_el, t, &c);
        let mut mmf = EF::ONE;

        // Base-field round 0 with SplitEq (uses eq_table as eq_remaining directly)
        {
            let half = n >> 1;
            let n_evals = bd;
            let eq_alpha = p_row[log_n_rows - 1];
            let n_packed_bf = half / BF_PACK_WIDTH;

            let mut raw_evals: Vec<EF> = if n_packed_bf >= RAYON_CHUNK / BF_PACK_WIDTH {
                let packed_sums = (0..n_packed_bf).into_par_iter()
                    .fold(|| [PEF::default(); 5], |mut acc, p| {
                        let start = p * BF_PACK_WIDTH;
                        // eq_table[h] IS eq_remaining[h] — no need to sum pairs
                        let eq_rem_p = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| eq_table[start + i]));
                        let contrib = bare_eval_base_direct(prev, eq_rem_p, &pre, start, t, n_evals, &c);
                        for point in 0..n_evals { acc[point] += contrib[point]; } acc
                    })
                    .reduce(|| [PEF::default(); 5], |mut a, b| { for p in 0..n_evals { a[p] += b[p]; } a });
                let mut evals: Vec<EF> = (0..n_evals).map(|p| hsum_pef(packed_sums[p])).collect();
                for e in evals.iter_mut() { *e *= mmf; }
                evals
            } else {
                let mut evals = vec![EF::ZERO; n_evals];
                for h in 0..half {
                    let a: [EF; WIDTH] = std::array::from_fn(|k| EF::from(prev[2*h][k]));
                    let b: [EF; WIDTH] = std::array::from_fn(|k| EF::from(prev[2*h+1][k]));
                    let contrib = bare_eval_scalar(&a, &b, eq_table[h], &eq_p_el, t, n_evals, &c);
                    for point in 0..n_evals { evals[point] += contrib[point]; }
                }
                for e in evals.iter_mut() { *e *= mmf; }
                evals
            };
            let bare_p_at_1 = (current_claim - (EF::ONE - eq_alpha) * raw_evals[0]) / eq_alpha;
            raw_evals.insert(1, bare_p_at_1);
            let bare_coeffs = evals_to_coeffs(&raw_evals);
            prover_state.add_sumcheck_polynomial(&bare_coeffs, Some(eq_alpha));
            let r_v: EF = prover_state.sample();
            challenges.push(r_v);
            let eq_eval = (EF::ONE - eq_alpha) * (EF::ONE - r_v) + eq_alpha * r_v;
            current_claim = eq_eval * bare_coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r_v + c);
            mmf *= eq_eval;
            // eq_table is already at size n/2 — no fold needed (it was computed as eval_eq of n-1 vars)
        }

        // Fold from F to EF
        let r0 = challenges[0];
        let mut folded_prev: Vec<[EF; WIDTH]> = prev.par_chunks_exact(2).map(|pair| {
            std::array::from_fn(|k| EF::from(pair[0][k]) + EF::from(pair[1][k] - pair[0][k]) * r0)
        }).collect();

        // EF rounds with SplitEq
        for v in 1..log_n_rows {
            let half = n >> (v + 1);
            let n_evals = bd;
            let eq_alpha = p_row[log_n_rows - 1 - v];
            let n_packed = half / PACK_WIDTH;

            let mut raw_evals: Vec<EF> = if n_packed >= RAYON_CHUNK / PACK_WIDTH {
                let packed_sums = (0..n_packed).into_par_iter()
                    .fold(|| [PEF::default(); 5], |mut acc, p| {
                        let contrib = bare_eval_ef(&folded_prev, &eq_table, &pre, p * PACK_WIDTH, t, n_evals, &c);
                        for point in 0..n_evals { acc[point] += contrib[point]; } acc
                    })
                    .reduce(|| [PEF::default(); 5], |mut a, b| { for p in 0..n_evals { a[p] += b[p]; } a });
                let mut evals: Vec<EF> = (0..n_evals).map(|p| hsum_pef(packed_sums[p])).collect();
                for h in (n_packed * PACK_WIDTH)..half {
                    let eq_rem = eq_table[2*h] + eq_table[2*h+1];
                    let contrib = bare_eval_scalar(&folded_prev[2*h], &folded_prev[2*h+1], eq_rem, &eq_p_el, t, n_evals, &c);
                    for point in 0..n_evals { evals[point] += contrib[point]; }
                }
                for e in evals.iter_mut() { *e *= mmf; }
                evals
            } else if half >= 4 {
                let mut evals = vec![EF::ZERO; n_evals];
                for h in 0..half {
                    let eq_rem = eq_table[2*h] + eq_table[2*h+1];
                    let contrib = bare_eval_scalar(&folded_prev[2*h], &folded_prev[2*h+1], eq_rem, &eq_p_el, t, n_evals, &c);
                    for point in 0..n_evals { evals[point] += contrib[point]; }
                }
                for e in evals.iter_mut() { *e *= mmf; }
                evals
            } else {
                let mut evals = vec![EF::ZERO; n_evals];
                for h in 0..half {
                    let eq_rem = eq_table[2*h] + eq_table[2*h+1];
                    let contrib = bare_eval_scalar(&folded_prev[2*h], &folded_prev[2*h+1], eq_rem, &eq_p_el, t, n_evals, &c);
                    for point in 0..n_evals { evals[point] += contrib[point]; }
                }
                for e in evals.iter_mut() { *e *= mmf; }
                evals
            };

            let bare_p_at_1 = (current_claim - (EF::ONE - eq_alpha) * raw_evals[0]) / eq_alpha;
            raw_evals.insert(1, bare_p_at_1);
            let bare_coeffs = evals_to_coeffs(&raw_evals);
            prover_state.add_sumcheck_polynomial(&bare_coeffs, Some(eq_alpha));
            let r_v: EF = prover_state.sample();
            challenges.push(r_v);
            let eq_eval = (EF::ONE - eq_alpha) * (EF::ONE - r_v) + eq_alpha * r_v;
            current_claim = eq_eval * bare_coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r_v + c);
            mmf *= eq_eval;
            // Fold eq_remaining (sum pairs instead of interpolate)
            if half >= RAYON_CHUNK {
                let new_eq: Vec<EF> = eq_table.par_chunks_exact(2).map(|p| p[0] + p[1]).collect();
                let new_prev: Vec<[EF; WIDTH]> = folded_prev.par_chunks_exact(2)
                    .map(|p| std::array::from_fn(|k| p[0][k] + (p[1][k] - p[0][k]) * r_v)).collect();
                eq_table = new_eq; folded_prev = new_prev;
            } else {
                for h in 0..half {
                    eq_table[h] = eq_table[2*h] + eq_table[2*h+1];
                    for k in 0..WIDTH { folded_prev[h][k] = folded_prev[2*h][k] + (folded_prev[2*h+1][k] - folded_prev[2*h][k]) * r_v; }
                }
                eq_table.truncate(half); folded_prev.truncate(half);
            }
        }
        debug_assert_eq!(folded_prev.len(), 1);
        let inner_evals: Vec<EF> = folded_prev[0].to_vec();
        prover_state.add_extension_scalars(&inner_evals);
        prover_state.duplex();
        let alpha: Vec<EF> = prover_state.sample_vec(4);
        let eq_alpha = eval_eq(&alpha);
        current_claim = EF::ZERO;
        for k in 0..WIDTH { current_claim += eq_alpha[k] * inner_evals[k]; }
        challenges.reverse();
        p_row = challenges;
        eq_p_el = eq_alpha;
    }

    let final_input_evals: Vec<EF> = {
        let eq_final = eval_eq(&p_row);
        (0..WIDTH).map(|k| (0..n_rows).into_par_iter().map(|i| eq_final[i] * input_cols[k][i]).sum()).collect()
    };
    (MultilinearPoint(p_row), final_input_evals)
}

pub fn verify_poseidon_gkr(verifier_state: &mut impl FSVerifier<EF>, log_n_rows: usize) -> Result<(MultilinearPoint<EF>, EF, Vec<EF>), ProofError> {
    let c = poseidon_constants();
    verifier_state.duplex();
    let p_el: Vec<EF> = verifier_state.sample_vec(4);
    let mut eq_p_el = eval_eq(&p_el);
    verifier_state.duplex();
    let mut p_row: Vec<EF> = verifier_state.sample_vec(log_n_rows);
    let mut claimed = verifier_state.next_extension_scalar()?;
    let mut final_inner_evals = Vec::new();

    for t in (0..N_TRANSITIONS).rev() {
        let degree = sumcheck_degree(t);
        if t < N_TRANSITIONS - 1 { claimed = verifier_state.next_extension_scalar()?; }

        // SplitEq: eq_alphas[round] = p_row[log_n_rows-1-round] (LSB-first)
        let eq_alphas: Vec<EF> = (0..log_n_rows).map(|v| p_row[log_n_rows - 1 - v]).collect();
        let sc_result = sumcheck_verify(verifier_state, log_n_rows, degree, claimed, Some(&eq_alphas))?;
        let challenges = sc_result.point.0;
        let final_sum = sc_result.value;

        let inner_evals = verifier_state.next_extension_scalars_vec(WIDTH)?;
        let trans_out = apply_transition_to_evals(t, &inner_evals, &c);
        let mut h_val = EF::ZERO;
        for k in 0..WIDTH { h_val += eq_p_el[k] * trans_out[k]; }
        // With SplitEq, final_sum includes the missing_mul_factor * eq_remaining * h_val
        // The eq_remaining at the end is eq_table[0] = 1 (product of all eq sums collapses)
        // And missing_mul_factor = Π eq_eval(alpha_v, r_v)
        // So final_sum = mmf * h_val. And mmf = eq(p_row, challenges_reversed)
        let mut challenges_rev = challenges.clone(); challenges_rev.reverse();
        let eq_at = MultilinearPoint(p_row.clone()).eq_poly_outside(&MultilinearPoint(challenges_rev));
        if final_sum != eq_at * h_val { return Err(ProofError::InvalidProof); }

        verifier_state.duplex();
        let alpha: Vec<EF> = verifier_state.sample_vec(4);
        eq_p_el = eval_eq(&alpha);
        claimed = EF::ZERO;
        for k in 0..WIDTH { claimed += eq_p_el[k] * inner_evals[k]; }
        challenges_rev = challenges.clone(); challenges_rev.reverse();
        p_row = challenges_rev;
        if t == 0 { final_inner_evals = inner_evals; }
    }
    Ok((MultilinearPoint(p_row), claimed, final_inner_evals))
}

use backend::ProofError;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_evals_to_coeffs() {
        let evals = vec![EF::from_usize(3), EF::from_usize(10), EF::from_usize(27)];
        let coeffs = super::evals_to_coeffs(&evals);
        assert_eq!(coeffs[0], EF::from_usize(3));
        assert_eq!(coeffs[1], EF::from_usize(2));
        assert_eq!(coeffs[2], EF::from_usize(5));
    }
}
