//! Layered-circuit GKR for Poseidon1 (KoalaBear, t=16, alpha=3).
//!
//! Each transition is a SINGLE round (not a pair), keeping degree at most 4.
//! Transition structure:
//! - t=0..3: 4 beginning full rounds (add rc, cube all, MDS) — degree 4
//! - t=4: linear transition (add frc, M_i multiply) — degree 2
//! - t=5..24: 20 partial rounds — degree 4
//! - t=25..26: 2 ending full rounds — degree 4
//! Total: 27 transitions. The final 2 ending rounds are AIR-verified.

use backend::*;
use lean_vm::{EF, F};
use rayon::prelude::*;

const WIDTH: usize = 16;
const N_TRANSITIONS: usize = 29;

type PEF = EFPacking<EF>;
const PACK_WIDTH: usize = <<F as Field>::Packing as PackedValue>::WIDTH;
const RAYON_CHUNK: usize = 256;

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

struct PoseidonExtraData { eq_p_el: Vec<EF>, c: PoseidonConstants }
unsafe impl Send for PoseidonExtraData {}
unsafe impl Sync for PoseidonExtraData {}

#[inline(always)]
fn eval_packed_transition(t: usize, state_in: &[PEF; WIDTH], extra: &PoseidonExtraData) -> PEF {
    let c = &extra.c;
    let eq_el: [PEF; WIDTH] = std::array::from_fn(|k| PEF::from(extra.eq_p_el[k]));
    match t {
        0..=3 => {
            let rc = &c.initial_rc[t];
            let mut state = *state_in;
            for k in 0..WIDTH { state[k] += rc[k]; state[k] = state[k] * state[k] * state[k]; }
            mds_circ_16(&mut state);
            eq_el.iter().zip(state.iter()).map(|(&e, &s)| e * s).fold(PEF::default(), |a, b| a + b)
        }
        4 => {
            let mut state = *state_in;
            for (s, &rc) in state.iter_mut().zip(c.frc.iter()) { *s += rc; }
            let inp = state;
            for k in 0..WIDTH { state[k] = PEF::default(); for j in 0..WIDTH { state[k] += inp[j] * c.m_i[k][j]; } }
            eq_el.iter().zip(state.iter()).map(|(&e, &s)| e * s).fold(PEF::default(), |a, b| a + b)
        }
        5..=24 => {
            let r = t - 5;
            let mut cw = eq_el[0] * c.first_rows[r][0];
            for k in 1..WIDTH { cw += eq_el[k] * c.v_vecs[r][k - 1]; }
            let mut lin = PEF::default();
            for k in 1..WIDTH { lin += (eq_el[k] + eq_el[0] * c.first_rows[r][k]) * state_in[k]; }
            let mut cubed = state_in[0] * state_in[0] * state_in[0];
            if r < 19 { cubed += c.scalar_rc[r]; }
            lin + cw * cubed
        }
        25..=28 => {
            let rc = &c.final_rc[t - 25];
            let mut state = *state_in;
            for k in 0..WIDTH { state[k] += rc[k]; state[k] = state[k] * state[k] * state[k]; }
            mds_circ_16(&mut state);
            eq_el.iter().zip(state.iter()).map(|(&e, &s)| e * s).fold(PEF::default(), |a, b| a + b)
        }
        _ => unreachable!(),
    }
}

fn compute_checkpoint_states_base(input_cols: &[&[F]], n_rows: usize) -> Vec<Vec<[F; WIDTH]>> {
    let c = poseidon_constants();
    // Pre-allocate all checkpoints at once to avoid per-step allocation
    let mut checkpoints: Vec<Vec<[F; WIDTH]>> = (0..N_TRANSITIONS + 1)
        .map(|_| unsafe { let mut v = Vec::with_capacity(n_rows); v.set_len(n_rows); v })
        .collect();
    // Initialize checkpoint 0 from input columns
    for i in 0..n_rows { checkpoints[0][i] = std::array::from_fn(|k| input_cols[k][i]); }
    let mut cp_idx = 0;
    for r in 0..4 {
        let (src, dst) = if cp_idx + 1 < checkpoints.len() {
            let (left, right) = checkpoints.split_at_mut(cp_idx + 1);
            (&left[cp_idx], &mut right[0])
        } else { unreachable!() };
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| {
            *d = *s;
            for k in 0..WIDTH { d[k] += c.initial_rc[r][k]; d[k] = d[k].cube(); }
            mds_circ_16(d);
        });
        cp_idx += 1;
    }
    {
        let (src, dst) = { let (left, right) = checkpoints.split_at_mut(cp_idx + 1); (&left[cp_idx], &mut right[0]) };
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| {
            *d = *s;
            for (ds, &rc) in d.iter_mut().zip(c.frc.iter()) { *ds += rc; }
            let inp = *d;
            for k in 0..WIDTH { d[k] = F::ZERO; for j in 0..WIDTH { d[k] += inp[j] * c.m_i[k][j]; } }
        });
        cp_idx += 1;
    }
    for r in 0..20 {
        let (src, dst) = { let (left, right) = checkpoints.split_at_mut(cp_idx + 1); (&left[cp_idx], &mut right[0]) };
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| {
            *d = *s;
            d[0] = d[0].cube();
            if r < 19 { d[0] += c.scalar_rc[r]; }
            let old_s0 = d[0];
            let mut new_s0 = F::ZERO;
            for j in 0..WIDTH { new_s0 += d[j] * c.first_rows[r][j]; }
            d[0] = new_s0;
            for j in 1..WIDTH { d[j] += old_s0 * c.v_vecs[r][j - 1]; }
        });
        cp_idx += 1;
    }
    for r in 0..4 {
        let (src, dst) = { let (left, right) = checkpoints.split_at_mut(cp_idx + 1); (&left[cp_idx], &mut right[0]) };
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| {
            *d = *s;
            for k in 0..WIDTH { d[k] += c.final_rc[r][k]; d[k] = d[k].cube(); }
            mds_circ_16(d);
        });
        cp_idx += 1;
    }
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

#[inline(always)]
fn row_pair_contributions(a: &[EF; WIDTH], b: &[EF; WIDTH], eq_lo: EF, eq_hi: EF, eq_p_el: &[EF], t: usize, n_evals: usize, c: &PoseidonConstants) -> [EF; 5] {
    // n_evals = degree (not degree+1): evaluate at 0, 2, 3, ..., degree
    // p(1) will be derived from p(0) + p(1) = claimed_sum at the aggregation level
    let diff_eq = eq_hi - eq_lo;
    let mut diffs = [EF::ZERO; WIDTH];
    for k in 0..WIDTH { diffs[k] = b[k] - a[k]; }
    if (5..=24).contains(&t) {
        let r = t - 5;
        let mut cw = eq_p_el[0] * c.first_rows[r][0];
        for k in 1..WIDTH { cw += eq_p_el[k] * c.v_vecs[r][k - 1]; }
        let mut lc = EF::ZERO; let mut ls = EF::ZERO;
        for k in 1..WIDTH { let w = eq_p_el[k] + eq_p_el[0] * c.first_rows[r][k]; lc += w * a[k]; ls += w * diffs[k]; }
        let mut result = [EF::ZERO; 5];
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 }; // points: 0, 2, 3, ..., degree
            let pt = F::from_usize(point);
            let s0 = a[0] + diffs[0] * pt;
            let mut cubed = s0.cube();
            if r < 19 { cubed += c.scalar_rc[r]; }
            result[idx] = (eq_lo + diff_eq * pt) * ((lc + ls * pt) + cw * cubed);
        }
        return result;
    }
    let mut result = [EF::ZERO; 5];
    for idx in 0..n_evals {
        let point = if idx == 0 { 0 } else { idx + 1 };
        let pt = F::from_usize(point);
        let prev_interp: [EF; WIDTH] = std::array::from_fn(|k| a[k] + diffs[k] * pt);
        let out = apply_transition_to_evals(t, &prev_interp, c);
        let mut w = EF::ZERO;
        for k in 0..WIDTH { w += eq_p_el[k] * out[k]; }
        result[idx] = (eq_lo + diff_eq * pt) * w;
    }
    result
}

type PBF = <F as Field>::Packing;
const BF_PACK_WIDTH: usize = <PBF as PackedValue>::WIDTH;

/// First-round base-field evaluation: checkpoint data is F, only eq/dot products are EF.
/// The transition (cube, MDS) runs entirely in packed base field, then dot with eq_p_el gives PEF.
fn packed_row_pairs_base(prev: &[[F; WIDTH]], eq_table: &[EF], eq_p_el: &[EF], start: usize, t: usize, n_evals: usize, c: &PoseidonConstants) -> [PEF; 5] {
    let eq_lo_p = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| eq_table[2 * (start + i)]));
    let eq_hi_p = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| eq_table[2 * (start + i) + 1]));
    let diff_eq_p = eq_hi_p - eq_lo_p;
    let eq_el: [PEF; WIDTH] = std::array::from_fn(|k| PEF::from(eq_p_el[k]));

    // Load base-field state into packed base field
    let mut a_bf = [PBF::default(); WIDTH]; let mut d_bf = [PBF::default(); WIDTH];
    for k in 0..WIDTH {
        a_bf[k] = *PBF::from_slice(&std::array::from_fn::<F, BF_PACK_WIDTH, _>(|i| prev[2 * (start + i)][k]));
        d_bf[k] = *PBF::from_slice(&std::array::from_fn::<F, BF_PACK_WIDTH, _>(|i| prev[2 * (start + i) + 1][k])) - a_bf[k];
    }

    let mut result = [PEF::default(); 5];
    if (5..=24).contains(&t) {
        let r = t - 5;
        // Partial round: cube s0, sparse update, then dot with eq_p_el
        // Precompute eq-weighted constants (EF)
        let mut cw = eq_el[0] * c.first_rows[r][0];
        for k in 1..WIDTH { cw += eq_el[k] * c.v_vecs[r][k - 1]; }
        // eq-weighted column sums for linear part
        let mut w_k = [PEF::default(); WIDTH];
        for k in 1..WIDTH { w_k[k] = eq_el[k] + eq_el[0] * c.first_rows[r][k]; }
        let mut lc = PEF::default(); let mut ls = PEF::default();
        for k in 1..WIDTH { lc += w_k[k] * a_bf[k]; ls += w_k[k] * d_bf[k]; }
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let s0_bf = a_bf[0] + d_bf[0] * pt;
            let cubed_bf = s0_bf * s0_bf * s0_bf;
            let mut cubed_pef = PEF::from(cubed_bf);
            if r < 19 { cubed_pef += c.scalar_rc[r]; }
            result[idx] = (eq_lo_p + diff_eq_p * pt) * ((lc + ls * pt) + cw * cubed_pef);
        }
    } else if t <= 3 || t >= 25 {
        let rc = match t { 0..=3 => &c.initial_rc[t], 25..=28 => &c.final_rc[t - 25], _ => unreachable!() };
        // Precompute MDS^T * eq_el for base-field full rounds
        const MDS_COL_BF: [F; WIDTH] = F::new_array([1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1]);
        let mds_t_eq_bf: [PEF; WIDTH] = std::array::from_fn(|k| {
            let mut acc = PEF::default();
            for j in 0..WIDTH { acc += eq_el[j] * MDS_COL_BF[(j + WIDTH - k) % WIDTH]; }
            acc
        });
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            // <eq_el, MDS(cube(s+rc))> = <MDS^T*eq_el, cube(s+rc)>
            let mut w = PEF::default();
            for k in 0..WIDTH {
                let s = a_bf[k] + d_bf[k] * pt + rc[k];
                w += mds_t_eq_bf[k] * (s * s * s);
            }
            result[idx] = (eq_lo_p + diff_eq_p * pt) * w;
        }
    } else {
        // Linear transition (t=4)
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let mut interp_bf: [PBF; WIDTH] = std::array::from_fn(|k| a_bf[k] + d_bf[k] * pt);
            for (s, &rc) in interp_bf.iter_mut().zip(c.frc.iter()) { *s += rc; }
            let inp = interp_bf;
            for k in 0..WIDTH { interp_bf[k] = PBF::default(); for j in 0..WIDTH { interp_bf[k] += inp[j] * c.m_i[k][j]; } }
            let mut w = PEF::default();
            for k in 0..WIDTH { w += eq_el[k] * interp_bf[k]; }
            result[idx] = (eq_lo_p + diff_eq_p * pt) * w;
        }
    }
    result
}

/// Precomputed constants for a specific transition type, avoiding per-pack recomputation
struct TransitionPrecomp {
    eq_el: [PEF; WIDTH],
    cw: PEF,
    weights: [PEF; WIDTH],
    mds_t_eq: [PEF; WIDTH],
    mi_t_eq: [PEF; WIDTH], // M_i^T * eq_el for linear transition
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
        } else {
            (PEF::default(), [PEF::default(); WIDTH])
        };
        // Precompute MDS^T * eq_el: (MDS^T * eq_el)[k] = Σ_j MDS[j][k] * eq_el[j]
        // For circulant MDS with column c: MDS[j][k] = c[(j-k) mod 16]
        const MDS_COL: [F; WIDTH] = F::new_array([1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1]);
        let mds_t_eq: [PEF; WIDTH] = std::array::from_fn(|k| {
            let mut acc = PEF::default();
            for j in 0..WIDTH { acc += eq_el[j] * MDS_COL[(j + WIDTH - k) % WIDTH]; }
            acc
        });
        // M_i^T * eq_el for linear transition
        let mi_t_eq: [PEF; WIDTH] = if t == 4 {
            let mi = poseidon1_sparse_m_i();
            std::array::from_fn(|k| {
                let mut acc = PEF::default();
                for j in 0..WIDTH { acc += eq_el[j] * mi[j][k]; }
                acc
            })
        } else { [PEF::default(); WIDTH] };
        Self { eq_el, cw, weights, mds_t_eq, mi_t_eq }
    }
}

fn packed_row_pairs_pef(folded_prev: &[[EF; WIDTH]], eq_table: &[EF], pre: &TransitionPrecomp, start: usize, t: usize, n_evals: usize, c: &PoseidonConstants) -> [PEF; 5] {
    let eq_lo_p = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| eq_table[2 * (start + i)]));
    let eq_hi_p = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| eq_table[2 * (start + i) + 1]));
    let diff_eq_p = eq_hi_p - eq_lo_p;
    let mut a_p = [PEF::default(); WIDTH]; let mut d_p = [PEF::default(); WIDTH];
    for k in 0..WIDTH {
        a_p[k] = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| folded_prev[2 * (start + i)][k]));
        d_p[k] = PEF::from_ext_slice(&std::array::from_fn::<EF, PACK_WIDTH, _>(|i| folded_prev[2 * (start + i) + 1][k])) - a_p[k];
    }
    let eq_el = &pre.eq_el;
    let mut result = [PEF::default(); 5];
    if (5..=24).contains(&t) {
        let r = t - 5;
        let cw = pre.cw;
        let mut lc = PEF::default(); let mut ls = PEF::default();
        for k in 1..WIDTH { lc += pre.weights[k] * a_p[k]; ls += pre.weights[k] * d_p[k]; }
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let s0 = a_p[0] + d_p[0] * pt;
            let mut cubed = s0 * s0 * s0;
            if r < 19 { cubed += c.scalar_rc[r]; }
            result[idx] += (eq_lo_p + diff_eq_p * pt) * ((lc + ls * pt) + cw * cubed);
        }
    } else if t <= 3 || t >= 25 {
        let rc = match t { 0..=3 => &c.initial_rc[t], 25..=28 => &c.final_rc[t - 25], _ => unreachable!() };
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            // <eq_el, MDS(cube(s+rc))> = <MDS^T*eq_el, cube(s+rc)> — skips explicit MDS
            let mut w = PEF::default();
            for k in 0..WIDTH {
                let s = a_p[k] + d_p[k] * pt + rc[k];
                w += pre.mds_t_eq[k] * (s * s * s);
            }
            result[idx] += (eq_lo_p + diff_eq_p * pt) * w;
        }
    } else {
        // t=4: linear transition. <eq_el, M_i*(state+frc)> = <M_i^T*eq_el, state+frc>
        for idx in 0..n_evals {
            let point = if idx == 0 { 0 } else { idx + 1 };
            let pt = F::from_usize(point);
            let mut w = PEF::default();
            for k in 0..WIDTH {
                w += pre.mi_t_eq[k] * (a_p[k] + d_p[k] * pt + c.frc[k]);
            }
            result[idx] += (eq_lo_p + diff_eq_p * pt) * w;
        }
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

pub fn prove_poseidon_gkr(prover_state: &mut impl FSProver<EF>, input_cols: &[&[F]], n_rows: usize, log_n_rows: usize) -> (MultilinearPoint<EF>, Vec<EF>) {
    let c = poseidon_constants();
    let checkpoints = compute_checkpoint_states_base(input_cols, n_rows);

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
        let degree = sumcheck_degree(t);
        let prev = &checkpoints[t];
        let n = 1usize << log_n_rows;
        if t < N_TRANSITIONS - 1 { prover_state.add_extension_scalar(current_claim); }

        let extra = PoseidonExtraData { eq_p_el: eq_p_el.clone(), c: poseidon_constants() };
        let mut eq_table = eval_eq(&p_row);
        let mut challenges = Vec::with_capacity(log_n_rows);

        // First round (v=0): evaluate in base field
        {
            let half = n >> 1;
            let n_evals = degree;
            let n_packed_bf = half / BF_PACK_WIDTH;

            let mut raw_evals: Vec<EF> = if n_packed_bf >= RAYON_CHUNK / BF_PACK_WIDTH {
                let packed_sums = (0..n_packed_bf).into_par_iter()
                    .fold(|| [PEF::default(); 5], |mut acc, p| {
                        let contrib = packed_row_pairs_base(prev, &eq_table, &eq_p_el, p * BF_PACK_WIDTH, t, n_evals, &c);
                        for point in 0..n_evals { acc[point] += contrib[point]; } acc
                    })
                    .reduce(|| [PEF::default(); 5], |mut a, b| { for p in 0..n_evals { a[p] += b[p]; } a });
                (0..n_evals).map(|p| hsum_pef(packed_sums[p])).collect()
            } else {
                vec![EF::ZERO; n_evals] // fallback for tiny sizes
            };
            let p_at_1 = current_claim - raw_evals[0];
            raw_evals.insert(1, p_at_1);
            let coeffs = evals_to_coeffs(&raw_evals);
            prover_state.add_extension_scalars(&coeffs);
            let r_v: EF = prover_state.sample();
            challenges.push(r_v);
            let one_minus_r = EF::ONE - r_v;
            // Fold from F to EF using the challenge
            eq_table = eq_table.par_chunks_exact(2).map(|pair| pair[0] * one_minus_r + pair[1] * r_v).collect();
            current_claim = coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r_v + c);
        }

        // Fold from F to EF
        let r0 = challenges[0];
        let mut folded_prev: Vec<[EF; WIDTH]> = prev.par_chunks_exact(2).map(|pair| {
            std::array::from_fn(|k| EF::from(pair[0][k]) + EF::from(pair[1][k] - pair[0][k]) * r0)
        }).collect();
        let pre = TransitionPrecomp::new(&eq_p_el, t, &c);
        for v in 1..log_n_rows {
            let half = n >> (v + 1);
            let n_evals = degree;
            let n_packed = half / PACK_WIDTH;

            let mut raw_evals: Vec<EF> = if n_packed >= RAYON_CHUNK / PACK_WIDTH {
                let packed_sums = (0..n_packed).into_par_iter()
                    .fold(|| [PEF::default(); 5], |mut acc, p| {
                        let contrib = packed_row_pairs_pef(&folded_prev, &eq_table, &pre, p * PACK_WIDTH, t, n_evals, &c);
                        for point in 0..n_evals { acc[point] += contrib[point]; } acc
                    })
                    .reduce(|| [PEF::default(); 5], |mut a, b| { for p in 0..n_evals { a[p] += b[p]; } a });
                let mut evals: Vec<EF> = (0..n_evals).map(|p| hsum_pef(packed_sums[p])).collect();
                for h in (n_packed * PACK_WIDTH)..half {
                    let contrib = row_pair_contributions(&folded_prev[2*h], &folded_prev[2*h+1], eq_table[2*h], eq_table[2*h+1], &eq_p_el, t, n_evals, &c);
                    for point in 0..n_evals { evals[point] += contrib[point]; }
                }
                evals
            } else if half >= 256 {
                let sums = (0..half).into_par_iter()
                    .fold(|| [EF::ZERO; 5], |mut acc, h| {
                        let contrib = row_pair_contributions(&folded_prev[2*h], &folded_prev[2*h+1], eq_table[2*h], eq_table[2*h+1], &eq_p_el, t, n_evals, &c);
                        for point in 0..n_evals { acc[point] += contrib[point]; } acc
                    })
                    .reduce(|| [EF::ZERO; 5], |mut a, b| { for p in 0..n_evals { a[p] += b[p]; } a });
                sums[..n_evals].to_vec()
            } else {
                let mut evals = vec![EF::ZERO; n_evals];
                for h in 0..half {
                    let contrib = row_pair_contributions(&folded_prev[2*h], &folded_prev[2*h+1], eq_table[2*h], eq_table[2*h+1], &eq_p_el, t, n_evals, &c);
                    for point in 0..n_evals { evals[point] += contrib[point]; }
                }
                evals
            };
            // Insert p(1) = current_claim - p(0) at position 1
            let p_at_1 = current_claim - raw_evals[0];
            raw_evals.insert(1, p_at_1);
            let coeffs = evals_to_coeffs(&raw_evals);
            prover_state.add_extension_scalars(&coeffs);
            let r_v: EF = prover_state.sample();
            challenges.push(r_v);
            let one_minus_r = EF::ONE - r_v;
            if half >= RAYON_CHUNK {
                // Fused fold: eq and prev folded in a single parallel pass
                let mut new_eq: Vec<EF> = Vec::with_capacity(half);
                let mut new_prev: Vec<[EF; WIDTH]> = Vec::with_capacity(half);
                unsafe { new_eq.set_len(half); new_prev.set_len(half); }
                new_eq.par_iter_mut().zip(new_prev.par_iter_mut()).enumerate().for_each(|(h, (eq_out, prev_out))| {
                    *eq_out = eq_table[2*h] * one_minus_r + eq_table[2*h+1] * r_v;
                    *prev_out = std::array::from_fn(|k| folded_prev[2*h][k] * one_minus_r + folded_prev[2*h+1][k] * r_v);
                });
                eq_table = new_eq; folded_prev = new_prev;
            } else {
                for h in 0..half { eq_table[h] = eq_table[2*h] * one_minus_r + eq_table[2*h+1] * r_v; for k in 0..WIDTH { folded_prev[h][k] = folded_prev[2*h][k] * one_minus_r + folded_prev[2*h+1][k] * r_v; } }
                eq_table.truncate(half); folded_prev.truncate(half);
            }
            current_claim = coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r_v + c);
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
        let mut challenges = Vec::with_capacity(log_n_rows);
        if t < N_TRANSITIONS - 1 { claimed = verifier_state.next_extension_scalar()?; }
        for _v in 0..log_n_rows {
            let n_coeffs = degree + 1;
            let coeffs = verifier_state.next_extension_scalars_vec(n_coeffs)?;
            let p0 = coeffs[0]; let p1: EF = coeffs.iter().copied().sum();
            if p0 + p1 != claimed { return Err(ProofError::InvalidProof); }
            let r_v: EF = verifier_state.sample();
            challenges.push(r_v);
            claimed = coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r_v + c);
        }
        let inner_evals = verifier_state.next_extension_scalars_vec(WIDTH)?;
        let trans_out = apply_transition_to_evals(t, &inner_evals, &c);
        let mut h_val = EF::ZERO;
        for k in 0..WIDTH { h_val += eq_p_el[k] * trans_out[k]; }
        let mut challenges_rev = challenges.clone(); challenges_rev.reverse();
        let eq_at = MultilinearPoint(p_row.clone()).eq_poly_outside(&MultilinearPoint(challenges_rev));
        if claimed != eq_at * h_val { return Err(ProofError::InvalidProof); }
        verifier_state.duplex();
        let alpha: Vec<EF> = verifier_state.sample_vec(4);
        eq_p_el = eval_eq(&alpha);
        claimed = EF::ZERO;
        for k in 0..WIDTH { claimed += eq_p_el[k] * inner_evals[k]; }
        challenges.reverse(); p_row = challenges;
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
