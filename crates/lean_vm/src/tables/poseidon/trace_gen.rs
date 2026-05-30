use tracing::instrument;

use crate::{
    F,
    tables::{Poseidon1Cols16, WIDTH, poseidon::PARTIAL_ROUNDS},
};
use backend::*;

#[instrument(name = "generate Poseidon16 AIR trace", skip_all)]
pub fn fill_trace_poseidon_16(trace: &mut [Vec<F>]) {
    let n = trace.iter().map(|col| col.len()).max().unwrap();
    for col in trace.iter_mut() {
        if col.len() != n {
            col.resize(n, F::ZERO);
        }
    }

    let m = n - (n % packing_width::<F>());
    let trace_packed: Vec<_> = trace.iter().map(|col| FPacking::<F>::pack_slice(&col[..m])).collect();

    // fill the packed rows
    (0..m / packing_width::<F>()).into_par_iter().for_each(|i| {
        let ptrs: Vec<*mut FPacking<F>> = trace_packed
            .iter()
            .map(|col| unsafe { (col.as_ptr() as *mut FPacking<F>).add(i) })
            .collect();
        let perm: &mut Poseidon1Cols16<&mut FPacking<F>> =
            unsafe { &mut *(ptrs.as_ptr() as *mut Poseidon1Cols16<&mut FPacking<F>>) };

        generate_trace_rows_for_perm(perm);
    });

    // fill the remaining rows (non packed)
    for i in m..n {
        let ptrs: Vec<*mut F> = trace
            .iter()
            .map(|col| unsafe { (col.as_ptr() as *mut F).add(i) })
            .collect();
        let perm: &mut Poseidon1Cols16<&mut F> = unsafe { &mut *(ptrs.as_ptr() as *mut Poseidon1Cols16<&mut F>) };
        generate_trace_rows_for_perm(perm);
    }
}

pub(super) fn generate_trace_rows_for_perm<F: Algebra<KoalaBear> + Copy>(perm: &mut Poseidon1Cols16<&mut F>) {
    let inputs: [F; WIDTH] = std::array::from_fn(|i| *perm.inputs[i]);
    let mut state = inputs;

    // Beginning full rounds — state computed but intermediates NOT written (GKR-verified)
    for constants in poseidon1_initial_constants().chunks_exact(2) {
        for (s, &c) in state.iter_mut().zip(constants[0].iter()) { *s += c; *s = s.cube(); }
        mds_circ_16(&mut state);
        for (s, &c) in state.iter_mut().zip(constants[1].iter()) { *s += c; *s = s.cube(); }
        mds_circ_16(&mut state);
    }

    // Sparse partial rounds — intermediates NOT written (GKR-verified)
    let frc = poseidon1_sparse_first_round_constants();
    for (s, &c) in state.iter_mut().zip(frc.iter()) { *s += c; }
    let m_i = poseidon1_sparse_m_i();
    let input_for_mi = state;
    for i in 0..WIDTH {
        let row: [F; WIDTH] = m_i[i].map(F::from);
        state[i] = F::dot_product(&input_for_mi, &row);
    }

    let first_rows = poseidon1_sparse_first_row();
    let v_vecs = poseidon1_sparse_v();
    let scalar_rc = poseidon1_sparse_scalar_round_constants();
    let n_partial = PARTIAL_ROUNDS;
    for round in 0..n_partial {
        state[0] = state[0].cube();
        if round < n_partial - 1 { state[0] += scalar_rc[round]; }
        let old_s0 = state[0];
        let row: [F; WIDTH] = first_rows[round].map(F::from);
        state[0] = F::dot_product(&state, &row);
        for i in 1..WIDTH { state[i] += old_s0 * v_vecs[round][i - 1]; }
    }

    // Ending full rounds — only write LAST pair (used by AIR compression check)
    // Ending full rounds (4 rounds = 2 pairs)
    let final_consts = poseidon1_final_constants();
    for constants in final_consts.chunks_exact(2) {
        for (s, &c) in state.iter_mut().zip(constants[0].iter()) { *s += c; *s = s.cube(); }
        mds_circ_16(&mut state);
        for (s, &c) in state.iter_mut().zip(constants[1].iter()) { *s += c; *s = s.cube(); }
        mds_circ_16(&mut state);
    }
    // Write final state (GKR-verified)
    for k in 0..WIDTH { *perm.final_state[k] = state[k]; }

    // Output compression from the GKR-verified final state (same logic as generate_last_2_full_rounds)
    let flag_permute = *perm.flag_permute;
    for i in 0..(WIDTH / 2) {
        let compression_value = state[i] + inputs[i];
        *perm.out_lo[i] = (F::ONE - flag_permute) * compression_value + flag_permute * state[i];
        *perm.out_hi[i] = flag_permute * state[i + WIDTH / 2];
    }
}

#[inline]
fn generate_2_full_round<F: Algebra<KoalaBear> + Copy>(
    state: &mut [F; WIDTH],
    post_full_round: &mut [&mut F; WIDTH],
    round_constants_1: &[KoalaBear; WIDTH],
    round_constants_2: &[KoalaBear; WIDTH],
) {
    for (state_i, const_i) in state.iter_mut().zip(round_constants_1) {
        *state_i += *const_i;
        *state_i = state_i.cube();
    }
    mds_circ_16(state);

    for (state_i, const_i) in state.iter_mut().zip(round_constants_2.iter()) {
        *state_i += *const_i;
        *state_i = state_i.cube();
    }
    mds_circ_16(state);

    post_full_round.iter_mut().zip(*state).for_each(|(post, x)| {
        **post = x;
    });
}

#[inline]
fn generate_last_2_full_rounds<F: Algebra<KoalaBear> + Copy>(
    state: &mut [F; WIDTH],
    inputs: &[F; WIDTH],
    out_lo: &mut [&mut F; WIDTH / 2],
    out_hi: &mut [&mut F; WIDTH / 2],
    flag_permute: F,
    round_constants_1: &[KoalaBear; WIDTH],
    round_constants_2: &[KoalaBear; WIDTH],
) {
    for (state_i, const_i) in state.iter_mut().zip(round_constants_1) {
        *state_i += *const_i;
        *state_i = state_i.cube();
    }
    mds_circ_16(state);

    for (state_i, const_i) in state.iter_mut().zip(round_constants_2.iter()) {
        *state_i += *const_i;
        *state_i = state_i.cube();
    }
    mds_circ_16(state);

    for i in 0..(WIDTH / 2) {
        let compression_value = state[i] + inputs[i];
        *out_lo[i] = (F::ONE - flag_permute) * compression_value + flag_permute * state[i];
        *out_hi[i] = flag_permute * state[i + WIDTH / 2];
    }
}
