use backend::*;
use lean_vm::{EF, F};
use utils::ToUsize;

/// Compute eq(bits(a), s) for a field element `a` and random point `s`.
///
/// Decomposes `a` into `n_bits` binary digits and computes:
///   prod_{b=0}^{n_bits-1} (bit_b * s[b] + (1 - bit_b) * (1 - s[b]))
#[inline]
pub fn eq_bits_at_point(a: F, s: &[EF], n_bits: usize) -> EF {
    let a_val = a.to_usize();
    let mut result = EF::ONE;
    for b in 0..n_bits.min(s.len()) {
        let bit = EF::from(F::from_usize((a_val >> b) & 1));
        result *= bit * s[b] + (EF::ONE - bit) * (EF::ONE - s[b]);
    }
    result
}

/// Build the joint pushforward P_joint[k] = sum_{i: addr[i]=k} eq_r[i].
///
/// Scatters eq_r values into a table of size `memory_size` keyed by address.
pub fn compute_joint_pushforward(
    addr_col: &[F],
    memory_size: usize,
    eq_r: &[EF],
) -> Vec<EF> {
    assert_eq!(addr_col.len(), eq_r.len());
    let mut p_joint = EF::zero_vec(memory_size);
    for (i, &addr) in addr_col.iter().enumerate() {
        let k = addr.to_usize();
        if k < memory_size {
            p_joint[k] += eq_r[i];
        }
    }
    p_joint
}

/// Batch memory values across value columns:
///   batched_mem[j] = sum_{k=0}^{n_value_cols-1} gamma^k * memory[j + k]
///
/// The memory is laid out linearly; the k-th value column at address j is memory[j + k].
/// Returns a vector of size `memory.len()`.
pub fn compute_batched_memory(
    memory: &[F],
    n_value_cols: usize,
    gamma: EF,
) -> Vec<EF> {
    let k_size = memory.len();
    (0..k_size)
        .into_par_iter()
        .map(|j| {
            let mut acc = EF::ZERO;
            let mut gp = EF::ONE;
            for k in 0..n_value_cols {
                if j + k < k_size {
                    acc += gp * memory[j + k];
                }
                gp *= gamma;
            }
            acc
        })
        .collect()
}

/// Fold a table in-place: table[j] = table[2j] + r * (table[2j+1] - table[2j])
/// for j in 0..half. Truncates the table to half its size.
fn fold_table(table: &mut Vec<EF>, r: EF) {
    let half = table.len() / 2;
    for j in 0..half {
        let lo = table[2 * j];
        let hi = table[2 * j + 1];
        table[j] = lo + r * (hi - lo);
    }
    table.truncate(half);
}

/// Prove the Shout value sumcheck (degree 2, log(K) rounds).
///
/// Proves: `claimed_sum = sum_{k in {0,1}^n} P_joint_tilde(k) * batched_mem_tilde(k)`
///
/// This is a degree-2 multilinear sumcheck (product of two MLEs). Each round
/// computes the univariate polynomial p(X) = c_0 + c_1*X + c_2*X^2 from
/// evaluations at X = 0, 1, 2.
///
/// Returns (endpoint_value, sumcheck_point) where:
///   - endpoint_value = P_joint_tilde(s) * batched_mem_tilde(s)
///   - sumcheck_point s = (r_0, ..., r_{n-1})
pub fn prove_shout_value_sumcheck(
    prover_state: &mut impl FSProver<EF>,
    p_joint: &mut Vec<EF>,
    batched_mem: &mut Vec<EF>,
    claimed_sum: EF,
) -> (EF, Vec<EF>) {
    let n_vars = log2_ceil_usize(p_joint.len());
    assert_eq!(p_joint.len(), 1 << n_vars);
    assert_eq!(batched_mem.len(), 1 << n_vars);

    let mut _running_sum = claimed_sum;
    let mut challenges = Vec::with_capacity(n_vars);

    for _ in 0..n_vars {
        let half = p_joint.len() / 2;

        // Compute p(0), p(1), p(2) for the product of two multilinear polynomials.
        //   p(x) = sum_j a_fold(j,x) * b_fold(j,x)
        // where a_fold(j,x) = a[2j] + (a[2j+1] - a[2j])*x, same for b.
        //   p(0) = sum_j a[2j]*b[2j]
        //   p(1) = sum_j a[2j+1]*b[2j+1]
        //   p(2) = sum_j (2*a[2j+1]-a[2j])*(2*b[2j+1]-b[2j])
        let (c_0, c_2) = (0..half)
            .into_par_iter()
            .map(|j| {
                let a0 = p_joint[2 * j];
                let a1 = p_joint[2 * j + 1];
                let b0 = batched_mem[2 * j];
                let b1 = batched_mem[2 * j + 1];
                // c_0 contribution: a0 * b0
                // c_2 contribution: (a1 - a0) * (b1 - b0)
                (a0 * b0, (a1 - a0) * (b1 - b0))
            })
            .reduce(
                || (EF::ZERO, EF::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );

        // Polynomial: p(x) = c_0 + c_1*x + c_2*x^2
        // From product_computation.rs pattern:
        //   c_0 = sum_j a[2j]*b[2j] = p(0)
        //   c_2 = sum_j (a[2j+1]-a[2j])*(b[2j+1]-b[2j])
        //   c_1 = claimed_sum - 2*c_0 - c_2  (from p(0)+p(1) = sum)
        let c_1 = _running_sum - c_0.double() - c_2;
        let coeffs = DensePolynomial::new(vec![c_0, c_1, c_2]);

        prover_state.add_sumcheck_polynomial(&coeffs.coeffs, None);
        let r = prover_state.sample();
        challenges.push(r);

        // Update running sum
        _running_sum = coeffs.evaluate(r);

        // Fold both tables
        fold_table(p_joint, r);
        fold_table(batched_mem, r);
    }

    assert_eq!(p_joint.len(), 1);
    assert_eq!(batched_mem.len(), 1);

    let endpoint = p_joint[0] * batched_mem[0];
    (endpoint, challenges)
}

/// Verify the Shout value sumcheck (degree 2, log(K) rounds).
///
/// Returns (endpoint_value, sumcheck_point s) or an error.
pub fn verify_shout_value_sumcheck(
    verifier_state: &mut impl FSVerifier<EF>,
    n_vars: usize,
    claimed_sum: EF,
) -> Result<(EF, Vec<EF>), ProofError> {
    let eval = sumcheck_verify(verifier_state, n_vars, 2, claimed_sum, None)?;
    Ok((eval.value, eval.point.0))
}

/// Build eq tables for tensor decomposition:
///   eq_hi_table[row] = eq(bits(addr_hi[row]), s_hi)
///   eq_lo_table[row] = eq(bits(addr_lo[row]), s_lo)
///
/// where addr_hi and addr_lo are the high and low halves of the address column,
/// s_hi and s_lo are the first and second halves of the sumcheck point s,
/// and half_bits = log(sqrt(K)) = n/2.
pub fn build_eq_addr_tables(
    addr_hi: &[F],
    addr_lo: &[F],
    s_hi: &[EF],
    s_lo: &[EF],
    half_bits: usize,
) -> (Vec<EF>, Vec<EF>) {
    assert_eq!(addr_hi.len(), addr_lo.len());

    let n_rows = addr_hi.len();
    let eq_hi_table: Vec<EF> = (0..n_rows)
        .into_par_iter()
        .map(|row| eq_bits_at_point(addr_hi[row], s_hi, half_bits))
        .collect();

    let eq_lo_table: Vec<EF> = (0..n_rows)
        .into_par_iter()
        .map(|row| eq_bits_at_point(addr_lo[row], s_lo, half_bits))
        .collect();

    (eq_hi_table, eq_lo_table)
}

/// Prove the tensor decomposition sumcheck (degree 3, log(T) rounds).
///
/// Proves: `claimed_sum = sum_{row in {0,1}^{log T}} eq_r[row] * eq_hi_table[row] * eq_lo_table[row]`
///
/// This is a degree-3 multilinear sumcheck with three bookkeeping polynomials.
/// Each round computes the univariate polynomial p(X) = c_0 + c_1*X + c_2*X^2 + c_3*X^3
/// from evaluations at X = 0, 1, 2, 3.
///
/// Returns (hi_eval, lo_eval, sumcheck_point) where:
///   - hi_eval = eq_hi_table_tilde(row') at the final point
///   - lo_eval = eq_lo_table_tilde(row') at the final point
///   - sumcheck_point row' = (r_0, ..., r_{log(T)-1})
pub fn prove_tensor_decomp_sumcheck(
    prover_state: &mut impl FSProver<EF>,
    eq_r: &mut Vec<EF>,
    eq_hi_table: &mut Vec<EF>,
    eq_lo_table: &mut Vec<EF>,
    claimed_sum: EF,
) -> (EF, EF, Vec<EF>) {
    let n_vars = log2_ceil_usize(eq_r.len());
    assert_eq!(eq_r.len(), 1 << n_vars);
    assert_eq!(eq_hi_table.len(), 1 << n_vars);
    assert_eq!(eq_lo_table.len(), 1 << n_vars);

    let mut _running_sum = claimed_sum;
    let mut challenges = Vec::with_capacity(n_vars);

    for _ in 0..n_vars {
        let half = eq_r.len() / 2;

        // Compute p(0), p(1), p(2), p(3) from three bookkeeping tables.
        //
        // The univariate restriction for the i-th round is:
        //   p(x) = sum_j a_fold(j,x) * b_fold(j,x) * c_fold(j,x)
        // where a_fold(j,x) = a[2j] + (a[2j+1] - a[2j])*x, etc.
        // This is degree 3 in x (product of three linear functions).
        let (p_at_0, p_at_1, p_at_2, p_at_3) = (0..half)
            .into_par_iter()
            .map(|j| {
                let a0 = eq_r[2 * j];
                let a1 = eq_r[2 * j + 1];
                let b0 = eq_hi_table[2 * j];
                let b1 = eq_hi_table[2 * j + 1];
                let c0 = eq_lo_table[2 * j];
                let c1 = eq_lo_table[2 * j + 1];

                let da = a1 - a0;
                let db = b1 - b0;
                let dc = c1 - c0;

                let v0 = a0 * b0 * c0;
                let v1 = a1 * b1 * c1;

                // At x=2: f(2) = f0 + 2*df
                let a2 = a0 + da.double();
                let b2 = b0 + db.double();
                let c2 = c0 + dc.double();
                let v2 = a2 * b2 * c2;

                // At x=3: f(3) = f0 + 3*df
                let a3 = a2 + da;
                let b3 = b2 + db;
                let c3 = c2 + dc;
                let v3 = a3 * b3 * c3;

                (v0, v1, v2, v3)
            })
            .reduce(
                || (EF::ZERO, EF::ZERO, EF::ZERO, EF::ZERO),
                |(a0, a1, a2, a3), (b0, b1, b2, b3)| (a0 + b0, a1 + b1, a2 + b2, a3 + b3),
            );

        // Lagrange interpolation from (0, p0), (1, p1), (2, p2), (3, p3)
        // using DensePolynomial::lagrange_interpolation
        let coeffs = DensePolynomial::lagrange_interpolation(&[
            (F::ZERO, p_at_0),
            (F::ONE, p_at_1),
            (F::TWO, p_at_2),
            (F::from_usize(3), p_at_3),
        ]).unwrap();

        prover_state.add_sumcheck_polynomial(&coeffs.coeffs, None);
        let r = prover_state.sample();
        challenges.push(r);

        // Update running sum
        _running_sum = coeffs.evaluate(r);

        // Fold all three tables
        fold_table(eq_r, r);
        fold_table(eq_hi_table, r);
        fold_table(eq_lo_table, r);
    }

    assert_eq!(eq_r.len(), 1);
    assert_eq!(eq_hi_table.len(), 1);
    assert_eq!(eq_lo_table.len(), 1);

    let hi_eval = eq_hi_table[0];
    let lo_eval = eq_lo_table[0];
    (hi_eval, lo_eval, challenges)
}

/// Verify the tensor decomposition sumcheck (degree 3, log(T) rounds).
///
/// Returns (endpoint_value, sumcheck_point row') or an error.
/// The endpoint_value is eq_r_tilde(row') * eq_hi_tilde(row') * eq_lo_tilde(row').
/// The verifier can compute eq_r_tilde(row') = eq(row', r_air) independently.
/// The hi_eval and lo_eval claims are sent by the prover after the sumcheck
/// and verified via d=2 pushforward + GKR.
pub fn verify_tensor_decomp_sumcheck(
    verifier_state: &mut impl FSVerifier<EF>,
    n_vars: usize,
    claimed_sum: EF,
) -> Result<(EF, Vec<EF>), ProofError> {
    let eval = sumcheck_verify(verifier_state, n_vars, 3, claimed_sum, None)?;
    Ok((eval.value, eval.point.0))
}

/// Compute the d=2 batched pushforward at the tensor decomp endpoint.
///
/// P_hi[h] = sum_{row: addr_hi[row]=h} eq_row_prime[row]
/// P_lo[l] = sum_{row: addr_lo[row]=l} eq_row_prime[row]
/// P_batched[h] = P_hi[h] + beta * P_lo[h]
///
/// where eq_row_prime[row] = eq(bits(row), row') for all rows,
/// and h, l range over {0, ..., 2^half_bits - 1}.
pub fn compute_d2_pushforward(
    addr_hi: &[F],
    addr_lo: &[F],
    eq_row_prime: &[EF],
    half_bits: usize,
    beta: EF,
) -> (Vec<EF>, Vec<EF>, Vec<EF>) {
    assert_eq!(addr_hi.len(), addr_lo.len());
    assert_eq!(addr_hi.len(), eq_row_prime.len());

    let s = 1usize << half_bits;
    let mut p_hi = EF::zero_vec(s);
    let mut p_lo = EF::zero_vec(s);

    for (row, &eq_val) in eq_row_prime.iter().enumerate() {
        let h = addr_hi[row].to_usize();
        let l = addr_lo[row].to_usize();
        if h < s {
            p_hi[h] += eq_val;
        }
        if l < s {
            p_lo[l] += eq_val;
        }
    }

    // P_batched[h] = P_hi[h] + beta * P_lo[h]
    let p_batched: Vec<EF> = (0..s)
        .into_par_iter()
        .map(|h| p_hi[h] + beta * p_lo[h])
        .collect();

    (p_hi, p_lo, p_batched)
}

/// Evaluate pushforward MLE at a point: P_tilde(s) = sum_h P[h] * eq(bits(h), s).
///
/// This treats the pushforward vector P as a multilinear polynomial and evaluates
/// it at the given point by iterative folding.
pub fn eval_pushforward_mle(
    pushforward: &[EF],
    point: &[EF],
) -> EF {
    let n_vars = point.len();
    assert_eq!(pushforward.len(), 1 << n_vars);

    let mut table = pushforward.to_vec();
    for i in 0..n_vars {
        let half = table.len() / 2;
        let r = point[i];
        for j in 0..half {
            let lo = table[2 * j];
            let hi = table[2 * j + 1];
            table[j] = lo + r * (hi - lo);
        }
        table.truncate(half);
    }
    table[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use utils::get_poseidon16;

    /// Compute eq(bits(index), point) by decomposing index into n_bits binary digits
    fn eq_index_at_point(index: usize, point: &[EF], n_bits: usize) -> EF {
        let mut result = EF::ONE;
        for b in 0..n_bits.min(s.len()) {
            let bit = EF::from(F::from_usize((index >> b) & 1));
            result *= bit * point[b] + (EF::ONE - bit) * (EF::ONE - point[b]);
        }
        result
    }

    #[test]
    fn test_eq_bits_at_point() {
        let mut rng = StdRng::seed_from_u64(42);
        let n_bits = 5;
        let s: Vec<EF> = (0..n_bits).map(|_| rng.random()).collect();

        for a_val in 0..(1 << n_bits) {
            let a = F::from_usize(a_val);
            let result = eq_bits_at_point(a, &s, n_bits);
            let expected = eq_index_at_point(a_val, &s, n_bits);
            assert_eq!(result, expected, "eq_bits_at_point mismatch for a={a_val}");
        }
    }

    #[test]
    fn test_joint_pushforward() {
        let mut rng = StdRng::seed_from_u64(123);
        let n_rows = 64;
        let memory_size = 16;

        let addr_col: Vec<F> = (0..n_rows)
            .map(|_| F::from_usize(rng.random_range(0..memory_size)))
            .collect();
        let eq_r: Vec<EF> = (0..n_rows).map(|_| rng.random()).collect();

        let p_joint = compute_joint_pushforward(&addr_col, memory_size, &eq_r);

        // Verify manually
        for k in 0..memory_size {
            let expected: EF = addr_col
                .iter()
                .enumerate()
                .filter(|&(_, &addr)| addr.to_usize() == k)
                .map(|(i, _)| eq_r[i])
                .sum();
            assert_eq!(p_joint[k], expected, "pushforward mismatch at k={k}");
        }
    }

    #[test]
    fn test_batched_memory() {
        let mut rng = StdRng::seed_from_u64(456);
        let k_size = 32;
        let n_value_cols = 3;
        let gamma: EF = rng.random();

        let memory: Vec<F> = (0..k_size).map(|_| rng.random()).collect();
        let batched = compute_batched_memory(&memory, n_value_cols, gamma);

        // Verify first element
        let mut expected = EF::ZERO;
        let mut gp = EF::ONE;
        for k in 0..n_value_cols {
            if k < k_size {
                expected += gp * memory[k];
            }
            gp *= gamma;
        }
        assert_eq!(batched[0], expected);
    }

    #[test]
    fn test_shout_value_sumcheck_roundtrip() {
        let mut rng = StdRng::seed_from_u64(789);
        let log_k = 4;
        let k = 1 << log_k;

        // Create random P_joint and batched_mem
        let p_joint_orig: Vec<EF> = (0..k).map(|_| rng.random()).collect();
        let batched_mem_orig: Vec<EF> = (0..k).map(|_| rng.random()).collect();

        // Compute claimed sum
        let claimed_sum: EF = p_joint_orig
            .iter()
            .zip(batched_mem_orig.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        // Prove
        let mut p_joint = p_joint_orig.clone();
        let mut batched_mem = batched_mem_orig.clone();
        let mut prover_state =
            ProverState::new(get_poseidon16().clone(), Default::default());
        let (endpoint, point) = prove_shout_value_sumcheck(
            &mut prover_state,
            &mut p_joint,
            &mut batched_mem,
            claimed_sum,
        );

        assert_eq!(point.len(), log_k);

        // Verify
        let mut verifier_state = VerifierState::<EF, _>::new(
            prover_state.into_proof(),
            get_poseidon16().clone(),
            Default::default(),
        )
        .unwrap();
        let (v_endpoint, v_point) =
            verify_shout_value_sumcheck(&mut verifier_state, log_k, claimed_sum).unwrap();

        assert_eq!(point, v_point, "prover and verifier points must match");
        assert_eq!(
            endpoint, v_endpoint,
            "prover and verifier endpoints must match"
        );

        // Check that the endpoint equals P_joint_tilde(s) * batched_mem_tilde(s)
        let p_joint_at_s = eval_pushforward_mle(&p_joint_orig, &point);
        let batched_mem_at_s = eval_pushforward_mle(&batched_mem_orig, &point);
        assert_eq!(endpoint, p_joint_at_s * batched_mem_at_s);
    }

    #[test]
    fn test_tensor_decomp_sumcheck_roundtrip() {
        let mut rng = StdRng::seed_from_u64(321);
        let log_t = 4;
        let t = 1 << log_t;

        // Create random tables
        let eq_r_orig: Vec<EF> = (0..t).map(|_| rng.random()).collect();
        let eq_hi_orig: Vec<EF> = (0..t).map(|_| rng.random()).collect();
        let eq_lo_orig: Vec<EF> = (0..t).map(|_| rng.random()).collect();

        // Compute claimed sum
        let claimed_sum: EF = (0..t)
            .map(|i| eq_r_orig[i] * eq_hi_orig[i] * eq_lo_orig[i])
            .sum();

        // Prove
        let mut eq_r = eq_r_orig.clone();
        let mut eq_hi = eq_hi_orig.clone();
        let mut eq_lo = eq_lo_orig.clone();
        let mut prover_state =
            ProverState::new(get_poseidon16().clone(), Default::default());
        let (hi_eval, lo_eval, point) = prove_tensor_decomp_sumcheck(
            &mut prover_state,
            &mut eq_r,
            &mut eq_hi,
            &mut eq_lo,
            claimed_sum,
        );

        assert_eq!(point.len(), log_t);

        // Verify
        let mut verifier_state = VerifierState::<EF, _>::new(
            prover_state.into_proof(),
            get_poseidon16().clone(),
            Default::default(),
        )
        .unwrap();
        let (v_endpoint, v_point) =
            verify_tensor_decomp_sumcheck(&mut verifier_state, log_t, claimed_sum).unwrap();

        assert_eq!(point, v_point, "prover and verifier points must match");

        // The verifier endpoint is eq_r_tilde(row') * eq_hi_tilde(row') * eq_lo_tilde(row')
        let eq_r_at_point = eval_pushforward_mle(&eq_r_orig, &point);
        let eq_hi_at_point = eval_pushforward_mle(&eq_hi_orig, &point);
        let eq_lo_at_point = eval_pushforward_mle(&eq_lo_orig, &point);
        let expected_endpoint = eq_r_at_point * eq_hi_at_point * eq_lo_at_point;

        assert_eq!(
            v_endpoint, expected_endpoint,
            "endpoint must match product of evaluations"
        );
        assert_eq!(hi_eval, eq_hi_at_point, "hi_eval must match MLE evaluation");
        assert_eq!(lo_eval, eq_lo_at_point, "lo_eval must match MLE evaluation");
    }

    #[test]
    fn test_build_eq_addr_tables() {
        let mut rng = StdRng::seed_from_u64(555);
        let half_bits = 3;
        let n_rows = 16;
        let max_addr = 1usize << half_bits;

        let addr_hi: Vec<F> = (0..n_rows)
            .map(|_| F::from_usize(rng.random_range(0..max_addr)))
            .collect();
        let addr_lo: Vec<F> = (0..n_rows)
            .map(|_| F::from_usize(rng.random_range(0..max_addr)))
            .collect();
        let s_hi: Vec<EF> = (0..half_bits).map(|_| rng.random()).collect();
        let s_lo: Vec<EF> = (0..half_bits).map(|_| rng.random()).collect();

        let (eq_hi_table, eq_lo_table) =
            build_eq_addr_tables(&addr_hi, &addr_lo, &s_hi, &s_lo, half_bits);

        for row in 0..n_rows {
            let expected_hi = eq_bits_at_point(addr_hi[row], &s_hi, half_bits);
            let expected_lo = eq_bits_at_point(addr_lo[row], &s_lo, half_bits);
            assert_eq!(eq_hi_table[row], expected_hi, "eq_hi mismatch at row {row}");
            assert_eq!(eq_lo_table[row], expected_lo, "eq_lo mismatch at row {row}");
        }
    }

    #[test]
    fn test_d2_pushforward() {
        let mut rng = StdRng::seed_from_u64(777);
        let half_bits = 3;
        let s = 1usize << half_bits;
        let n_rows = 32;

        let addr_hi: Vec<F> = (0..n_rows)
            .map(|_| F::from_usize(rng.random_range(0..s)))
            .collect();
        let addr_lo: Vec<F> = (0..n_rows)
            .map(|_| F::from_usize(rng.random_range(0..s)))
            .collect();
        let eq_row_prime: Vec<EF> = (0..n_rows).map(|_| rng.random()).collect();
        let beta: EF = rng.random();

        let (p_hi, p_lo, p_batched) =
            compute_d2_pushforward(&addr_hi, &addr_lo, &eq_row_prime, half_bits, beta);

        // Verify P_hi
        for h in 0..s {
            let expected: EF = addr_hi
                .iter()
                .enumerate()
                .filter(|&(_, &a)| a.to_usize() == h)
                .map(|(row, _)| eq_row_prime[row])
                .sum();
            assert_eq!(p_hi[h], expected, "P_hi mismatch at h={h}");
        }

        // Verify P_lo
        for l in 0..s {
            let expected: EF = addr_lo
                .iter()
                .enumerate()
                .filter(|&(_, &a)| a.to_usize() == l)
                .map(|(row, _)| eq_row_prime[row])
                .sum();
            assert_eq!(p_lo[l], expected, "P_lo mismatch at l={l}");
        }

        // Verify batching
        for h in 0..s {
            assert_eq!(
                p_batched[h],
                p_hi[h] + beta * p_lo[h],
                "P_batched mismatch at h={h}"
            );
        }
    }

    #[test]
    fn test_eval_pushforward_mle() {
        let mut rng = StdRng::seed_from_u64(888);
        let n_vars = 4;
        let size = 1 << n_vars;
        let pushforward: Vec<EF> = (0..size).map(|_| rng.random()).collect();
        let point: Vec<EF> = (0..n_vars).map(|_| rng.random()).collect();

        let result = eval_pushforward_mle(&pushforward, &point);

        // Verify by computing sum_h P[h] * eq(bits(h), point)
        let expected: EF = (0..size)
            .map(|h| pushforward[h] * eq_index_at_point(h, &point, n_vars))
            .sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_fold_table() {
        let mut rng = StdRng::seed_from_u64(999);
        let size = 8;
        let mut table: Vec<EF> = (0..size).map(|_| rng.random()).collect();
        let table_orig = table.clone();
        let r: EF = rng.random();

        fold_table(&mut table, r);

        assert_eq!(table.len(), size / 2);
        for j in 0..size / 2 {
            let expected =
                table_orig[2 * j] + r * (table_orig[2 * j + 1] - table_orig[2 * j]);
            assert_eq!(table[j], expected, "fold mismatch at j={j}");
        }
    }

    /// End-to-end integration test: construct a small memory binding scenario,
    /// run the full Shout + tensor decomposition protocol, and verify consistency.
    #[test]
    fn test_full_shout_protocol() {
        let mut rng = StdRng::seed_from_u64(1234);
        let half_bits = 3;
        let memory_size = 1 << (2 * half_bits); // K = 2^6 = 64
        let n_rows = 32; // T = 32, log(T) = 5

        // Construct address column with hi/lo decomposition
        let sqrt_k = 1usize << half_bits;
        let addr_hi: Vec<F> = (0..n_rows)
            .map(|_| F::from_usize(rng.random_range(0..sqrt_k)))
            .collect();
        let addr_lo: Vec<F> = (0..n_rows)
            .map(|_| F::from_usize(rng.random_range(0..sqrt_k)))
            .collect();
        let addr_col: Vec<F> = (0..n_rows)
            .map(|i| {
                F::from_usize(addr_hi[i].to_usize() * sqrt_k + addr_lo[i].to_usize())
            })
            .collect();

        // Random eq_r (simulating eq(bits(row), r_air))
        let eq_r: Vec<EF> = (0..n_rows).map(|_| rng.random()).collect();

        // Random memory
        let memory: Vec<F> = (0..memory_size).map(|_| rng.random()).collect();
        let n_value_cols = 2;
        let gamma: EF = rng.random();

        // Step 1: Joint pushforward
        let p_joint = compute_joint_pushforward(&addr_col, memory_size, &eq_r);

        // Step 2: Batched memory
        let batched_mem = compute_batched_memory(&memory, n_value_cols, gamma);
        assert_eq!(batched_mem.len(), memory_size);

        // Compute claimed sum for Shout value sumcheck
        let claimed_sum: EF = p_joint
            .iter()
            .zip(batched_mem.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        // Step 3: Shout value sumcheck (prove + verify)
        let mut p_joint_copy = p_joint.clone();
        let mut batched_mem_copy = batched_mem[..memory_size].to_vec();
        let mut prover_state =
            ProverState::new(get_poseidon16().clone(), Default::default());

        let (endpoint, s_point) = prove_shout_value_sumcheck(
            &mut prover_state,
            &mut p_joint_copy,
            &mut batched_mem_copy,
            claimed_sum,
        );

        let mut verifier_state = VerifierState::<EF, _>::new(
            prover_state.into_proof(),
            get_poseidon16().clone(),
            Default::default(),
        )
        .unwrap();
        let (v_endpoint, v_point) = verify_shout_value_sumcheck(
            &mut verifier_state,
            2 * half_bits,
            claimed_sum,
        )
        .unwrap();

        assert_eq!(s_point, v_point);
        assert_eq!(endpoint, v_endpoint);

        // Step 4: Build eq addr tables for tensor decomposition
        let n = 2 * half_bits; // log(K)
        let s_lo = &s_point[..half_bits];
        let s_hi = &s_point[half_bits..n];

        let (eq_hi_table, eq_lo_table) =
            build_eq_addr_tables(&addr_hi, &addr_lo, s_hi, s_lo, half_bits);

        // P_joint_tilde(s) = sum_row eq_r[row] * eq_hi_table[row] * eq_lo_table[row]
        let p_joint_at_s = eval_pushforward_mle(&p_joint, &s_point);
        let tensor_sum: EF = (0..n_rows)
            .map(|row| eq_r[row] * eq_hi_table[row] * eq_lo_table[row])
            .sum();
        assert_eq!(
            tensor_sum, p_joint_at_s,
            "tensor decomposition sum must equal P_joint_tilde(s)"
        );

        // Step 5: Tensor decomposition sumcheck (prove + verify)
        let log_t = log2_ceil_usize(n_rows);
        let t_padded = 1 << log_t;
        let mut eq_r_padded = eq_r.clone();
        eq_r_padded.resize(t_padded, EF::ZERO);
        let mut eq_hi_padded = eq_hi_table.clone();
        eq_hi_padded.resize(t_padded, EF::ZERO);
        let mut eq_lo_padded = eq_lo_table.clone();
        eq_lo_padded.resize(t_padded, EF::ZERO);

        let mut prover_state2 =
            ProverState::new(get_poseidon16().clone(), Default::default());
        let (hi_eval, lo_eval, row_prime) = prove_tensor_decomp_sumcheck(
            &mut prover_state2,
            &mut eq_r_padded,
            &mut eq_hi_padded,
            &mut eq_lo_padded,
            tensor_sum,
        );

        let mut verifier_state2 = VerifierState::<EF, _>::new(
            prover_state2.into_proof(),
            get_poseidon16().clone(),
            Default::default(),
        )
        .unwrap();
        let (v_endpoint2, v_row_prime) =
            verify_tensor_decomp_sumcheck(&mut verifier_state2, log_t, tensor_sum)
                .unwrap();

        assert_eq!(row_prime, v_row_prime);

        // Check that the endpoint matches the product of individual MLE evaluations
        let eq_r_padded_orig = {
            let mut v = eq_r.clone();
            v.resize(t_padded, EF::ZERO);
            v
        };
        let eq_hi_padded_orig = {
            let mut v = eq_hi_table.clone();
            v.resize(t_padded, EF::ZERO);
            v
        };
        let eq_lo_padded_orig = {
            let mut v = eq_lo_table.clone();
            v.resize(t_padded, EF::ZERO);
            v
        };

        let eq_r_at_row = eval_pushforward_mle(&eq_r_padded_orig, &row_prime);
        let eq_hi_at_row = eval_pushforward_mle(&eq_hi_padded_orig, &row_prime);
        let eq_lo_at_row = eval_pushforward_mle(&eq_lo_padded_orig, &row_prime);
        assert_eq!(v_endpoint2, eq_r_at_row * eq_hi_at_row * eq_lo_at_row);
        assert_eq!(hi_eval, eq_hi_at_row);
        assert_eq!(lo_eval, eq_lo_at_row);

        // Step 6: d=2 pushforward
        let beta: EF = rng.random();
        let eq_row_prime_vec: Vec<EF> = (0..n_rows)
            .map(|row| eq_index_at_point(row, &row_prime, log_t))
            .collect();

        let (p_hi, p_lo, _p_batched) = compute_d2_pushforward(
            &addr_hi,
            &addr_lo,
            &eq_row_prime_vec,
            half_bits,
            beta,
        );

        // Verify: P_hi_tilde(s_hi) should equal hi_eval
        let p_hi_at_s_hi = eval_pushforward_mle(&p_hi, s_hi);
        let p_lo_at_s_lo = eval_pushforward_mle(&p_lo, s_lo);
        assert_eq!(
            p_hi_at_s_hi, hi_eval,
            "P_hi_tilde(s_hi) must equal hi_eval"
        );
        assert_eq!(
            p_lo_at_s_lo, lo_eval,
            "P_lo_tilde(s_lo) must equal lo_eval"
        );
    }
}
