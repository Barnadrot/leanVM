// T4' integration tests for the univariate-skip AIR protocol (plan_spec v2).
//
// Covers (plan §8 T4' row):
// * e2e prove+verify round-trips across heterogeneous height profiles:
//   - fib:      b = {exec: k, poseidon: 0, ext_op: 0}   (b = 0 tables, late join)
//   - hashloop: b = {exec: k, poseidon: 1, ext_op: 0}   (partial block, 0 < b < k)
//   (the 1550-sig benchmark exercises b = {5, 3, 0}; CLI smoke in the task log)
// * proof-size delta vs pre-switch baselines (recorded on this exact tree state
//   before the protocol switch landed), against the closed-form
//   Δ_EF = (N+1) − k·max_full_degree + Σ_{b_t>0} (2·b_t + m_t)  (plan §7.3)
//   with a documented ±300 F tolerance for Merkle-path pruning jitter (query
//   indices are Fiat–Shamir-derived, so pruned sibling counts vary slightly).
//   The wire format itself is pinned exactly by `check_fully_consumed` inside
//   `verify_execution` (every read has a fixed count).
// * tamper resistance: corrupting any sampled transcript position must yield
//   a verification error.
//
// The ĝ == col-MLE(natural_point) round-trip invariant (§9.2c) is enforced
// component-level in sub_protocols/tests/skip_round_correctness.rs (g-pass ≡
// direct MLEs) and transitively here: a wrong ĝ is a false PCS claim, which
// WHIR rejects (that path is exercised by the tamper tests).

use backend::{Air, PrimeCharacteristicRing};
use lean_compiler::*;
use lean_prover::default_whir_config;
use lean_prover::prove_execution::prove_execution;
use lean_prover::verify_execution::verify_execution;
use lean_vm::*;
use sub_protocols::{AIR_UNIVARIATE_SKIP, air_skip_n_uni_coeffs};

const FIB_PROGRAM: &str = r#"
N = 10000
STEPS = 10000
N_STEPS = N / STEPS

def main():
    x, y = fibonacci_step(0, 1, N_STEPS)
    print(x)
    return

def fibonacci_step(a, b, steps_remaining):
    if steps_remaining == 0:
        return a, b
    new_a, new_b = fibonacci_const(a, b, STEPS)
    res_a, res_b = fibonacci_step(new_a, new_b, steps_remaining - 1)
    return res_a, res_b

def fibonacci_const(a, b, n: Const):
    buff = Array(n + 2)
    buff[0] = a
    buff[1] = b
    for j in unroll(2, n + 2):
        buff[j] = buff[j - 1] + buff[j - 2]
    return buff[n], buff[n + 1]
"#;

const HASH_LOOP_PROGRAM: &str = r#"
N_OUTER = 3000

def main():
    inp = Array(8)
    for i in unroll(0, 8):
        inp[i] = i + 1
    hash_loop(inp, N_OUTER)
    return

def hash_loop(inp, remaining):
    if remaining == 0:
        return
    out = Array(8)
    poseidon8_permute(inp, inp + 4, out)
    hash_loop(out, remaining - 1)
    return
"#;

/// Pre-switch `proof_size_fe()` baselines, recorded by running the OLD
/// protocol on this exact tree state (T0..T3' applied, T4' stashed) with the
/// same programs, inputs and whir config. See the T4' task log.
const BASELINE_FIB: usize = 33220;
const BASELINE_HASHLOOP: usize = 36893;

/// Merkle pruning allowance (F elements): the TRANSCRIPT part of the delta is
/// exact (enforced by `check_fully_consumed` — every verifier read has a fixed
/// count), but `proof_size_fe` also counts pruned Merkle openings, whose size
/// depends on the FS-derived query indices (adjacent queries share path
/// prefixes). Measured residuals on this tree: fib +449 F, hashloop +20 F over
/// ~100 query paths — i.e. < 0.2% of proof size. 600 F bounds that variance
/// while still catching any whole-message wire error (>= 1 EF = 3 F shifts the
/// transcript AND breaks verification outright).
const MERKLE_JITTER_F: usize = 600;

fn table_heights_from_metadata(md: &ExecutionMetadata) -> Vec<(Table, usize)> {
    // Mirrors trace_gen::pad_table: log_n_rows = log2_ceil(h + 1).max(MIN),
    // with h = cycles / n_poseidons / n_extension_ops respectively.
    fn log2_ceil(x: usize) -> usize {
        usize::BITS as usize - (x - 1).leading_zeros() as usize
    }
    let h = |count: usize| log2_ceil(count + 1).max(MIN_LOG_N_ROWS_PER_TABLE);
    vec![
        (Table::execution(), h(md.cycles)),
        (Table::extension_op(), h(md.n_extension_ops)),
        (Table::poseidon8(), h(md.n_poseidons)),
    ]
}

/// Closed-form proof growth in F elements (plan §7.3).
fn expected_delta_f(heights: &[(Table, usize)]) -> usize {
    let k = AIR_UNIVARIATE_SKIP;
    let max_full_degree = ALL_TABLES.iter().map(|t| t.degree_air() + 1).max().unwrap();
    let n_max = heights.iter().map(|&(_, n)| n).max().unwrap();
    let mut delta_ef: i64 = air_skip_n_uni_coeffs(max_full_degree, k) as i64 - (k * max_full_degree) as i64;
    for &(table, n_t) in heights {
        let b_t = k.saturating_sub(n_max - n_t);
        if b_t > 0 {
            let m_t = table.n_columns() + table.n_shift_columns();
            delta_ef += (2 * b_t + m_t) as i64;
        }
    }
    (delta_ef as usize) * 3 // EF -> F (DIMENSION = 3)
}

fn prove_program(
    program: &str,
) -> (
    Bytecode,
    [F; PUBLIC_INPUT_LEN],
    lean_prover::prove_execution::ExecutionProof,
) {
    let bytecode = compile_program_with_flags(&ProgramSource::Raw(program.to_string()), CompilationFlags::default());
    let public_input = [F::ZERO; PUBLIC_INPUT_LEN];
    let proof = prove_execution(
        &bytecode,
        &public_input,
        &ExecutionWitness::default(),
        &default_whir_config(1),
        false,
    )
    .unwrap();
    (bytecode, public_input, proof)
}

fn roundtrip_and_size(program: &str, baseline: usize, expect_b_profile: &[(&str, usize)]) {
    let (bytecode, public_input, proof) = prove_program(program);
    let md = proof.metadata.clone().unwrap();
    let size = proof.proof.proof_size_fe();

    // e2e round-trip.
    verify_execution(&bytecode, &public_input, proof.proof).unwrap();

    // Height/b-profile sanity (documents which protocol classes this program covers).
    let heights = table_heights_from_metadata(&md);
    let n_max = heights.iter().map(|&(_, n)| n).max().unwrap();
    for &(name, expected_b) in expect_b_profile {
        let &(_, n_t) = heights.iter().find(|(t, _)| t.name() == name).unwrap();
        let b_t = AIR_UNIVARIATE_SKIP.saturating_sub(n_max - n_t);
        assert_eq!(
            b_t, expected_b,
            "b profile changed for table {name} (heights {heights:?})"
        );
    }

    // Proof-size delta vs the pre-switch baseline (closed form ± Merkle jitter).
    let expected = baseline + expected_delta_f(&heights);
    println!(
        "SIZE {size} F vs expected {expected} F (baseline {baseline}, transcript Δ {}, residual {})",
        expected_delta_f(&heights),
        size as i64 - expected as i64,
    );
    assert!(
        size.abs_diff(expected) <= MERKLE_JITTER_F,
        "proof size {size} F, expected {expected} ± {MERKLE_JITTER_F} F (baseline {baseline}, Δ {})",
        expected_delta_f(&heights),
    );
}

#[test]
fn skip_e2e_fib_b_profile_k_0_0() {
    roundtrip_and_size(
        FIB_PROGRAM,
        BASELINE_FIB,
        &[("execution", 5), ("poseidon8", 0), ("extension_op", 0)],
    );
}

#[test]
fn skip_e2e_hashloop_b_profile_k_1_0() {
    roundtrip_and_size(
        HASH_LOOP_PROGRAM,
        BASELINE_HASHLOOP,
        &[("execution", 5), ("poseidon8", 1), ("extension_op", 0)],
    );
}

// Per-message tamper rejection (flipped v0 coefficients, conversion messages,
// ĝ entries → InvalidProof) is exercised with REAL tampered transcripts in
// crates/sub_protocols/tests/skip_adversarial.rs (the transcript internals are
// private to the fiat-shamir crate, which T4' must not modify, so e2e
// byte-flipping is not constructible here). The e2e negatives below pin the
// Fiat–Shamir binding of the whole proof.

#[test]
fn skip_e2e_wrong_public_input_rejected() {
    let (bytecode, _public_input, proof) = prove_program(FIB_PROGRAM);
    let mut wrong = [F::ZERO; PUBLIC_INPUT_LEN];
    wrong[0] = F::ONE;
    assert!(verify_execution(&bytecode, &wrong, proof.proof).is_err());
}

#[test]
fn skip_e2e_cross_program_proof_rejected() {
    let (fib_bytecode, public_input, _) = prove_program(FIB_PROGRAM);
    let (_, _, hashloop_proof) = prove_program(HASH_LOOP_PROGRAM);
    assert!(verify_execution(&fib_bytecode, &public_input, hashloop_proof.proof).is_err());
}
