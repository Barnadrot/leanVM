use fiat_shamir::*;
use field::*;
use poly::*;

use crate::univariate_skip::{coset_sum, horner, lagrange_evals_on_subgroup};

/// Verifier-side artifacts of the univariate skip round (plan_spec v2 §4.1).
#[derive(Debug, Clone)]
pub struct UnivariateSkipRound<EF> {
    pub r0: EF,
    /// `v0(r0)` — the claimed sum entering the remaining multivariate rounds.
    pub target: EF,
    /// `lagrange_evals_on_subgroup(r0, k)` — `2^k` values, reused for the
    /// public weights `κ_t = A_t(r0)` and the conversion weights `ℓ_t`.
    pub lagrange_on_d: Vec<EF>,
}

/// Verifies the univariate skip round (plan_spec v2 §2.1 steps 1–2): reads the
/// `n_uni_coeffs` coefficients of `v0`, checks the coset-sum identity
/// `Σ_{y∈D} v0(y) == expected_sum`, samples `r0` and evaluates the new target.
///
/// `r0 ∈ D` (i.e. `r0^{2^k} = 1`) would make the Lagrange weights divide by
/// zero; it is rejected as `InvalidProof` (completeness error ≤ 2^k/|EF|,
/// plan_spec v2 §4 edge case).
pub fn verify_univariate_skip_round<EF: ExtensionField<PF<EF>>>(
    verifier_state: &mut impl FSVerifier<EF>,
    skip_k: usize,
    n_uni_coeffs: usize,
    expected_sum: EF,
) -> Result<UnivariateSkipRound<EF>, ProofError>
where
    PF<EF>: TwoAdicField,
{
    let coeffs = verifier_state.next_extension_scalars_vec(n_uni_coeffs)?;
    if coset_sum(&coeffs, skip_k) != expected_sum {
        return Err(ProofError::InvalidProof);
    }
    let r0: EF = verifier_state.sample();
    if r0.exp_power_of_2(skip_k) == EF::ONE {
        return Err(ProofError::InvalidProof);
    }
    let target = horner(&coeffs, r0);
    let lagrange_on_d = lagrange_evals_on_subgroup::<PF<EF>, EF>(r0, skip_k);
    Ok(UnivariateSkipRound {
        r0,
        target,
        lagrange_on_d,
    })
}

pub fn sumcheck_verify<EF: ExtensionField<PF<EF>>>(
    verifier_state: &mut impl FSVerifier<EF>,
    n_vars: usize,
    degree: usize,
    expected_sum: EF,
    eq_alphas: Option<&[EF]>,
) -> Result<Evaluation<EF>, ProofError> {
    let mut target = expected_sum;
    let mut challenges = Vec::with_capacity(n_vars);

    for round in 0..n_vars {
        let eq_alpha = eq_alphas.map(|a| a[round]);
        let coeffs = verifier_state.next_sumcheck_polynomial(degree + 1, target, eq_alpha)?;
        let pol = DensePolynomial::new(coeffs);

        let challenge = verifier_state.sample();
        challenges.push(challenge);

        target = pol.evaluate(challenge);
    }

    Ok(Evaluation::new(challenges, target))
}
