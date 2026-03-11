"""
Shared test fixtures for insurance-mediation.

All synthetic data is generated with fixed random seeds so tests
are deterministic.

DATA GENERATION APPROACH
-------------------------
We generate synthetic insurance data with a KNOWN true causal structure
so we can verify estimator properties (sign, approximate magnitude)
against ground truth.

True DGP:
  C ~ Uniform(0, 10)           (confounder, e.g., driver age)
  A ~ Bernoulli(0.5)           (treatment, binary: postcode group 0 vs 1)
  M = 2 + 1.5*A + 0.3*C + eps_M   (mediator, continuous: IMD-like score)
  log(mu) = -3 + 0.2*A + 0.1*M + 0.05*C + log(exposure)
  Y ~ Poisson(mu * exposure)

True effects (all on log scale):
  TE  = 0.2*1 + 0.1*1.5 = 0.35
  NIE = 0.1 * 1.5 = 0.15  (beta_M_Y * beta_A_M)
  NDE = 0.2                (direct effect of A on Y)
  CDE(m) = 0.2             (no interaction, so CDE doesn't depend on m)
"""

import numpy as np
import pandas as pd
import pytest


def make_poisson_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Poisson claim count data with known causal structure."""
    rng = np.random.default_rng(seed)

    # Confounder
    C = rng.uniform(0, 10, n)
    # Binary treatment
    A = rng.integers(0, 2, n).astype(float)
    # Continuous mediator
    M = 2 + 1.5 * A + 0.3 * C + rng.normal(0, 1, n)
    # Exposure (policies)
    exposure = rng.uniform(0.5, 2.0, n)
    # Log mean
    log_mu = -3 + 0.2 * A + 0.1 * M + 0.05 * C + np.log(exposure)
    mu = np.exp(log_mu)
    # Claim counts
    Y = rng.poisson(mu)

    return pd.DataFrame({
        "treatment": A,
        "mediator": M,
        "outcome": Y,
        "confounder": C,
        "exposure": exposure,
    })


def make_gaussian_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Gaussian data for benchmarking."""
    rng = np.random.default_rng(seed)

    C = rng.uniform(0, 10, n)
    A = rng.integers(0, 2, n).astype(float)
    M = 2 + 1.5 * A + 0.3 * C + rng.normal(0, 1, n)
    Y = 5 + 0.2 * A + 0.1 * M + 0.05 * C + rng.normal(0, 0.5, n)

    return pd.DataFrame({
        "treatment": A,
        "mediator": M,
        "outcome": Y,
        "confounder": C,
    })


def make_binary_mediator_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate data with a binary mediator (e.g., flood zone indicator)."""
    rng = np.random.default_rng(seed)

    C = rng.uniform(0, 10, n)
    A = rng.integers(0, 2, n).astype(float)
    # Logistic mediator: higher A and C -> more likely in flood zone
    p_M = 1 / (1 + np.exp(-(-2.0 + 1.5 * A + 0.1 * C)))
    M = rng.binomial(1, p_M).astype(float)
    exposure = rng.uniform(0.5, 2.0, n)
    log_mu = -3 + 0.3 * A + 0.5 * M + 0.05 * C + np.log(exposure)
    mu = np.exp(log_mu)
    Y = rng.poisson(mu)

    return pd.DataFrame({
        "treatment": A,
        "mediator": M,
        "outcome": Y,
        "confounder": C,
        "exposure": exposure,
    })


def make_gamma_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Gamma severity data."""
    rng = np.random.default_rng(seed)

    C = rng.uniform(0, 10, n)
    A = rng.integers(0, 2, n).astype(float)
    M = 2 + 1.5 * A + 0.3 * C + rng.normal(0, 1, n)
    log_mu = 6 + 0.3 * A + 0.1 * M + 0.02 * C
    mu = np.exp(log_mu)
    # Gamma with shape = 2 (moderate dispersion)
    shape = 2.0
    scale = mu / shape
    Y = rng.gamma(shape, scale)

    return pd.DataFrame({
        "treatment": A,
        "mediator": M,
        "outcome": Y,
        "confounder": C,
    })


@pytest.fixture
def poisson_data():
    return make_poisson_data()


@pytest.fixture
def gaussian_data():
    return make_gaussian_data()


@pytest.fixture
def binary_mediator_data():
    return make_binary_mediator_data()


@pytest.fixture
def gamma_data():
    return make_gamma_data()
