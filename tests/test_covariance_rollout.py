"""Unit tests for covariance propagation through DoubleIntegrator."""

import torch
import pytest

from dynamics import DoubleIntegrator


@pytest.fixture
def setup():
    dyn = DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)
    T, nx, nu = 10, dyn.nx, dyn.nu
    mu0    = torch.tensor([0.0, 0.0, 1.0, 0.0])
    Sigma0 = torch.diag(torch.tensor([0.1, 0.1, 0.05, 0.05]))
    V      = torch.randn(T, nu) * 0.1
    K_zero = torch.zeros(T, nu, nx)
    return dyn, T, nx, nu, mu0, Sigma0, V, K_zero


def test_open_loop_output_shapes(setup):
    """mu_trace: [1, T+1, nx]   cov_trace: [1, T+1, nx, nx]."""
    dyn, T, nx, nu, mu0, Sigma0, V, _ = setup
    mu_trace, cov_trace = dyn(V, mu0, Sigma0)
    assert mu_trace.shape  == (1, T + 1, nx)
    assert cov_trace.shape == (1, T + 1, nx, nx)


def test_closed_loop_output_shapes(setup):
    dyn, T, nx, nu, mu0, Sigma0, V, _ = setup
    K = torch.randn(T, nu, nx) * 0.01
    mu_trace, cov_trace = dyn(V, mu0, Sigma0, K=K)
    assert mu_trace.shape  == (1, T + 1, nx)
    assert cov_trace.shape == (1, T + 1, nx, nx)


def test_closed_loop_k0_matches_open_loop(setup):
    """DoubleIntegrator(K=0) must exactly match open-loop covariance."""
    dyn, T, nx, nu, mu0, Sigma0, V, K_zero = setup
    mu_ol,  cov_ol  = dyn(V, mu0, Sigma0)
    mu_cl,  cov_cl  = dyn(V, mu0, Sigma0, K=K_zero)
    torch.testing.assert_close(mu_ol,  mu_cl,  atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cov_ol, cov_cl, atol=1e-6, rtol=1e-5)


def test_covariance_is_psd_at_every_step(setup):
    """All Σ_t must be positive semi-definite."""
    dyn, T, nx, nu, mu0, Sigma0, V, _ = setup
    K = torch.randn(T, nu, nx) * 0.05
    _, cov_trace = dyn(V, mu0, Sigma0, K=K)
    for t in range(T + 1):
        eigs = torch.linalg.eigvalsh(cov_trace[0, t])
        assert (eigs >= -1e-6).all(), f"Negative eigenvalue at t={t}: {eigs}"


def test_K_grad_flows(setup):
    """∂(tr Σ_T)/∂K must be nonzero — gradient path through covariance is live."""
    dyn, T, nx, nu, mu0, Sigma0, V, _ = setup
    K = torch.nn.Parameter(torch.randn(T, nu, nx) * 0.01)
    _, cov_trace = dyn(V, mu0, Sigma0, K=K)
    loss = torch.trace(cov_trace[0, -1])
    loss.backward()
    assert K.grad is not None, "K.grad is None"
    assert K.grad.abs().sum() > 0, "K.grad is all zeros"


def test_V_grad_flows(setup):
    """∂(μ_T sum)/∂V must be nonzero — gradient path through mean is live."""
    dyn, T, nx, nu, mu0, Sigma0, V_data, _ = setup
    V = torch.nn.Parameter(V_data.clone())
    mu_trace, _ = dyn(V, mu0, Sigma0)
    loss = mu_trace[0, -1].sum()
    loss.backward()
    assert V.grad is not None, "V.grad is None"
    assert V.grad.abs().sum() > 0, "V.grad is all zeros"


def test_nonzero_K_changes_covariance(setup):
    """A nonzero K must produce a different covariance trace than K=0."""
    dyn, T, nx, nu, mu0, Sigma0, V, K_zero = setup
    K_nonzero = torch.zeros(T, nu, nx)
    K_nonzero[:, 0, 0] = -0.2   # ax feeds back on px
    K_nonzero[:, 1, 1] = -0.2

    _, cov_zero = dyn(V, mu0, Sigma0, K=K_zero)
    _, cov_k    = dyn(V, mu0, Sigma0, K=K_nonzero)

    trace_zero = (cov_zero[0, -1, 0, 0] + cov_zero[0, -1, 1, 1]).item()
    trace_k    = (cov_k[0,    -1, 0, 0] + cov_k[0,    -1, 1, 1]).item()
    assert abs(trace_zero - trace_k) > 1e-6, "K had no effect on covariance"
