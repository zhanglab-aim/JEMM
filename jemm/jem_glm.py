"""Joint Exon Model that considers RNA-seq read counts for Junctions and Bootstrapping variances for
Transcript Measurements.

Moved the GLM model development to a new script file
"""


from .model import JointExonModel
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from tqdm import tqdm
from collections import OrderedDict
from multiprocessing import Pool
import pickle
import warnings
import copy
from .utils import sigmoid, fdr_bh, logit


def neg_loglik_junction_measure(beta, data):
    y, N, x = data['y.junc'], data['N.junc'], data['x.junc']
    assert len(beta) == x.shape[1]  # p : parameters
    assert len(N) == len(y) == x.shape[0]  # n : samples
    assert all(N - y >= 0)
    z = np.matmul(x, beta)
    z = np.clip(z, a_min=-5, a_max=5)
    # ll = np.sum(y * z - N * np.log(1 + np.exp(z)))
    # approximate for large Z: log(1+e^z) = log(e^z*(1+e^(-z))) = z + log(1+e^-z) ~= z + e^-z
    # reference here: https://math.stackexchange.com/questions/2744553/how-can-i-evaluate-this-function-numerically-stable
    log_1p_expz = np.array([np.log(1 + np.exp(z_)) if z_ < 10 else z_ + np.exp(-z_) for z_ in z])
    ll = np.mean(y * z - N * log_1p_expz)
    return - ll


def grad_junction_measure(beta, data):
    y, N, x = data['y.junc'], data['N.junc'], data['x.junc']
    z = np.matmul(x, beta)
    z = np.clip(z, a_min=-5, a_max=5)
    gr = (y - N * sigmoid(z)).transpose().dot(x)
    gr /= x.shape[0]
    return - gr


def hess_junction_measure(beta, data):
    y, N, x = data['y.junc'], data['N.junc'], data['x.junc']
    z = np.matmul(x, beta)
    # z = np.clip(z, a_min=-5, a_max=5)
    p = sigmoid(z)
    var_diag = np.diag(-N * p * (1 - p))
    n = x.shape[0]
    x_xt = np.matmul(x.transpose().dot(var_diag), x) / n
    return x_xt


def est_sigma_cond_tau(beta, data):
    """Estimate sigma2 given tau2
    """
    y, tau2, x = data['y.tpm'], data['tau2.tpm'], data['x.tpm']
    z = np.matmul(x, beta)
    tau2_inv = 1. / (1 + tau2)
    sigma2 = np.sum(tau2_inv * (y - z) ** 2) / max(x.shape[0] - x.shape[1], 1)  # unbiased estimator
    return sigma2


def neg_loglik_transcript_measure(beta, data):
    y, tau2, x = data['y.tpm'], data['tau2.tpm'], data['x.tpm']
    assert len(beta) == x.shape[1]  # p : parameters
    assert len(y) == x.shape[0]  # n : samples
    # assert all(tau2 >= 0 | np.isnan(tau2))
    z = np.matmul(x, beta)
    ll = -0.5 * np.mean(1. / (1 + tau2) * (y - z) ** 2)
    return - ll


def grad_transcript_measure(beta, data):
    y, tau2, x = data['y.tpm'], data['tau2.tpm'], data['x.tpm']
    z = np.matmul(x, beta)
    gr = (1. / (1 + tau2) * (y - z)).dot(x)
    gr /= x.shape[0]
    return - gr


def hess_transcript_measure(beta, data):
    y, tau2, x = data['y.tpm'], data['tau2.tpm'], data['x.tpm']
    sigma2 = est_sigma_cond_tau(beta, data)
    tau2_inv = np.diag(-1. / ((1 + tau2) * sigma2))
    n = x.shape[0]
    x_xt = np.matmul(x.transpose().dot(tau2_inv), x) / n
    return x_xt


def neg_loglik_joint_exon_measure(beta, data):
    if "y.junc" in data:
        jct_nll = neg_loglik_junction_measure(beta, data)
    else:
        jct_nll = 0
    if "y.tpm" in data:
        txr_nll = neg_loglik_transcript_measure(beta, data)
    else:
        txr_nll = 0
    return jct_nll + txr_nll


def grad_joint_exon_measure(beta, data):
    gr_junc = grad_junction_measure(beta, data) if 'y.junc' in data else 0
    gr_tpm = grad_transcript_measure(beta, data) if 'y.tpm' in data else 0
    gr = gr_junc + gr_tpm
    return gr


def mle_var_joint_exon_measure(beta, data, verbose=0):
    """Asymptotic variance for maximum likelihood estimates
    Slutzky's theorem says: sqrt(n)*(theta_mle - theta)/sqrt(1/I(theta_mle)) --> N(0, 1)
    I(.) is the Fisher information evaluated at theta_mle

    Reference
    ---------
    http://www.utstat.toronto.edu/~brunner/oldclass/appliedf11/handouts/2101f11Wald.pdf
    """
    if "y.junc" in data:
        hess_junc = hess_junction_measure(beta, data) * data['x.junc'].shape[0]
    else:
        hess_junc = 0
    if "y.tpm" in data:
        hess_tpm = hess_transcript_measure(beta, data) * data['x.tpm'].shape[0]
    else:
        hess_tpm = 0
    hess = hess_junc + hess_tpm
    try:
        hess_inv = np.linalg.inv(hess)
        est_var = - np.diag(hess_inv)
        est_var[est_var < 0] = np.inf
    except np.linalg.LinAlgError:
        if verbose: print("Singular Hessian matrix")
        est_var = np.inf
    # est_var = - 1. / np.diag(hess)
    est_var = np.clip(est_var, a_min=1e-8, a_max=1e4)
    return est_var


def epsilon_grad(beta, data, func, epsilon=1e-8):
    """Approximate accuracy checks

    Parameters
    ----------
    beta
    data
    func
    epsilon

    Returns
    -------

    """
    f0 = func(beta, data)
    grad = []
    for i in range(len(beta)):
        beta_ = np.copy(beta)
        beta_[i] += epsilon
        f1 = func(beta_, data)
        beta_ = np.copy(beta)
        beta_[i] -= epsilon
        f2 = func(beta_, data)
        grad.append(((f1 - f0) + (f0 - f2)) / 2. / epsilon)
    return grad


class JemmGLM(JointExonModel):
    def optimize(self, data, test_index=None):
        assert data is not None and len(data) > 0
        p = data['x.junc'].shape[1] if 'x.junc' in data else data['x.tpm'].shape[1]
        x_mean = data['x.junc'].mean(axis=0) if 'x.junc' in data else data['x.tpm'].mean(axis=0)
        beta_bounds = 2.95 / (np.abs(x_mean)+0.1)
        res = scipy.optimize.minimize(
            x0=np.zeros(p),
            bounds=[(-beta_bounds[i], beta_bounds[i]) for i in range(p)],
            fun=neg_loglik_joint_exon_measure,
            args=(data,),
            method='l-bfgs-b',
            options={'disp': False},
            jac=grad_joint_exon_measure,
        )
        opt_val = res.fun
        beta_mle = res.x
        mle_var = mle_var_joint_exon_measure(beta_mle, data)
        chi2_stats = beta_mle**2 / mle_var
        pvals = 1 - scipy.stats.chi2.cdf(chi2_stats, df=1)
        wald_test = {'coef': beta_mle, 'var': mle_var, 'chi2.wald': chi2_stats, 'pvals': pvals}
        if test_index is None:
            test_index = self.covariates.columns.tolist()
            test_index = test_index + ["_junc_diff_"] if len(pvals) == self.covariates.shape[1] + 1 else test_index
        wald_test = pd.DataFrame(wald_test, index=test_index)
        return wald_test, opt_val
