"""
Utilities for jemm
"""


import numpy as np
import scipy.stats
import logging


def chunkify(a, n):
    """Separate a list (a) into consecutive n chunks.
    Returns the chunkified index
    """
    k, m = len(a) / n, len(a) % n
    return (a[int(i * k + min(i, m)):int((i + 1) * k + min(i + 1, m))] for i in range(n))


def lm(y, x):
    """one-liner linear regression MLE
    """
    return np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)


def wlm(y, x, w):
    """one-liner weighted linear regresion MLE
    """
    return np.linalg.inv(x.transpose().dot(np.diag(w)).dot(x)).dot(x.transpose().dot(np.diag(w)).dot(y))


def logit(p, trim_margin=0.05):
    assert trim_margin < 0.5
    p = np.clip(p, a_min=trim_margin, a_max=1-trim_margin)
    return np.log(p/(1-p))


def sigmoid(z):
    z = np.clip(z, a_min=-100, a_max=100)
    return 1 / (1+np.exp(-z))


def fdr_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing.

    References
    ----------
    https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python/33532498#33532498

    Notes
    -----
    Based on R-code::

        BH = {
        i <- lp:1L   # lp is the number of p-values
        o <- order(p, decreasing = TRUE) # "o" will reverse sort the p-values
        ro <- order(o)
        pmin(1, cummin(n/i * p[o]))[ro]  # n is also the number of p-values
      }
    """
    p = np.asfarray(p)
    p[np.isnan(p)] = 1
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def fisher_method_combine_pvals(pvals):
    """Fisher's method for combining M tests' p-values into one single meta p-value

    The null hypothesis is that **all** of the null hypothesis (of each individual tests) are true. If either
    one rejects the null, the meta p-value is supposed to come significant. Thus, the combined meta p-value is
    usually more significant than the most significant individual p-value.

    References
    ----------
    https://en.wikipedia.org/wiki/Fisher%27s_method
    """
    # cut-off at a regular precision bound
    pvals = np.maximum(2.2e-16, pvals)
    meta_chi2 = -2 * np.sum(np.log(pvals))
    n_df = 2*len(pvals)
    pv = 1 - scipy.stats.chi2.cdf(meta_chi2, df=n_df)
    return pv


def setup_logger():
	"""Set up the logger for the whole pipeline
	"""
	# setup logger
	logger = logging.getLogger('jemm')
	logger.setLevel(logging.DEBUG)
	fh = logging.FileHandler('log.jemm.txt')
	fh.setLevel(logging.INFO)
	# create console handler with a higher log level
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	# create formatter and add it to the handlers
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	# add the handlers to the logger
	logger.addHandler(fh)
	logger.addHandler(ch)
	return logger
