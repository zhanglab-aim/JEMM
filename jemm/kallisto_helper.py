"""
A series of helpers for processing Kallisto hdf5 file
"""

# Author : zzjfrank
# Date   : Aug. 24, 2020

import h5py
import numpy as np


def count_to_tpm(est_counts, eff_len):
    """Convert estimated counts to tpm

    Returns
    -------

    Notes
    -----
    This function is based on the R sleuth function here https://rdrr.io/github/pachterlab/sleuth/src/R/units.R::

        counts_to_tpm <- function(est_counts, eff_len) {
          stopifnot( length(eff_len) == length(est_counts) )

          which_valid <- which(eff_len > 0)

          num <- (est_counts / eff_len)
          num[-which_valid] <- 0
          denom <- sum(num)

          (1e6 * num) / denom
        }
    """
    assert len(eff_len) == len(est_counts)
    which_invalid = np.where(eff_len <= 0)[0]
    num = est_counts / eff_len
    num[which_invalid] = 0
    denom = np.sum(num)
    return (1e6 * num) / denom


def read_bootstrap_hdf5(store):
    """

    Parameters
    ----------
    store

    Returns
    -------
    bs_samples : numpy.array
        A numpy array with shape (num_transcripts, num_boostrap) that holds all bootstrap TPM estimates

    Notes
    -----
    This function is partially based on the R sleuth function `.read_bootstrap_hdf5`
    here https://rdrr.io/github/pachterlab/sleuth/src/R/read_write.R
    """
    bs_samples = []
    num_bs = len(store['bootstrap'])
    eff_len = store['aux/eff_lengths'][()]
    for b in range(num_bs):
        est_counts = store['bootstrap/bs%i'%b][()]
        bs_samples.append(count_to_tpm(est_counts, eff_len))
    bs_samples = np.array(bs_samples, dtype="float16").transpose()
    return bs_samples


def read_kallisto_h5(fname):
    """

    Parameters
    ----------
    fname

    Returns
    -------

    Notes
    -----
    This function is partially based on the R sleuth function `read_kallisto_h5`
    here https://rdrr.io/github/pachterlab/sleuth/src/R/read_write.R
    """
    with h5py.File(fname, "r") as store:
        target_id = store['aux/ids'][()]
        target_id = [x.decode("utf-8").split('|')[0] for x in target_id]
        assert len(target_id) == len(set(target_id)), "target_ids in your kallisto file are not unique"
        est_counts = store['est_counts'][()]
        eff_len = store['aux/eff_lengths'][()]
        tpm = count_to_tpm(est_counts, eff_len)
        bs_samples = read_bootstrap_hdf5(store)

    target_id = {target_id[i] : i for i in range(len(target_id))}
    res = {'tpm': tpm, 'target_id': target_id, 'bs_samples': bs_samples}
    return res

