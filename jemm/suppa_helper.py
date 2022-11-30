# -*- coding: utf-8 -*-

"""The helper class for handling SUPPA on transcript-based measure
"""

# author : zzjfrank
# date: Sep. 20, 2020

import pandas as pd
from collections import OrderedDict
import numpy as np


def convert_tpm_to_psi(target_id, tpm, exon_index, minimum_sanity_checker=None, logit_transform=False):
    psi_list = []
    tpm_ = np.copy(tpm)
    if len(tpm_.shape) == 1:
        tpm_ = np.expand_dims(tpm, -1)
    for eid in exon_index[0]:
        alt_tx = [target_id[x] for x in exon_index[0][eid]]
        tot_tx = [target_id[x] for x in exon_index[1][eid]]
        i = np.apply_along_axis(np.sum, 0, tpm_[alt_tx])
        t = np.apply_along_axis(np.sum, 0, tpm_[tot_tx])
        psi = logit(i / (t + 1e-4)) if logit_transform is True else i / (t + 1e-4)
        psi[t == 0] = np.nan
        psi_list.append(psi)
    psi_list = np.array(psi_list)
    return psi_list


def get_min_sanity_checker(min_avg_psi=0.01, max_avg_psi=0.99, max_nan_ratio=0.3):
    def checker(row):
        psi = [x.psi for x in row]
        if np.mean(np.isnan(psi)) > max_nan_ratio:
            return False
        avg = np.nanmean(psi)
        if avg < min_avg_psi:
            return False
        if avg > max_avg_psi:
            return False
        return True
    return checker


def _se_and_a3_a5(s):
    gene, chrom, upstream, downstream, strand = s.split(':')
    u1, u2 = upstream.split('-')[0], str(int(upstream.split('-')[1]) - 1)
    d1, d2 = downstream.split('-')[0], str(int(downstream.split('-')[1]) - 1)
    eid = ':'.join([chrom, strand, u1, u2, d1, d2])
    return eid


def _ri(s):
    gene, chrom, upstream_es, ri, downstream_ee, strand = s.split(':')
    r1, r2 = ri.split('-')[0], str(int(ri.split('-')[1]) - 1)
    u, d = str(int(upstream_es) - 1), downstream_ee
    eid = ':'.join([chrom, strand, u, r1, r2, d])
    return eid


def suppaID_to_dartsID():
    """Convert rMATS columns to DARTS condensed ID format

    Reference
    ---------
    https://github.com/comprna/SUPPA#generation-of-transcript-events-and-local-alternative-splicing-event_ids

    TODO
    ----
    Add other types of events, i.e. MX, SS, FL
    """
    suppa_type = {
            # TODO: add other AS types
            "SE": _se_and_a3_a5,
            "A3": _se_and_a3_a5,
            "A3SS": _se_and_a3_a5,
            "A5": _se_and_a3_a5,
            "A5SS": _se_and_a3_a5,
            "RI": _ri,
    }
    return suppa_type


def read_suppa_index(fp, event_type='SE', convert_id_to_darts=True, strip_tx_version=False):
    """

    Parameters
    ----------
    fp : str
    event_type : str
    convert_id_to_darts : bool
    strip_tx_version : bool

    Returns
    -------
    index : list of two collections.OrderedDict
        A list of two elements corresponding to alternative transcripts and total transcripts.
   """
    df = pd.read_table(fp)
    alt_tx = OrderedDict({})
    tot_tx = OrderedDict({})
    darts_id_converter = suppaID_to_dartsID()[event_type]
    for i in range(df.shape[0]):
        eid = df['event_id'][i]
        if convert_id_to_darts is True:
            eid = darts_id_converter(eid)
        alt = df['alternative_transcripts'][i].split(",")
        tot = df['total_transcripts'][i].split(",")
        if strip_tx_version is True:
            alt = [x.split(".")[0] for x in alt]
            tot = [x.split(".")[0] for x in tot]
        alt_tx[eid] = alt
        tot_tx[eid] = tot
    index = [alt_tx, tot_tx]
    return index



