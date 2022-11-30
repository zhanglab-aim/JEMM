"""This module handles the meta loading of multiple types of splicing
events into gene-level annotations
"""

# author : zzjfrank
# date : Nov 18, 2020

import os
import pickle
import pandas as pd
from collections import defaultdict
from .junction import JunctionCountTable
from .transcript import TranscriptMeasureTable
from .model import get as get_model
from .genomic_annotation import ExonSet, GeneSet
from .utils import logit, sigmoid
import time


class MetaSplicingEvent:
    """This class is to host different types of Alternative Splicing
    events in a unified container.

    Currently only supports coef to be a factor; i.e. will always use the coefficient
    to compute a unit increase on top of baseline, and get delta-PSI values.

    Parameters
    ----------
    event_type : str
    event_id : str
    coef : float
        coefficient for the factor in logit scale.
    baseline : float
        baseline (or equivalently, intercept) in logit scale

    """
    def __init__(self, event_type, event_id, coef, baseline):
        self.event_type = event_type
        self.event_id = event_id
        self.coef = coef
        self.baseline = baseline
        self._base_psi = sigmoid(baseline)
        self._delta_psi = sigmoid(baseline+coef) - self._base_psi

    def __lt__(self, other):
        return abs(self._delta_psi) < abs(other._delta_psi)

    def __le__(self, other):
        return abs(self._delta_psi) <= abs(other._delta_psi)

    def __gt__(self, other):
        return abs(self._delta_psi) > abs(other._delta_psi)

    def __ge__(self, other):
        return abs(self._delta_psi) >= abs(other._delta_psi)

    def __str__(self):
        return "%s,%s,%.3f,%.3f"%(self.event_id, self.event_type, self._base_psi, self._delta_psi)


def smart_data_table_loader(fp, as_type, measure_type):
    """reload a JunctionCountTable or TranscriptMeasureTable from storage,
    leveraging filename suffixes and patterns to determine the loader to use.

    Parameters
    ----------
    fp : str or jemm.JunctionCountTable, or jemm.TranscriptMeasureTable
    as_type : {'SE', 'A5SS', 'A3SS', 'RI'}
    measure_type : {'jct', 'txr'}

    Returns
    --------
    JunctionCountTable or TranscriptMeasureTable : reloaded data table
    """
    assert measure_type in ('jct', 'txr')
    assert as_type in ('SE', 'A5SS', 'A3SS', 'RI')
    if type(fp) is str:
        assert os.path.isfile(fp), "Cannot find file: %s" % fp
    measure_fn = JunctionCountTable if measure_type == 'jct' else TranscriptMeasureTable
    if fp is None:
        return None
    if type(fp) in (JunctionCountTable, TranscriptMeasureTable):
        return fp
    elif fp.endswith(".txt"):
        dat = measure_fn.from_plaintext(fp)
    elif fp.endswith(".pkl") or fp.endswith(".p"):
        dat = pickle.load(open(fp, "rb"))
    else:
        raise TypeError('cannot determine the loader for file: %s'%fp)
    return dat


class MetaLoader:
    """Load the data as well as regression tables (optional) for a study
    for convenient re-extraction

    Parameters
    ----------
    covariates : jemm.covariates.Covariate
    data_files : dict
        dictionary that maps type abbreviations (e.g SE, RI) to filepaths
        of data files; example layout is {'SE': {'jct': '/path/to/jct', 'txr': '/path/to/txr'}}
    reg_tables : dict
        dictionary of regression tables. layout is similar to data_files
    """
    def __init__(self, covariates, data_files, reg_tables=None, jem_type='abstract', jem_kwargs=None, verbose=True):
        self.covariates = covariates
        self.data = {}
        self.verbose = verbose
        jem_kwargs = jem_kwargs or {}
        reg_tables = reg_tables or {}
        self._jem_kwargs = jem_kwargs
        self._reg_tables = {}
        self._count_tables = {}
        for as_type in data_files:
            self._count_tables[as_type] = {}
            if self.verbose: print('loading %s..'%as_type)
            start = time.time()
            jct = smart_data_table_loader(
                    fp = data_files[as_type]['jct'],
                    as_type = as_type,
                    measure_type = 'jct'
                    )
            self._count_tables[as_type]['jct'] = jct
            txr = smart_data_table_loader(
                    fp = data_files[as_type]['txr'],
                    as_type = as_type,
                    measure_type = 'txr'
                    )
            self._count_tables[as_type]['txr'] = txr
            if self.verbose:
                try:
                    n_jct = jct.data.shape if jct else 0
                    n_txr = txr.data.shape if txr else 0
                    print('loaded n=%s jct, n=%s txr; took %.3f sec' % (n_jct, n_txr, time.time()-start))
                except:
                    pass
            self._jem_type = jem_type
            start = time.time()
            jem = get_model(jem_type)(junction_measure=jct,
                                 transcript_measure=txr,
                                 covariates=self.covariates,
                                 **jem_kwargs
                                 )
            if self.verbose:
                print('get model took %.3f sec' % (time.time()-start))
            if as_type in reg_tables:
                start = time.time()
                jem.load_regression_table(outfp=reg_tables[as_type], fast_load=True)
                self._reg_tables[as_type] = pd.read_table(reg_tables[as_type], index_col=0)
                if self.verbose:
                    print('load reg table took %.3f sec' % (time.time()-start))
            self.data[as_type] = jem

    def __str__(self):
        return "Jemm MetaLoader"

    def get_gene_centric_dict(self, cols, gene_col_name='geneSymbol', fdr_thresh=0.05):
        """
        Parameters
        ----------
        cols : list(str)
            which covariate columns to extract; these are usually the conditions/groups
        fdr_thresh : float
        """
        gene_centric_dict = {}
        reg_tables = {}
        for as_type in self.data:
            #df = pd.read_table(self._reg_tables[as_type])
            df = self._reg_tables[as_type]
            for i in range(df.shape[0]):
                row = df.iloc[i]
                eid = row.name
                gene = row[gene_col_name]
                # not precise due to string formatting at saving
                #coef_name = [x for x in row['covariate_names'].split(',')]
                #coefs = [float(x) for x in row['covariate_est'].split(',')]
                #qvals = [float(x) for x in row['covariate_qval'].split(',')]
                #reg_table = {coef_name[i]:coefs[i] for i in range(len(coef_name)) if qvals[i]<fdr_thresh and coef_name[i] in cols}
                qvals = self.data[as_type].stats_tests[eid]['qvals'].to_dict()
                coefs = self.data[as_type].stats_tests[eid]['coef'].to_dict()
                coef_name = self.data[as_type].stats_tests[eid].index.tolist()
                reg_table = {coef_name[i]:coefs[coef_name[i]] for i in range(len(coef_name)) if qvals[coef_name[i]]<fdr_thresh and coef_name[i] in cols}
                baseline = coefs[ 'intercept' ]
                if not gene in gene_centric_dict:
                    gene_centric_dict[gene] = {}
                for cond in reg_table:
                    meta_evt = MetaSplicingEvent(
                                        event_type=as_type,
                                        event_id=eid,
                                        coef=reg_table[cond],
                                        baseline=baseline
                                    )
                    if cond in gene_centric_dict[gene]:
                        gene_centric_dict[gene][cond].append(meta_evt)
                    else:
                        gene_centric_dict[gene][cond] = [ meta_evt ]
        return gene_centric_dict


