"""
This module handles reading and writing related tasks for transcript-based exon measure
"""

# Author  :   zzjfrank
# Date    :   Aug 24, 2020

import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .data_table import DataTable, Measure
from .suppa_helper import read_suppa_index, convert_tpm_to_psi, get_min_sanity_checker
from . import kallisto_helper
from .utils import logit


class TranscriptMeasure(Measure):
    def __init__(self, event_id, sample_id, est_psi, bs_psi=None, var_psi=None, var_logit_psi=None, *args, **kwargs):
        super().__init__(event_id, sample_id, est_psi, var_logit_psi, *args, **kwargs)
        self.event_id = event_id
        self.sample_id = sample_id
        self.__psi = est_psi
        self.__logit_psi = logit(self.__psi)
        if np.isnan(self.__psi):
            self.__var_psi = np.nan
            self.__var_logit_psi = np.nan
        else:
            if bs_psi is not None:
                valid_bs = np.where(~np.isnan(bs_psi))[0]
                if len(valid_bs) <= 3:
                    self.__var_psi = np.nan
                    self.__var_logit_psi = np.nan
                else:
                    self.__var_psi = np.var(bs_psi[valid_bs])
                    self.__var_logit_psi = np.var(logit(bs_psi[valid_bs]))
            else:
                self.__var_psi = var_psi
                self.__var_logit_psi = var_logit_psi

    @property
    def psi(self):
        return self.__psi

    @property
    def logit_psi(self):
        return self.__logit_psi

    @property
    def var_logit_psi(self):
        return self.__var_logit_psi

    def to_plaintext(self):
        s = "%.3f;%.3f,%.3f" % (self.psi, self.var_psi, self.var_logit_psi)
        return s

    @staticmethod
    def from_plaintext(s):
        # first assume the presence of variance terms
        try:
            psi, vares = s.split(";")
            var, logit_var = vares.split(",")
            return TranscriptMeasure(event_id=None, sample_id=None, est_psi=float(psi),
                    var_psi=float(var), var_logit_psi=float(logit_var))
        # otherwise, just load PSI
        except (ValueError, AttributeError):
            psi = s
            return float(psi)
            #return TranscriptMeasure(event_id=None, sample_id=None, est_psi=float(psi),
            #        var_psi=None, var_logit_psi=None)

    @staticmethod
    def from_rsem(s):
        psi = float(s)
        return TranscriptMeasure(event_id=None, sample_id=None, est_psi=psi,
                var_psi=0, var_logit_psi=0)


class TranscriptMeasureTable(DataTable):
    def __init__(self, wd=None, index_file=None, input_type=None, event_type="SE", lazy_init=False, plaintext=False):
        """This class provides easy access to TranscriptMeasure across large datasets

        Parameters
        ----------
        wd : str
        index_file : str
        input_type : str
        event_type : str
        lazy_init : bool

        Attributes
        ----------
        plaintext : bool
        data : pandas.DataFrame

        Examples
        --------
        Initialize a TranscriptMeasureTable from Kallisto bootstrapping results::

            >>> from jemm.transcript import TranscriptMeasureTable
            >>> tmt = TranscriptMeasureTable(wd="./data/jemm/kallisto/",
            >>>       index_file="./data/jemm/suppa_index/suppa_gencodev34_SE_strict.ioe", input_type="kallisto")
            >>> tmt.data.head()
            >>> tmt.save("./data/jemm/tmt.pkl")

        """
        self._wd = wd
        self._input_type = input_type
        self._index_file = index_file
        self._event_type = event_type
        self.data = None
        self.plaintext = plaintext

        if lazy_init is False:
            assert os.path.isdir(self._wd)
            assert os.path.isfile(self._index_file)
            if self._input_type == "kallisto":
                self.from_kallisto_hdf5(wd=self._wd, event_type=self._event_type, index_file=self._index_file, tmt=self)
            elif self._input_type == "plaintext":
                self.from_plaintext(filepath=wd, tmt=self)
            elif self._input_type == "rsem":
                self.from_rsem_table(filepath=wd, tmt=self)
            else:
                raise ValueError("Input type not understood: %s" % self._input_type)

    def save(self, fp, mode="auto"):
        if mode == "auto":
            if fp.endswith(".pkl"):
                mode = "pickle"
            elif fp.endswith(".h5"):
                mode = "hdf5"
            elif fp.endswith(".txt") or fp.endswith(".gz"):
                mode = "txt"
            else:
                raise ValueError("Cannot determine mode for given filepath")
        if mode == "pickle":
            pickle.dump(self, open(fp, "wb"))
        elif mode == "hdf5":
            self.data.to_hdf(fp, key="data", mode="w")
        elif mode == "txt":
            self.data.to_csv(fp, sep="\t")

    @staticmethod
    def _worker_reader(args):
        """worker reader for kallisto hdf5 to enable multiprocessing
        """
        sn, fp, exon_index= args
        event_ids = [e for e in exon_index[0]]
        if type(fp) is str:
            res = kallisto_helper.read_kallisto_h5(fname=fp)
        else:
            res = fp
        est_psi = convert_tpm_to_psi(target_id=res['target_id'], tpm=res['tpm'], exon_index=exon_index).flatten()
        bs_psi = convert_tpm_to_psi(target_id=res['target_id'], tpm=res['bs_samples'], exon_index=exon_index)
        sample_tm = [
            TranscriptMeasure(sample_id=sn, event_id=event_ids[j], est_psi=est_psi[j], bs_psi=bs_psi[j])
            for j in range(len(est_psi))
        ]
        return sn, sample_tm

    @staticmethod
    def from_plaintext(filepath, tmt=None):
        data = pd.read_table(filepath, index_col=0)
        data = data.applymap(lambda x: TranscriptMeasure.from_plaintext(x))
        if tmt is None:
            tmt = TranscriptMeasureTable(input_type="plaintext", lazy_init=True)
        tmt.data = data
        return tmt

    @staticmethod
    def from_rsem_table(filepath, tmt=None):
        data = pd.read_table(filepath, index_col=0)
        data = data.applymap(lambda x: TranscriptMeasure.from_rsem(x))
        if tmt is None:
            tmt = TranscriptMeasureTable(input_type="rsem", lazy_init=True)
        tmt.data = data
        return tmt

    @staticmethod
    def from_kallisto_hdf5(wd, index_file, event_type, sample_name_getter=None, tmt=None, nthreads=None, strip_tx_version=False, minimum_sanity_checker=None):
        """read kallisto transcript estimates

        Parameters
        ----------
        wd : str or dict
            if is string, expects the filepath to a folder of kallisto results; if dict, expects a mapping from sample name to hdf5 estimates
        index_file : str
        event_type : str
        sample_name_getter : callable, or None
        tmt : jemm.TranscriptMeasureTable
        nthreads : int or None
        """
        if nthreads is None:
            nthreads = min(32, cpu_count())
            print("Using n=%i threads automatically.."%nthreads)
        exon_index = read_suppa_index(fp=index_file, event_type=event_type, convert_id_to_darts=True, strip_tx_version=strip_tx_version)
        event_ids = [e for e in exon_index[0]]

        if type(wd) is str:
            fp_list = [os.path.join(wd, x, "abundance.h5")
                       for x in os.listdir(wd) if os.path.isfile(os.path.join(wd, x, "abundance.h5"))]
            if sample_name_getter is None:
                sample_name_getter = lambda x: x.split("/")[-2]
            else:
                assert callable(sample_name_getter) is True

            sample_names = [sample_name_getter(x) for x in fp_list]
        elif type(wd) is dict:
            sample_names = list(wd.keys())
            fp_list = [wd[s] for s in sample_names]
        else:
            raise TypeError('Non-supported wd type: %s' % type(wd))
        measure_dict = {}
        if nthreads == 1:
            for sn, fp in tqdm(zip(sample_names, fp_list), total=len(fp_list)):
                _, sample_tm = TranscriptMeasureTable._worker_reader((sn, fp, exon_index))
                measure_dict[sn] = sample_tm
        else:
            with Pool(nthreads) as pool:
                pool_args = [(sn, fp, exon_index) for sn, fp in zip(sample_names, fp_list)]
                pbar = tqdm(total=len(fp_list))
                def pbar_update(*args):
                    pbar.update()
                holders = [pool.apply_async(TranscriptMeasureTable._worker_reader, args=(pool_args[i],), callback=pbar_update) for i in range(pbar.total)]
                res_list = [res.get() for res in holders]
                for res in res_list:
                    sn, sample_tm = res
                    measure_dict[sn] = sample_tm

        measure_df = pd.DataFrame.from_dict(measure_dict, orient="columns")
        measure_df.index = np.array(event_ids, dtype="str")
        if minimum_sanity_checker is not None:
            eids = []
            for eid in measure_df.index:
                if minimum_sanity_checker(measure_df.loc[eid]):
                    eids.append(eid)
            measure_df = measure_df.loc[eids]

        if tmt is None:
            tmt = TranscriptMeasureTable(wd=wd, index_file=index_file, input_type="kallisto", lazy_init=True)
        else:
            if tmt.plaintext is True:
                measure_df = measure_df.applymap(lambda x: x.to_plaintext())
        tmt.data = measure_df
        return tmt


