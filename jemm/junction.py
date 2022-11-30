"""
This module handles reading and writing related tasks for junction-based exon measure
"""

# Author  :   zzjfrank
# Date    :   Aug 23, 2020


import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import pickle
from tqdm import tqdm
from .data_table import DataTable, Measure


class JunctionMeasure(Measure):
    def __init__(self, event_id, sample_id, inc, skp, inc_len, skp_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_id = event_id
        self.sample_id = sample_id
        self.inc = inc
        self.skp = skp
        self.inc_len = inc_len
        self.skp_len = skp_len

    @property
    def psi(self):
        i = self.inc / float(self.inc_len)
        s = self.skp / float(self.skp_len)
        p = i / (i + s) if i + s > 0 else np.nan
        return p

    def __str__(self):
        return "%.3f" % self.psi

    def __eq__(self, other):
        return self.inc == other.inc and self.skp == other.skp and self.inc_len == other.inc_len and self.skp_len == other.skp_len

    def to_plaintext(self):
        s = "%i,%i;%i,%i" % (self.inc, self.skp, self.inc_len, self.skp_len)
        return s

    @staticmethod
    def from_plaintext(s):
        # first assume the presence of variance terms
        try:
            inc_skp, ilen_slen = s.split(";")
            inc, skp = inc_skp.split(",")
            ilen, slen = ilen_slen.split(",")
            return JunctionMeasure(event_id=None, sample_id=None,
                    inc=float(inc), skp=float(skp),
                    inc_len=float(ilen), skp_len=float(slen))
        # otherwise, just load PSI
        except (ValueError, AttributeError):
            psi = float(s)
            return psi
            #return JunctionMeasure(event_id=None, sample_id=None,
            #        inc=psi, skp=0,
            #        inc_len=1, skp_len=1)


class JunctionCountTable(DataTable):
    def __init__(self, filepath=None, input_type=None, event_type='SE', lazy_init=False, plaintext=False):
        """This class provides easy access to JunctionMeasure across large datasets

        Parameters
        ----------
        filepath : str
        input_type : str
        lazy_init : bool

        Attributes
        ----------
        data : pandas.DataFrame
            each row is a splicing event, each column is a sample
        aux : dict
        plaintext : bool

        Examples
        --------
        Initialize a JunctionCountTable from Darts::

            >>> from jemm.junction import JunctionCountTable
            >>> jct = JunctionCountTable(filepath="./data/jemm/darts/input.txt", input_type="darts")
            >>> jct.data.head()
            >>> jct.save("./data/jemm/jct.pkl")

        """
        self._filepath = filepath
        self._input_type = input_type
        self._event_type = event_type
        self.data = None
        self.aux = None
        self.plaintext = plaintext
        if lazy_init is False:
            if self._input_type == 'darts':
                self.from_darts(filepath=self._filepath, sample_name_getter=None, jct=self)
            elif self._input_type == 'rmats':
                self.from_rmats(wd=self._filepath, event_type=self._event_type, jct=self)
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
            self.data.to_hdf(fp, key="data", mode="w" )
        elif mode == "txt":
            self.data.to_csv(fp, sep="\t")

    @staticmethod
    def from_plaintext(filepath, jct=None):
        data = pd.read_table(filepath, index_col=0)
        data = data.applymap(lambda x: JunctionMeasure.from_plaintext(x))
        if jct is None:
            jct = JunctionCountTable(filepath=filepath, input_type="plaintext", lazy_init=True)
        jct.data = data
        return jct

    @staticmethod
    def from_rmats(wd, event_type, jct=None):
        if event_type not in ['SE', 'RI', 'A3SS', 'A5SS', 'MXE']:
            raise ValueError("Unknown event type: %s" % event_type)
        from .rmats_helper import rmatsID_to_dartsID
        annot_fp = os.path.join(wd, "fromGTF.%s.txt"%event_type)
        count_fp = os.path.join(wd, "JC.raw.input.%s.txt"%event_type)
        annot = pd.read_table(annot_fp, index_col=0)
        count = pd.read_table(count_fp, index_col=0)
        #col_to_concat = rmatsID_to_dartsID()[event_type]
        #annot['ID'] = annot.apply(lambda x: ":".join([str(x[c]) for c in col_to_concat ]), axis=1)
        darts_id_converter = rmatsID_to_dartsID()[event_type]
        annot['ID'] = darts_id_converter(annot)
        df = annot[['ID']].join(count, how="inner")
        df.rename(mapper={
            'IJC_SAMPLE_1': 'I1',
            'SJC_SAMPLE_1': 'S1',
            'IJC_SAMPLE_2': 'I2',
            'SJC_SAMPLE_2': 'S2',
            'IncFormLen': 'inc_len',
            'SkipFormLen': 'skp_len'}, axis=1, inplace=True)
        jct = JunctionCountTable.from_darts(file=df, wd=wd, jct=jct)
        jct._input_type = "rmats"
        return jct

    @staticmethod
    def from_darts(file, sample_name_getter=None, jct=None, wd=None, b1=None, b2=None):
        """Constructor method for reading in DARTS junction-based measurements

        Parameters
        ----------
        file : str or pandas.DataFrame
            filepath to DARTS input; if is a pandas.DataFrame, then expect it's already an in-memory data frame with
            columns specified as in DARTS
        sample_name_getter : None or callable
        jct : jemm.junction.JunctionCountTable or None
        wd : str or None
        b1 : list or None
        b2 : list or None

        Returns
        -------
        jemm.junction.JunctionCountTable
        """
        if type(file) is str:
            # Read in the file 
            df = pd.read_table(file)
            wd = os.path.dirname(file)
        else:
            assert type(file) is pd.DataFrame
            assert (wd is not None) or (b1 is not None and b2 is not None)
            df = file
        if b1 is None:
            b1 = open(os.path.join(wd, "b1.txt"), "r").readline().strip().split(",")
        if b2 is None:
            b2 = open(os.path.join(wd, "b2.txt"), "r").readline().strip().split(",")
        if sample_name_getter is None:
            sample_name_getter = lambda x: x.split("/")[-2]
        else:
            assert callable(sample_name_getter) is True
        b1 = tuple([sample_name_getter(x) for x in b1])
        b2 = tuple([sample_name_getter(x) for x in b2])

        # Process raw dataframe to make junction-count data frame
        event_ids = df.ID.tolist()
        count_dict = {}
        inc_lens = np.array(df.inc_len.values, dtype="float16")
        skp_lens = np.array(df.skp_len.values, dtype="float16")
        colmapper = OrderedDict({b1: ["I1", "S1"], b2: ["I2", "S2"]})
        pbar = tqdm(total=len(b1)+len(b2))
        duplicated_cnt = 0
        for k, (sample_names, colnames) in enumerate(colmapper.items()):
            tqdm.write("split counts by samples, for group %i.."%k)
            incs = df[colnames[0]].values
            skps = df[colnames[1]].values
            incs = np.array([[int(c) for c in x.split(",")] for x in incs])
            skps = np.array([[int(c) for c in x.split(",")] for x in skps])
            for i, sn in enumerate(sample_names):
                tmp = [
                    JunctionMeasure(sample_id=None, event_id=None, inc=incs[j, i], skp=skps[j, i],
                                    inc_len=inc_lens[j], skp_len=skp_lens[j])
                    for j in range(len(incs))
                ]
                if sn in count_dict:
                    assert all([tmp[k] == count_dict[sn][k] for k in range(len(tmp))]), "Unmatched %s" % sn
                    duplicated_cnt += 1
                else:
                    count_dict[sn]  = tmp
                pbar.update(n=1)
        pbar.close()
        if duplicated_cnt > 0:
            print('found duplicated n=%i samples but no controversy' % duplicated_cnt)
        count_df = pd.DataFrame.from_dict(count_dict, orient="columns")
        count_df.index = df.ID
        tqdm.pandas()
        tqdm.write("computing cross-sample variance for each event and keep only var>1e-3")
        event_vars = count_df.progress_apply(lambda x: np.nanvar([e.psi for e in x]), axis=1)
        count_df = count_df.loc[event_vars>1e-3]

        # Initialize the JunctionCountTable object
        if jct is None:
            filepath = file if type(file) is str else wd
            jct = JunctionCountTable(filepath=filepath, input_type="darts", lazy_init=True)
        else:
            if jct.plaintext is True:
                count_df = count_df.applymap(lambda x: x.to_plaintext())

        jct.data = count_df
        return jct
