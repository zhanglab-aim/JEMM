import os
import pandas as pd
import numpy as np
from .utils import logit


class Measure:
    """TODO: parent class for Junction and Transcripts.
    Use this to define norms of a Measure"""
    def __init__(self, event_id=None, sample_id=None, est_psi=0.5, var_logit_psi=None, *args, **kwargs):
        self.event_id = event_id
        self.sample_id = sample_id
        self.__psi = est_psi
        self.__logit_psi = logit(self.__psi)
        self.__var_logit_psi = var_logit_psi
        self.__var_psi = None

    @property
    def psi(self):
        return self.__psi

    @property
    def logit_psi(self):
        return self.__logit_psi

    @property
    def var_psi(self):
        return self.__var_psi

    @property
    def var_logit_psi(self):
        return self.__var_logit_psi

    def __str__(self):
        return "%.3f" % self.psi


class DataTable:
    """TODO: parent class for Junction and Transcripts.
    Use this to define norms of DataTables"""
    pass


class DataTableIndexed(DataTable):
    """storing large measure tables in disk, allowing random access by IDs through index building
    """

    def __init__(self, data_file):
        self.data_file = data_file
        self.index_file = self.data_file + ".jem.idx"
        assert os.path.isfile(self.data_file)
        if not os.path.isfile(self.index_file):
            self.build_index(self.data_file)
        else:
            data_time = os.path.getmtime(self.data_file)
            index_time = os.path.getmtime(self.index_file)
            if data_time > index_time:
                print('index is older than data; rebuilding..')
                self.build_index(self.data_file)
        # initial parse
        index = pd.read_table(self.index_file, index_col=0, header=None)
        self.event_index = index.index.tolist()[1:]
        index = index[1].to_dict()
        with open(self.data_file, 'r') as f:
            f.seek(index['sid'], 0)
            header = np.asarray(f.readline().strip().split('\t'))
        self.sample_index = header[1:]

    @staticmethod
    def build_index(data_file):
        out_fp = data_file + '.jem.idx'
        line_formatter = "{id}\t{offset}\n"
        offset = 0
        N_SKIP = 0
        with open(data_file, 'r') as fin, open(out_fp, 'w') as fout:
            for _ in range(N_SKIP):
                offset += len(fin.readline())
            for line in fin:
                ele = line.strip().split('\t')
                jct_id = ele[0]
                fout.write(line_formatter.format(id=jct_id, offset=offset))
                offset += len(line)

    @property
    def index(self):
        return self.event_index

    def get(self, eid):
        """wrap random access like a list and property

        Examples
        ----------
        >>> txr_row = transcript_measure.get(eid)  # returns a pandas.Series for eid row
        """
        index = pd.read_table(self.index_file, index_col=0, header=None)[1].to_dict()
        if eid not in index:
            raise Exception('ID "%s" not in index' % eid)
        with open(self.data_file, 'r') as f:
            f.seek(index['sid'], 0)
            header = np.asarray(f.readline().strip().split('\t'))
            f.seek(index[eid], 0)
            data = np.asarray(f.readline().strip().split('\t'))
        # df = pd.DataFrame(data=data, index=header)
        ds = pd.Series([Measure(event_id=None, sample_id=None, est_psi=x, var_logit_psi=1) for x in
                        data[1:].astype(float)], index=self.sample_index, name=data[0])
        return ds
