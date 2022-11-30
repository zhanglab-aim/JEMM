# -*- coding: utf-8 -*-

"""helper functions and classes for processing and reading rMATS related data
"""

# author: zzjfrank
# date : Sep. 19, 2020

import os

annot_dtypes = {
    'SE': {
        'exonStart_0base': 'int64', 'exonEnd': 'int64', 'upstreamES': 'int64', 'upstreamEE': 'int64',
        'downstreamES': 'int64','downstreamEE': 'int64'
        },
    'A3SS': {
        'longExonStart_0base': 'int64', 'longExonEnd': 'int64', 'shortES': 'int64', 'shortEE': 'int64',
        'flankingES': 'int64', 'flankingEE': 'int64'
        },
    'A5SS': {
        'longExonStart_0base': 'int64', 'longExonEnd': 'int64', 'shortES': 'int64', 'shortEE': 'int64',
        'flankingES': 'int64', 'flankingEE': 'int64'
        },
    'RI': {
        'riExonStart_0base': 'int64', 'riExonEnd': 'int64', 'upstreamES': 'int64', 'upstreamEE': 'int64',
        'downstreamES': 'int64', 'downstreamEE': 'int64'
        }
}


def merge_rmats_runs(run1_dir, run2_dir, minimum_sanity_checker=None, as_types = ['SE', 'A5SS', 'A3SS', 'RI']):
    """Merge two rmats runs, and return a dict of all splicing type data frames in Darts format

    Parameters
    ----------
    run1_dir : str
        directory path to the first run
    run2_dir : str or dict
        directory path (str) or darts dataFrames (dict) to the secon run
    minimum_sanity_checker : callable or None, optional
        when merging, apply the minimum sanity checker to each row and only keep the rows with which 
        a True value is returned
    as_types : list-like

    Returns
    --------
    dict :
        A merged set of data frames in Darts Format
    """
    import pandas as pd
    merged_dict = {}
    for as_type in as_types:
        print('merging %s' % as_type)
        darts_id_converter = rmatsID_to_dartsID()[as_type]
        annot1 = pd.read_table(os.path.join(run1_dir, 'fromGTF.%s.txt' % as_type), index_col=0)
        count1 = pd.read_table(os.path.join(run1_dir, 'JC.raw.input.%s.txt' % as_type), index_col=0)
        annot1['ID'] = darts_id_converter(annot1)
        df1 = annot1[['ID']].join(count1, how='inner')
        df1.index = df1.ID
        df1.rename(mapper={
            'IJC_SAMPLE_1': 'I1',
            'SJC_SAMPLE_1': 'S1',
            'IJC_SAMPLE_2': 'I2',
            'SJC_SAMPLE_2': 'S2',
            'IncFormLen': 'inc_len',
            'SkipFormLen': 'skp_len'}, axis=1, inplace=True)

        if type(run2_dir) is str:
            annot2 = pd.read_table(os.path.join(run2_dir, 'fromGTF.%s.txt' % as_type), index_col=0)
            count2 = pd.read_table(os.path.join(run2_dir, 'JC.raw.input.%s.txt' % as_type), index_col=0)
            annot2['ID'] = darts_id_converter(annot2)
            df2 = annot2[['ID']].join(count2, how='inner')
            df2.index = df2.ID
            df2.rename(mapper={
                'IJC_SAMPLE_1': 'I1',
                'SJC_SAMPLE_1': 'S1',
                'IJC_SAMPLE_2': 'I2',
                'SJC_SAMPLE_2': 'S2',
                'IncFormLen': 'inc_len',
                'SkipFormLen': 'skp_len'}, axis=1, inplace=True)
        elif type(run2_dir) is dict:
            df2 = run2_dir[as_type]
        else:
            raise TypeError('Error parsing run2 with type: %s' % type(run2_dir))
        ids_set = df1.index.intersection(df2.index)
        new_df = []
        for eid in ids_set:
            i1 = df1.loc[eid, 'I1'] + ',' + df2.loc[eid, 'I1']
            s1 = df1.loc[eid, 'S1'] + ',' + df2.loc[eid, 'S1']
            i2 = df1.loc[eid, 'I2'] + ',' + df2.loc[eid, 'I2']
            s2 = df1.loc[eid, 'S2'] + ',' + df2.loc[eid, 'S2']
            assert df1.loc[eid, 'inc_len'] == df2.loc[eid, 'inc_len']
            assert df1.loc[eid, 'skp_len'] == df2.loc[eid, 'skp_len']
            row = pd.Series(
                data=dict(ID=eid, I1=i1, S1=s1, I2=i2, S2=s2, inc_len=df1.loc[eid, 'inc_len'], skp_len=df1.loc[eid, 'skp_len'])
                )
            if minimum_sanity_checker is not None and minimum_sanity_checker(row) is not True:
                continue
            new_df.append(row)
        df = pd.DataFrame(new_df)
        df.index = df['ID']
        merged_dict[as_type] = df

    # finally, parse sample info
    b1_1 = open(os.path.join(run1_dir, 'b1.txt'), 'r').readline().strip().split(',')
    b2_1 = open(os.path.join(run1_dir, 'b2.txt'), 'r').readline().strip().split(',')
    if type(run2_dir) is str:
        b1_2 = open(os.path.join(run2_dir, 'b1.txt'), 'r').readline().strip().split(',')
        b2_2 = open(os.path.join(run2_dir, 'b2.txt'), 'r').readline().strip().split(',')
    elif type(run2_dir) is dict:
        b1_2 = run2_dir['b1']
        b2_2 = run2_dir['b2']
    merged_dict['b1'] = b1_1 + b1_2
    merged_dict['b2'] = b2_1 + b2_2
    return merged_dict


def get_min_sanity_checker(min_avg_count=5, min_avg_psi=0.01, max_avg_psi=0.99):
    """Given criteria as input, returns a callable function that checks each row/record's PSI sanity
    """
    import numpy as np
    def checker(row):
        i1 = np.array([int(x) for x in row['I1'].split(',')])
        s1 = np.array([int(x) for x in row['S1'].split(',')])
        i2 = np.array([int(x) for x in row['I2'].split(',')])
        s2 = np.array([int(x) for x in row['S2'].split(',')])
        if np.mean(i1+s1) < min_avg_count or np.mean(i2+s2) < min_avg_count:
            return False
        psi1 = i1/row['inc_len'] / (i1/row['inc_len'] + s1/row['skp_len'] )
        psi2 = i2/row['inc_len'] / (i2/row['inc_len'] + s2/row['skp_len'] )
        psi = np.concatenate([psi1, psi2], axis=0)
        avg = np.nanmean(psi)
        if avg < min_avg_psi:
            return False
        if avg > max_avg_psi:
            return False
        return True
    return checker


def _se(df):
    col_to_concat = ["chr", "strand", "upstreamEE", "exonStart_0base", "exonEnd", "downstreamES"]
    IDs = df.apply(lambda x: ":".join([str(x[c]) for c in col_to_concat ]), axis=1)
    return IDs


def _a3(df):
    IDs = []
    col_to_concat = {
            '+': ["chr", "strand", "flankingEE", "longExonStart_0base", "flankingEE", "shortES"],
            '-': ["chr", "strand", "longExonEnd", "flankingES", "shortEE", "flankingES"]
    }
    for i in range(df.shape[0]):
        row = df.iloc[i]
        strand = row['strand']
        id = ":".join([str(row[c]) for c in col_to_concat[strand] ])
        IDs.append(id)
    return IDs


def _a5(df):
    IDs = []
    col_to_concat = {
            '+': ["chr", "strand", "longExonEnd", "flankingES",  "shortEE", "flankingES"],
            '-': ["chr", "strand", "flankingEE", "longExonStart_0base", "flankingEE", "shortES"]
    }
    for i in range(df.shape[0]):
        row = df.iloc[i]
        strand = row['strand']
        id = ":".join([str(row[c]) for c in col_to_concat[strand] ])
        IDs.append(id)
    return IDs


def _ri(df):
    col_to_concat = ["chr", "strand", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"]
    IDs = df.apply(lambda x: ":".join([str(x[c]) for c in col_to_concat ]), axis=1)
    return IDs


def rmatsID_to_dartsID():
    """Convert rMATS columsn to DARTS condensed ID format

    Reference
    ---------
    https://github.com/zj-zhang/Darts_BHT/blob/master/Darts_BHT/convert_rmats.py
    """
    rmats_type = {
            "SE": _se,
            "A3SS": _a3,
            "A5SS": _a5,
            "RI": _ri,
            #'MXE': ["chr", "strand", "1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"]
    }
    return rmats_type


