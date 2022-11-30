"""utilities for processing GTEx junction and transcript data
"""

# Author: FZZ
# Date: May 20, 2020

import os
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from jemm.genomic_annotation import ExonSet


def build_jct_index(fp):
    N_SKIP = 2
    out_fp = fp + '.idx'
    line_formatter =  "{id}\t{offset}\n"
    offset = 0
    with open(fp, 'r') as fin, open(out_fp, 'w') as fout:
        for _ in range(N_SKIP):
            offset += len(fin.readline())
        for line in tqdm(fin):
            ele = line.strip().split('\t')
            jct_id = ele[0]
            fout.write( line_formatter.format(id=jct_id, offset=offset) )
            offset += len(line)


def fetch_jct(fp, jct_id):
    index_fp = fp + '.idx'
    assert os.path.isfile(index_fp)
    index = pd.read_table(index_fp, index_col=0, header=None)[1].to_dict()
    if jct_id not in index:
        raise Exception('ID "%s" not in index'% jct_id)
    with open(fp, 'r') as f:
        f.seek(index['Name'], 0)
        header = np.asarray(f.readline().strip().split('\t'))
        f.seek(index[jct_id], 0)
        data = np.asarray(f.readline().strip().split('\t'))
    df = pd.DataFrame(data=data, index=header)
    return df


def get_isoform_junctions(evt, as_type):
    TYPE_TO_ISOFORM = {
        'SE': {
            'long': [
            (lambda x: str(int(x[0])+1), lambda x: x[1]),
            (lambda x: str(int(x[2])+1), lambda x: x[3])
            ],
            'short': [
            (lambda x: str(int(x[0])+1), lambda x: x[3])
            ]},
        'RI': {
            'long': [
                (lambda x: str(int(x[1])-20), lambda x: str(int(x[1])+20) ),
                (lambda x: str(int(x[2])-20), lambda x: str(int(x[2])+20) )
            ],
            'short': [
                (lambda x: x[1], lambda x: x[2])
            ]
        },
        'A3SS': {
            'long': [
                (lambda x: x[0], lambda x: x[1])
            ],
            'short': [
                (lambda x: x[2], lambda x: x[3])
            ]
        },
        'A5SS': {
            'long': [
                (lambda x: x[0], lambda x: x[1])
            ],
            'short': [
                (lambda x: x[2], lambda x: x[3])
            ]
        },
    }
    ele = evt.split(':')
    chrom = ele.pop(0)
    strand = ele.pop(0)
    if as_type!='RI':
        long_ = ['_'.join([f(ele) for f in x]) for x in TYPE_TO_ISOFORM[as_type]['long']]
    else:
        long_ = ['_'.join([str(int(ele[i]) + f) for i in x for f in [-20, 20]]) for x in TYPE_TO_ISOFORM[as_type]['long']]
    short_ = ['_'.join([f(ele) for f in x]) for x in TYPE_TO_ISOFORM[as_type]['short']]
    long = ['%s_%s' % (chrom, x) for x in long_]
    short = ['%s_%s' % (chrom, x) for x in short_]
    return {'strand':strand, 'long': long, 'short': short}


def get_junction_counts(exonset, gtex_fp, dryrun_index=None, dry_run=False):
    """
    Note
    ----------
    dry_run will look for junctions without getting the actual data, so runs fast, and can
    be used to filter invalid junctions without junctions in GTEx
    """
    as_type = exonset._event_type
    jct_df = exonset.data.copy()
    if dryrun_index is not None:
        jct_df = jct_df.loc[dryrun_index]
    jct_df['long_0'] = ''
    jct_df['long_1'] = ''
    jct_df['short_0'] = ''

    samp_index = fetch_jct(fp=gtex_fp, jct_id='Name')
    samp_index = samp_index.drop(index=['Name', 'Description']).index

    gtex_index = pd.read_table(gtex_fp+'.idx', index_col=0, header=None)[1].to_dict()
    if dry_run is True:
        print('Performing dry run')
    for darts_id in tqdm(jct_df.index):
        jct_row = get_isoform_junctions(evt=darts_id, as_type=as_type)
        # add long jct cnts
        for i in range(len(jct_row['long'])):
            jct_id = jct_row['long'][i]
            if jct_id in gtex_index:
                if dry_run is False:
                    jct_cnt = fetch_jct(fp=gtex_fp, jct_id=jct_id)
                    jct_df.at[darts_id, 'long_%i'%i] = ','.join([str(x) for x in jct_cnt.loc[samp_index][0]])
                else:
                    jct_df.at[darts_id, 'long_%i'%i] = '1'
        # add short jct cnts
        for i in range(len(jct_row['short'])):
            jct_id = jct_row['short'][i]
            if jct_id in gtex_index:
                if dry_run is False:
                    jct_cnt = fetch_jct(fp=gtex_fp, jct_id=jct_id)
                    jct_df.at[darts_id, 'short_%i'%i] = ','.join([str(x) for x in jct_cnt.loc[samp_index][0]])
                else:
                    jct_df.at[darts_id, 'short_%i'%i] = '1'

    # store data
    if dry_run is False:
        jct_df.to_csv(os.path.join(os.path.dirname(gtex_fp), 'GTEx-rMATS-%s.tsv.gz' % as_type), sep="\t", index=True)
        pd.Series(samp_index, index=samp_index).to_csv(os.path.join(os.path.dirname(gtex_fp), 'GTEx-rMATS-%s.SAMPLE_INDEX.tsv' % as_type), sep="\t", header=False)
    return jct_df, samp_index 


def make_psi_matrix(gtex_fp, as_type='SE'):
    jct_df = pd.read_csv(os.path.join(os.path.dirname(gtex_fp), 'GTEx-rMATS-%s.tsv.gz' % as_type), sep="\t", index_col=0)
    samp_index = pd.read_csv(os.path.join(os.path.dirname(gtex_fp), 'GTEx-rMATS-%s.SAMPLE_INDEX.tsv' % as_type), sep="\t", header=None, index_col=0)
    psi_df = pd.DataFrame(np.zeros((jct_df.shape[0], samp_index.shape[0])), index=jct_df.index, columns=samp_index.index)
    for eid in tqdm(jct_df.index):
        i0 = np.array([float(x) for x in jct_df.loc[eid, 'long_0'].split(',')])
        i1 = np.array([float(x) for x in jct_df.loc[eid, 'long_1'].split(',')])
        s = np.array([float(x) for x in jct_df.loc[eid, 'short_0'].split(',')])
        i = (i0 + i1) / 2
        psi = pd.Series(i / (i + s), index=samp_index.index)
        psi_df.loc[eid] = psi

    psi_df.to_csv(os.path.join(os.path.dirname(gtex_fp), 'GTEx-rMATS-%s-PSI.tsv' % as_type), sep="\t")



# just run main() for GTEx V8
# GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct
# downloaded from https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz
def main():
    exonset = ExonSet.from_rmats('./data/rmats/fromGTF.SE.txt', 'SE')
    gtex_fp = '/mnt/ceph/users/zzhang/GTEx_splice_junctions/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct'
    jct_df, _ = get_junction_counts(exonset, gtex_fp, dry_run=True)
    dryrun_index = jct_df.query('long_0=="1" and long_1=="1" and short_0=="1"').index
    print('dryrun_index = %i' % len(dryrun_index))
    _ = get_junction_counts(exonset, gtex_fp, dryrun_index=dryrun_index)
