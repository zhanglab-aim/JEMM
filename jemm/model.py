"""
This module performs the statistical testing for observed data
"""

# Author  : zzjfrank
# Date    : Aug 25, 2020
# Update  : Feb 18, 2021 - Add Linear Mixed Model

import os
import numpy as np
import pandas as pd
import warnings
import scipy.optimize
import scipy.stats
from tqdm import tqdm
from collections import OrderedDict
from multiprocessing import Pool
import pickle
import warnings
import copy
from .utils import sigmoid, fdr_bh, logit
from .utils import chunkify
import joblib
import multiprocessing
import tempfile
import logging
from .junction import JunctionCountTable
from .transcript import TranscriptMeasureTable
from .data_table import DataTableIndexed

LOGGER = logging.getLogger("jemm")


def get(type_name):
    assert type(type_name) is str, TypeError('type_name must be string')
    if type_name.lower() == 'abstract':
        return JointExonModel
    elif type_name.lower() == 'lm':
        return JemmLinearRegression
    elif type_name.lower() == 'lmm':
        return JemmLinearMixedModel
    else:
        raise ValueError('unknown identifier: %s' % type_name)


def deserialize_data_table(dt):
    # TODO: update event index to be attributes in jemm.junction.JunctionCountTable and
    #  jemm.transcript.TranscriptMeasureTable
    if dt is None:
        event_index = []
        sample_index = None
        df = None
    elif type(dt) is JunctionCountTable or type(dt) is TranscriptMeasureTable:
        event_index = dt.event_index if hasattr(dt, "event_index") else dt.data.index.tolist()
        sample_index = dt.sample_index if hasattr(dt, "sample_index") else dt.data.columns
        df = dt.data
    elif type(dt) is DataTableIndexed:
        event_index = dt.event_index
        sample_index = dt.sample_index
        df = dt
    elif type(dt) is pd.DataFrame:
        event_index = dt.index.tolist()
        sample_index = dt.columns
        df = dt
    else:
        raise TypeError("can't deserialize dataTable of type %s" % type(dt))
    return df, event_index, sample_index


def match_covariate_sample_index(df, sample_index):
    if df is None:
        return None
    elif type(df) is pd.DataFrame:
        return df[sample_index]
    elif type(df) is DataTableIndexed:
        assert list(df.sample_index) == list(sample_index)
        return df


class JointExonModel:
    def __init__(self, junction_measure=None, transcript_measure=None, covariates=None, diff_intercept_by_measure=True,
                 min_valid_samps=0.3,
                 skip_load=False,
                 make_memmap=False,
                 *args,
                 **kwargs):
        """The wrapper class for performing Jemm regression analysis

        Parameters
        ----------
        junction_measure : jemm.junction.JunctionCountTable
        transcript_measure : jemm.transcript.TranscriptMeasureTable
        covariates : jemm.covariate.Covariate
        diff_intercept_by_measure : bool
        min_valid_samps : float
            The minimum proportion of valid samples in order for an exon to be considered
        skip_load : bool
            whether skip initial loading and check

        Attributes
        ----------
        wald_tests : collections.OrderedDict
        stats_sheet : dict
            For each covariate, provide a stand-alone pandas.DataFrame for its estimates and p-values

        Notes
        ------
        An example id is "chr10:45443828-45444116:45444286-45445508:+", and corresponding darts id is
        "chr10:+:45443828:45444115:45444286:45445507"
        "chr11:-:85983973:85990249:85990399:85996825"
        "chr6:+:36470257:36474788:36475062:36479120"

        Examples
        --------
        An example using pre-built junction measure and transcript measre::

            >>> import pickle
            >>> from jemm.model import JointExonModel
            >>> import pandas as pd
            >>> txr = pickle.load(open("./data/jemm/txr.pkl", "rb"))
            >>> jct = pickle.load(open("./data/jemm/jct.pkl", "rb"))
            >>> covariates = pd.DataFrame({"intercept": [1]*9, "condition": [1]*5 + [0]*4},
            >>>     index=['27-0', '28-0', '29-0', '39-0', '41-0', '2169', '2209', '2211', '2213'])
            >>> jmm = JointExonModel(junction_measure=jct, transcript_measure=txr, covariates=covariates)
            >>> jmm.run_tests()
            >>> jmm.save("./data/jemm_pbmc.pkl")

        A debugging example; move this to test later::

            >>> covariates = pd.DataFrame({"intercept": [1]*9, "condition": [1]*5 + [0]*4})
            >>> covariates.index = ['27-0', '28-0', '29-0', '39-0', '41-0', '2169', '2209', '2211', '2213']
            >>> data = {'y.junc': np.array([50, 60, 70, 80, 90, 10, 20, 30, 40]), 'N': np.array([100]*9), 'x': covariates.to_numpy()}

        """
        # Not really necessary ?..
        # if self.junction_measure.data.shape[1] != self.transcript_measure.data.shape[1]:
        #    raise ValueError("junction measure and transcript measure have different set of samples")

        self.diff_intercept_by_measure = diff_intercept_by_measure
        self.covariates = covariates
        if skip_load is False:
            self.min_valid_samps = min_valid_samps
            # event index is the union of jct + txr
            jct_df, jct_event_index, jct_sample_index = deserialize_data_table(junction_measure)
            txr_df, txr_event_index, txr_sample_index = deserialize_data_table(transcript_measure)
            self.event_index = list(
                set(jct_event_index + txr_event_index))
            # covariate match sample ids in columns
            samps = []
            if jct_sample_index is not None:
                samps.append(jct_sample_index)
            if txr_sample_index is not None:
                samps.append(txr_sample_index)
            self.covariates.match_index(*samps)
            self.sample_index = self.covariates.index
            # Re-order the columns to have identical samples w.r.t. covariates df
            self.junction_measure = match_covariate_sample_index(jct_df, self.sample_index)
            self.transcript_measure = match_covariate_sample_index(txr_df, self.sample_index)
            if make_memmap:
                LOGGER.info('making mem map')
                temp_folder = tempfile.mkdtemp()
                if self.junction_measure is not None:
                    # jct
                    jct_mmap = os.path.join(temp_folder, 'jemm.jct.mmap')
                    if os.path.exists(jct_mmap): os.unlink(jct_mmap)
                    _ = joblib.dump(self.junction_measure, jct_mmap)
                    self.junction_measure = joblib.load(jct_mmap, mmap_mode='r')
                if self.transcript_measure is not None:
                    # txr
                    txr_mmap = os.path.join(temp_folder, 'jemm.txr.mmap')
                    if os.path.exists(txr_mmap): os.unlink(txr_mmap)
                    _ = joblib.dump(self.transcript_measure, txr_mmap)
                    self.transcript_measure = joblib.load(txr_mmap, mmap_mode='r')
                LOGGER.info('done mem map')
            self._covariates = self.covariates.to_numpy()
            _ = self._check_covariate_condition()

        # store of test results
        self.__viable_tests = ['LRT', 'Wald']
        self.stats_tests = OrderedDict({})
        self.stats_sheet = {}

        # private attributes
        # define a threshold for under/overflowing
        self._MIN_ABS_COEF = 1e-8
        self._MAX_ABS_COEF = 1e16

    def _get_event_junction_data(self, eid, covariates=None, sanitize_data=True):
        if eid in self.junction_measure.index:
            # TODO: can we make this unified as a get method? fzz, 2021.10.20
            if type(self.junction_measure) is pd.DataFrame:
                jct_row = self.junction_measure.loc[eid]
            else:
                jct_row = self.junction_measure.get(eid)
            # some weird  overflow here.. what pandas convert int to?
            y_junc = np.array([x.inc * (x.skp_len / x.inc_len) for x in jct_row])
            N = np.array([x.skp for x in jct_row]) + y_junc
            data = {'y.junc': y_junc, 'N.junc': N}
            if covariates is None:
                data.update({'x.junc': self._covariates})
            else:
                data.update({'x.junc': self.covariates.get_design_mat(covariates=covariates)})
            if sanitize_data is True:
                return self.check_data_sanity(data, min_valid_samps=self.min_valid_samps)
            else:
                return data
        else:
            return None

    def _get_event_transcript_data(self, eid, covariates=None, sanitize_data=True):
        if eid in self.transcript_measure.index:
            if type(self.transcript_measure) is pd.DataFrame:
                txr_row = self.transcript_measure.loc[eid]
            else:
                txr_row = self.transcript_measure.get(eid)
            if len(txr_row.shape) > 1:
                # TODO: Need better handling for overlapped IDs
                txr_row = txr_row.iloc[0]
            y_tpm = np.array([x.logit_psi for x in txr_row])
            tau2 = np.array([x.var_logit_psi for x in txr_row])
            tau2 = np.maximum(tau2, 0.1)  # this is important to avoid erronously large weights
            data = {'y.tpm': y_tpm, 'tau2.tpm': tau2}
            if covariates is None:
                data.update({'x.tpm': self._covariates})
            else:
                data.update({'x.tpm': self.covariates.get_design_mat(covariates=covariates)})
            if sanitize_data is True:
                return self.check_data_sanity(data, min_valid_samps=self.min_valid_samps)
            else:
                return data
        else:
            return None

    def _check_covariate_condition(self):
        cond_num = np.linalg.cond(self._covariates)
        if cond_num > 100:
            warnings.warn(
                "Input covariate matrix has condition number = %s > 100, indicating the presence of co-linearality; "
                "optimization might be unstable." % cond_num,
                stacklevel=2)
        return cond_num

    @staticmethod
    def check_data_sanity(data, min_valid_samps=0.5):
        """Check if data passes minimum sanity

        Parameters
        ----------
        data : dict

        Returns
        -------
        data or None
            if pass sanity check, return the data; otherwise returns None
        """
        if 'y.tpm' in data:
            valid_idx = np.where((~np.isnan(data['y.tpm'])) & (~np.isnan(data['tau2.tpm'])))[0]
            # if less than specified valid samples..
            if len(valid_idx) <= min_valid_samps * len(data['y.tpm']):
                return None
            data = {k: v[valid_idx] for k, v in data.items()}
            data.update({'sample_index.tpm': valid_idx})
            # if np.mean(np.isnan(data['y.tpm'])) > 0.5:
            #    return None
            # if np.mean(np.isnan(data['tau2.tpm'])) > 0.5:
            #    return None
            # if np.nanvar((data['y.tpm'])) < 1e-3:
            #    return None
        if 'y.junc' in data:
            valid_idx = np.where(data['N.junc'] >= 10)[0]
            # if less than specified valid samples..
            if len(valid_idx) <= min_valid_samps * len(data['y.junc']):
                return None
            data = {k: v[valid_idx] for k, v in data.items()}
            data.update({'sample_index.junc': valid_idx})
            # don't look at non-expressed exons
            # if np.nanmean(data['N.junc']) < 10:
            #    return None
            # psi = data['y.junc'] / (data['N.junc'])
            # if np.nanvar(psi) < 1e-3:
            #    return None
            # if np.mean(psi == 0) > 0.6 or np.mean(psi == 1) > 0.6:
            #    return None
        return data

    def get_data(self, eid, covariates=None, use_jct=True, use_txr=True):
        """

        Parameters
        ----------
        eid : str
            exon id to fetch data
        covariates : None, or list
            if None, get default design matrix; otherwise, get the design matrix for specified covariates
        use_jct : bool
            whether to fetch junction data
        use_txr : bool
            whether to fetch transcript data

        Returns
        -------

        """
        assert use_jct or use_txr, "Cannot have both use_jct=False and use_txr=False simultaniously"
        data_jct = self._get_event_junction_data(eid=eid, covariates=covariates) if use_jct is True else None
        data_txr = self._get_event_transcript_data(eid=eid, covariates=covariates) if use_txr is True else None
        data = {}
        use_diff_intercept = self.diff_intercept_by_measure and data_txr is not None and data_jct is not None
        if data_jct is not None:
            data.update({
                'y.junc': data_jct['y.junc'],
                'N.junc': data_jct['N.junc'],
                'sample_index.junc': data_jct['sample_index.junc'],
                'x.junc': np.append(data_jct['x.junc'], np.ones((data_jct['x.junc'].shape[0], 1)),
                                    axis=1) if use_diff_intercept else \
                    data_jct['x.junc'],
            })
        if data_txr is not None:
            data.update({
                'y.tpm': data_txr['y.tpm'],
                'tau2.tpm': data_txr['tau2.tpm'],
                'sample_index.tpm': data_txr['sample_index.tpm'],
                'x.tpm': np.append(data_txr['x.tpm'], np.zeros((data_txr['x.tpm'].shape[0], 1)),
                                   axis=1) if use_diff_intercept else \
                    data_txr['x.tpm']
            })
        return data, use_diff_intercept

    def optimize(self, data, test_index=None, *args, **kwargs):
        """Not implemented for the data parsing scaffold
        """
        raise NotImplementedError('Not Implemented for Data Parsing abstract class')
        wald_test = pd.DataFrame({})
        opt_val = np.nan
        return wald_test, opt_val

    def _filter_machine_precision(self):
        for eid in self.stats_tests:
            reg_table = self.stats_tests[eid]
            reg_table.at[reg_table['coef'].abs() < self._MIN_ABS_COEF, 'pvals'] = 1.
            reg_table.at[reg_table['coef'].abs() > self._MAX_ABS_COEF, 'pvals'] = 1.

    def multiple_testing_correction(self, method="fdr"):
        assert len(self.stats_tests) > 0, "Have not run tests yet"
        self._filter_machine_precision()
        tested_eids = [eid for eid in self.stats_tests]
        for cov_name in self.covariates.columns.tolist() + ["_junc_diff_"]:
            pvals = [self.stats_tests[eid].loc[cov_name, "pvals"] if cov_name in self.stats_tests[eid].index else np.nan
                     for
                     eid in tested_eids]
            coefs = [self.stats_tests[eid].loc[cov_name, "coef"] if cov_name in self.stats_tests[eid].index else np.nan
                     for
                     eid in tested_eids]
            if method == "fdr":
                qvals = fdr_bh(pvals)
            elif method == "bonferroni":
                pvals = np.array(pvals).flatten()
                qvals = np.clip(pvals * len(pvals), a_min=0, a_max=1)
            else:
                raise ValueError("Cannot understand/unimplemented correction method: %s" % method)
            for i, eid in enumerate(self.stats_tests):
                self.stats_tests[eid].at[cov_name, "qvals"] = qvals[i]
            self.stats_sheet[cov_name] = pd.DataFrame({"coefs": coefs, "pvals": pvals, "qvals": qvals},
                                                      index=tested_eids)
        return self.stats_tests

    def likelihood_ratio_test(self, wald, opt, data):
        test_index = wald.index
        chi2 = []
        for i in range(len(test_index)):
            data0 = copy.copy(data)
            xs = ['x.tpm', 'x.junc']
            for x in xs:
                if x in data0:
                    data0[x] = data0[x][:, [j for j in range(len(test_index)) if j != i]]
            _, alt = self.optimize(data=data0, test_index=test_index.drop(test_index[i]))
            chi2.append(2 * (alt - opt))
        chi2 = np.array(chi2)
        pvals = 1 - scipy.stats.chi2.cdf(chi2, df=1)
        wald['chi2.lrt'] = chi2
        wald['pvals'] = pvals
        del wald['chi2.wald']
        return wald

    def _test_worker(self, eid, test_type, use_jct, use_txr, force_diff_intercept):
        data, use_diff_intercept = self.get_data(eid, use_jct=use_jct, use_txr=use_txr)
        if force_diff_intercept is True and use_diff_intercept is False:
            return None
        if len(data) > 0:
            # self.stats_tests[eid] = self.optimize(data=data)
            try:
                wald, opt = self.optimize(data=data)
                if test_type == "Wald":
                    res = wald
                elif test_type == "LRT":
                    lrt_res = self.likelihood_ratio_test(wald=wald, opt=opt, data=data)
                    res = lrt_res
                else:
                    raise ValueError("Unsupported test type: %s" % test_type)
            except Exception as e:
                raise Exception("Error for %s.\n Traceback: %s" % (eid, e))
            if res is None:
                return None
            else:
                return eid, res
        else:
            return None

    def _test_batch_worker(self, eids, test_type, use_jct, use_txr, force_diff_intercept, force_rerun=False):
        res_list = []
        i = 0
        tot = len(eids)
        pid = os.getpid()
        for eid in eids:
            i += 1
            if eid in self.stats_tests and force_rerun is False:
                continue
            if not i % 5:
                LOGGER.debug('pid %s : %i / %i (%.2f%%)' % (pid, i, tot, i / tot * 100))
            res_list.append(self._test_worker(
                eid,
                test_type, use_jct, use_txr, force_diff_intercept
            ))
        return res_list

    def run_tests(self, test_type, data="both", pval_adjust_method="bonferroni", force_rerun=False,
                  force_diff_intercept=False, nthreads=1):
        """run LRT/Wald test for each event

        Parameters
        ----------
        test_type : str
        data : str
        pval_adjust_method : str
        force_rerun : bool
        force_diff_intercept : bool
            If true, will only use events that have data of **both** Junction and Transcript.
        nthreads : int
            number of threads for multiprocessing

        Returns
        -------
        self.stats_tests : dict
            A dictionary mapping exon index to statistical test results
        """
        assert nthreads >= 1
        if test_type not in self.__viable_tests:
            raise ValueError("test_type must be in %s" % self.__viable_tests)
        # if len(self.stats_tests) > 0 and force_rerun is False:
        #    print("Found previous results; you can re-run the tests by setting `force_rerun=True`")
        #    return self.stats_tests
        self.stats_tests = OrderedDict({})
        if data == "both":
            use_jct = True
            use_txr = True
        elif data == "jct":
            use_jct = True
            use_txr = False
        elif data == "txr":
            use_jct = False
            use_txr = True
        else:
            raise ValueError("Unknown data identifier: %s" % data)

        if force_diff_intercept is True:
            assert data == "both", "Cannot have force_diff_intercept=True but data!=both"

        if nthreads == 1:
            for eid in tqdm(self.event_index):
                if eid in self.stats_tests and force_rerun is False:
                    continue
                try:
                    res = self._test_worker(eid, test_type, use_jct, use_txr, force_diff_intercept)
                    if res is not None:
                        self.stats_tests[eid] = res[1]
                except KeyboardInterrupt:
                    LOGGER.info("User stopped testing")
                    break
                except Exception as e:
                    LOGGER.info(f"Error for eid {eid}. Detail:\n {e}")
                    break
        else:
            # pbar = tqdm(total=len(self.event_index))

            # def pbar_update(*args):
            #    pbar.update()
            LOGGER.info("multiprocessing with n=%i" % nthreads)
            res_list = joblib.Parallel(n_jobs=nthreads, prefer='processes')(joblib.delayed(self._test_worker)(
                self.event_index[i], test_type, use_jct, use_txr, force_diff_intercept)
                                                        for i in tqdm(range(len(self.event_index)))
                                                        if not (self.event_index[i] in self.stats_tests and
                                                                force_rerun is False))
            #eid_batches = [x for x in chunkify(self.event_index, nthreads)]
            #res_batches = joblib.Parallel(n_jobs=nthreads, prefer='processes')(joblib.delayed(self._test_batch_worker)(
            #    eid_batches[i], test_type, use_jct, use_txr, force_diff_intercept)
            #                                                                for i in range(nthreads)
            #                                                                )
            #for res_list in res_batches:
            #    for res in res_list:
            #        if res is not None:
            #            self.stats_tests[res[0]] = res[1]

            #with Pool(nthreads) as pool:
            # doing single event iteratively is too slow. fzz, 10.20.2021
            # holders = [pool.apply_async(self._test_worker, args=(
            #    self.event_index[i], test_type, use_jct, use_txr, force_diff_intercept), callback=pbar_update)
            #           for i in range(pbar.total)
            #           if not (self.event_index[i] in self.stats_tests and force_rerun is False)
            #           ]
            # res_list = [res.get() for res in holders]
            for res in res_list:
                if res is not None:
                    self.stats_tests[res[0]] = res[1]

             #   eid_batches = [x for x in chunkify(self.event_index, nthreads)]

            #    res_batches = pool.map(self._test_batch_worker,
            #                           [(eid_batches[i], test_type, use_jct, use_txr, force_diff_intercept)
            #                            for i in range(nthreads)])
            #for res_list in res_batches:
            #    for res in res_list:
            #        if res is not None:
            #            self.stats_tests[res[0]] = res[1]
        _ = self.multiple_testing_correction(method=pval_adjust_method)
        return self.stats_tests

    def predict(self, eid, data="both"):
        if data == "both":
            use_jct = True
            use_txr = True
        elif data == "jct":
            use_jct = True
            use_txr = False
        elif data == "txr":
            use_jct = False
            use_txr = True
        else:
            raise ValueError("Unknown data identifier: %s" % data)
        assert eid in self.stats_tests
        data, use_diff_intercept = self.get_data(eid, use_jct=use_jct, use_txr=use_txr)
        beta = self.stats_tests[eid]['coef']
        # if (self.diff_intercept_by_measure is False or data == "txr") and "_junc_diff_" in beta:
        if use_diff_intercept is False and "_junc_diff_" in beta:
            beta = beta.drop("_junc_diff_", axis=0, inplace=False)
        beta = beta.to_numpy()
        fit = {}
        if 'y.junc' in data:
            z = np.matmul(data['x.junc'], beta)
            fit_junc = sigmoid(z)
            obs_junc = data['y.junc'] / data['N.junc']
            # fit['fit.junc'] = fit_junc
            # fit['obs.junc'] = obs_junc
            fit['junc'] = self.covariates.iloc[data['sample_index.junc']]
            fit['junc'] = fit['junc'].assign(obs=obs_junc, fit=fit_junc)
        if 'y.tpm' in data:
            z = np.matmul(data['x.tpm'], beta)
            fit_tpm = sigmoid(z)
            obs_tpm = sigmoid(data['y.tpm'])
            # fit['fit.tpm'] = fit_tpm
            # fit['obs.tpm'] = obs_tpm
            fit['tpm'] = self.covariates.iloc[data['sample_index.tpm']]
            fit['tpm'] = fit['tpm'].assign(obs=obs_tpm, fit=fit_tpm)
        return fit

    def get_stringent_candidates(self, contrast_col, qval_thresh=0.05, min_change=0.05, is_consistent_measure=False):
        candidates = []
        for eid in self.stats_tests:
            # pval signif
            if self.stats_tests[eid].loc[contrast_col, 'qvals'] > qval_thresh:
                continue
            # consistent measure not rejected
            if is_consistent_measure and self.stats_tests[eid].loc["_junc_diff_", "qvals"] < qval_thresh:
                continue
            fit = self.predict(eid)
            is_pass = True
            for k in fit:
                # fitted values has enough change
                # if len(fit[k][contrast_col].unique()) <= 2:
                #    fit_mean = fit[k].groupby(contrast_col)['fit'].mean()
                # else:
                #    raise NotImplementedError("not implemented for continuous variable")
                # diffs = abs(fit_mean[1] - fit_mean[0])
                baseline = sigmoid(self.stats_tests[eid].loc["intercept", "coef"])
                base_plus = sigmoid(
                    self.stats_tests[eid].loc["intercept", "coef"] + self.stats_tests[eid].loc[contrast_col, "coef"])
                diffs = abs(base_plus - baseline)
                if diffs < min_change:
                    is_pass = False
                # goodness-of-fit measure by pearson
                try:
                    model_sp = scipy.stats.pearsonr(fit[k]['fit'], fit[k]['obs'])
                    if model_sp[0] < 0:
                        is_pass = False
                except ValueError:
                    is_pass = False
            if is_pass:
                candidates.append(eid)
        return candidates

    def regress_out_cofactors(self, eids, cofactors, measure_type='jct', sample_index=None):
        """Regress out a given set of co-factors from a fitted regression model

        Parameters
        ----------
        eids : list
        cofactors : list
        measure_type : str
        sample_index : list or pandas.Index

        Returns
        --------
        psi : pandas.DataFrame
            Logit PSI values after regress out the given cofactors
        """
        if measure_type == 'jct':
            psi_ = self.junction_measure.loc[eids].applymap(lambda x: logit(x.psi))
        elif measure_type == 'txr':
            psi_ = self.transcript_measure.loc[eids].applymap(lambda x: logit(x.psi))
        else:
            raise ValueError("Cannot understand measure type %s; must be jct or txr" % measure_type)

        for i in range(len(eids)):
            eid = eids[i]
            beta_all = self.stats_tests[eid]['coef']
            beta = beta_all[cofactors].to_numpy()
            cofactor_index = [i for i in range(beta_all.shape[0]) if beta_all.index[i] in cofactors]
            data, use_diff_intercept = self.get_data(eid)
            if measure_type == 'jct':
                fitted_ = data['x.junc'][:, cofactor_index].dot(beta)
                psi_.iloc[i, data['sample_index.junc']] -= fitted_
            elif measure_type == 'txr':
                fitted_ = data['x.tpm'][:, cofactor_index].dot(beta)
                psi_.iloc[i, data['sample_index.tpm']] -= fitted_

        if sample_index is not None:
            psi_ = psi_[sample_index]
        return psi_

    def save_pickle(self, fp):
        """Serialize an instance of Jemm.model to a pickle
        """
        pickle.dump(self.stats_tests, open(fp, "wb"))

    def load_pickle(self, fp):
        self.stats_tests = pickle.load(open(fp, "rb"))

    def save_rmats_fmt(self, annot_fp, out_fp, event_type, contrast_col):
        """Save a copy of the inference results in rMATS format

        Parameters
        ----------
        annot_fp : str
        out_fp : str
        event_type : str
        """
        from .genomic_annotation import ExonSet
        exonset = ExonSet.from_rmats(gtf_annot_fp=annot_fp, event_type=event_type)
        pvals = []
        fdr = []
        inc_diff = []
        psi1 = []
        psi2 = []
        for eid in self.stats_sheet[contrast_col].index:
            rmats_annot_row = exonset.data.reindex([eid])
            pvals.append(self.stats_sheet[contrast_col].loc[eid, "pvals"])
            fdr.append(self.stats_sheet[contrast_col].loc[eid, "qvals"])
            # diff = self.stats_tests[eid].loc[contrast_col, "coef"]
            diff = self.stats_sheet[contrast_col].loc[eid, "coefs"]
            base = self.stats_tests[eid].loc["intercept", "coef"]
            inc_diff.append(sigmoid(base + diff) - sigmoid(base))
            psi1.append(sigmoid(base + diff))
            psi2.append(sigmoid(base))
        stats_df = pd.DataFrame(zip(*[pvals, fdr, psi1, psi2, inc_diff]),
                                columns=['PValue', 'FDR', 'IncLevel1', 'IncLevel2', 'IncLevelDifference'],
                                index=self.stats_sheet[contrast_col].index)
        # annot_df.drop(columns=['darts_id'], inplace=True)
        annot_df = exonset.data.reindex(self.stats_sheet[contrast_col].index)
        annot_df['IJC_SAMPLE_1'] = 100
        annot_df['IJC_SAMPLE_2'] = 100
        annot_df['SJC_SAMPLE_1'] = 100
        annot_df['SJC_SAMPLE_2'] = 100
        annot_df['IncFormLen'] = 1
        annot_df['SkipFormLen'] = 1

        outdf = pd.concat([annot_df, stats_df], axis=1)
        outdf.to_csv(out_fp, sep="\t", index=False)
        return outdf

    def munge_covariates(self, covs, meta_name="combined"):
        """Merge a set of covariates to create a `meta` covariate that encompasses multiple conditions

        This function is mainly designed to get a summary-level significance for a group of biologically related covariates, such
        as the time-course conditions with different samples. In particular, the way to combine different covariates is:
            - for coefficients, take the average
            - for p-values, use fisher's method

        Parameters
        ----------
        covs : list of str
        meta_name : str
        """
        from .utils import fisher_method_combine_pvals
        tested_eids = [e for e in self.stats_tests]
        coefs = []
        pvals = []
        for eid in tested_eids:
            coefs.append(self.stats_tests[eid].loc[covs, 'coef'].mean())
            raw_pvals = self.stats_tests[eid].loc[covs, 'pvals'].to_numpy()
            pvals.append(fisher_method_combine_pvals(raw_pvals))
        qvals = fdr_bh(pvals)
        self.stats_sheet[meta_name] = pd.DataFrame({'coefs': coefs, 'pvals': pvals, 'qvals': qvals}, index=tested_eids)
        return self.stats_sheet[meta_name]

    def save_regression_table(self, outfp, exonset, annotations=None, eids=None, order_by_covariate=None,
                              order_by='logP'):
        """Save the regression/inference results by outputting the model estimates and p-values

        Parameters
        ----------
        outfp : str or None
            output filepath; if None, then do not save and only return the regression data frame
        exonset : jemm.genomic_annotation.ExonSet
            the annotation with the index being the exon ids; this is a must because we need to know which gene the exon belongs (at the
            minimum)
        annotations : list
            a list of strings for extracting exon annotations
        eids : list
        order_by_covariate : str or None
            the covariate name for ordering the resutls; if None, do not order
        order_by : str
            must be in ['logP', 'coef']

        Notes
        -----
        The log p-values for `order_by_covariate` is already adjusted for multiple-testing correction.
        """
        eids = eids or [e for e in self.stats_tests]
        if order_by_covariate is not None:
            assert order_by in ['logP', 'coef'], ValueError(
                "Unsupported order_by arugment '%s'; must be in ('logP', 'coef')" % order_by)
            assert order_by_covariate in self.stats_sheet
        target_gene = exonset.data.reindex(eids)  # [['GeneID', 'geneSymbol']]
        if 'ID' in target_gene:
            target_gene.drop(['ID'], axis=1, inplace=True)
        covariate_names = [",".join(self.stats_tests[eid].index) for eid in eids]
        covariate_est = [",".join(["%.5f" % x if abs(x) > 1e-5 else '%.2e' % x for x in self.stats_tests[eid]['coef']])
                         for eid in eids]
        covariate_var = [",".join(["%.5f" % x if x > 1e-5 else '%.2e' % x for x in self.stats_tests[eid]['var']]) for
                         eid in eids]
        covariate_qval = [",".join(["%.2f" % x if x > 1e-2 else '%.2e' % x for x in self.stats_tests[eid]['qvals']]) for
                          eid in eids]
        cov_df = pd.DataFrame(data=zip(*[covariate_names, covariate_est, covariate_var, covariate_qval]),
                              columns=['covariate_names', 'covariate_est', 'covariate_var', 'covariate_qval'],
                              index=eids)
        subdfs_to_concat = [target_gene]
        if annotations is not None:
            exon_annot = exonset.get_splice_site_annotations(eids=eids, annotations=annotations)
            subdfs_to_concat.append(exon_annot)

        # always put covariate df in the last, because it's pretty messy...
        subdfs_to_concat.append(cov_df)
        df = pd.concat(subdfs_to_concat, axis=1)
        if order_by_covariate is not None:
            df["%s|coef" % order_by_covariate] = [self.stats_sheet[order_by_covariate].loc[e, "coefs"] for e in eids]
            df["%s|neg.log10(Padj)" % order_by_covariate] = [
                -np.log10(self.stats_sheet[order_by_covariate].loc[e, "qvals"] + 1e-16) for e in eids]
            if order_by == 'logP':
                df.sort_values(by="%s|neg.log10(Padj)" % order_by_covariate, ascending=False, inplace=True)
            elif order_by == 'coef':
                df.sort_values(by="%s|coef" % order_by_covariate, ascending=True, inplace=True)
        if outfp is not None:
            df.to_csv(outfp, sep="\t", index=True)
        return df

    def load_regression_table(self, outfp, fast_load=False):
        """Load a pre-computed regression results into the current instance

        For each covariate, the coefficients and variance estimates for coefficients are reloaded to the precision of 1e-5. Then, the 
        chi2 statistics, p-values and q-values will be re-computed.

        Parameters
        ----------
        outfp : str
        fast_load : bool
            If true, skip re-computation of p-values and stats test, and use these values from the file as-is; may lose some precision.
            Default is False.

        Returns
        -------
        jemm.model.Jemm
            An updated instance with regression results

        Raises
        ------
        AssertionError: if reloaded covariates names does not match the covariates in current instance
        """
        outdf = pd.read_table(outfp)
        for i in range(outdf.shape[0]):
            row = outdf.iloc[i]
            eid = row['darts_id']
            cov_names = [str(x) for x in row['covariate_names'].split(',')]
            beta_mle = np.array([float(x) for x in row['covariate_est'].split(',')])
            mle_var = np.array([float(x) for x in row['covariate_var'].split(',')])
            if fast_load:
                qvals = np.array([float(x) for x in row['covariate_qval'].split(',')])
                wald_test = {'coef': beta_mle, 'var': mle_var, 'qvals': qvals}
                wald_test = pd.DataFrame(wald_test, index=cov_names)
            else:
                chi2_stats = beta_mle ** 2 / mle_var
                pvals = 1 - scipy.stats.chi2.cdf(chi2_stats, df=1)
                wald_test = {'coef': beta_mle, 'var': mle_var, 'chi2.wald': chi2_stats, 'pvals': pvals}
                wald_test = pd.DataFrame(wald_test, index=cov_names)
                if self.diff_intercept_by_measure:
                    assert sorted(cov_names) == sorted(self.covariates.columns.tolist() + [
                        '_junc_diff_']), 'reloaded regressiton table does not match covariates in this instance; aborting. check your file: %s, %s, %s' % (
                        outfp, sorted(cov_names), sorted(self.covariates.columns.tolist() + ['_junc_diff_']))
                else:
                    assert sorted(cov_names) == sorted(
                        self.covariates.columns.tolist()), 'reloaded regressiton table does not match covariates in this instance; aborting'
            self.stats_tests[eid] = wald_test
        if fast_load is False:
            _ = self.multiple_testing_correction()
        return self

    def get_das_cnt(self, target_covariate, qval_thresh=0.05):
        """Get a table of significant events for the target covariate
        """
        cov_levels = self.covariates.factor_conversion[target_covariate]
        cnt_df = {}
        for cov_level, cov_ in cov_levels.items():
            if cov_ in self.stats_sheet:
                cnt_df[cov_level] = (self.stats_sheet[cov_]['qvals'] < qval_thresh).sum()
        return cnt_df


class JemmLinearRegression(JointExonModel):
    """A simplified version for Jemm that only uses linear regression on Logit(PSI)
    """

    @staticmethod
    def _combine_measures(data):
        p = data['x.junc'].shape[1] if 'x.junc' in data else data['x.tpm'].shape[1]
        if 'x.junc' in data:
            x1 = data['x.junc']
            y1 = data['y.junc'] / data['N.junc']
            y1 = logit(y1)
        else:
            x1 = None
            y1 = None
        if 'x.tpm' in data:
            x2 = data['x.tpm']
            y2 = data['y.tpm']
        else:
            x2 = None
            y2 = None
        if x1 is not None and x2 is not None:
            x = np.concatenate([x1, x2], axis=0)
            y = np.concatenate([y1, y2], axis=0)
        else:
            x = x1 if x1 is not None else x2
            y = y1 if y1 is not None else y2
        return x, y

    def optimize(self, data, test_index=None, *args, **kwargs):
        assert data is not None and len(data) > 0
        from statsmodels.api import OLS
        if type(data) is dict:
            x, y = self._combine_measures(data)
        elif type(data) is tuple:
            x, y = data
        else:
            raise TypeError
        mod = OLS(y, x)
        res = mod.fit()
        opt_val = float(res.summary2().tables[0].loc[3, 3])
        reg_table = res.summary2().tables[1]
        beta_mle = reg_table['Coef.'].to_numpy()
        mle_var = reg_table['Std.Err.'].to_numpy() ** 2
        t_stats = reg_table['t'].to_numpy()
        pvals = reg_table['P>|t|'].to_numpy()
        t_test = {'coef': beta_mle, 'var': mle_var, 't': t_stats, 'pvals': pvals}
        if test_index is None:
            test_index = self.covariates.columns.tolist()
            test_index = test_index + ["_junc_diff_"] if len(pvals) == self.covariates.shape[1] + 1 else test_index
        t_test = pd.DataFrame(t_test, index=test_index)
        return t_test, opt_val


class JemmLinearMixedModel(JemmLinearRegression):
    """Linear Mixed Model for regressing Exon usage PSI to a set of fixed effect convariates and random intercepts

    Leverages statsmodels.regression.MixedLM for mixed model inference

    TODO
    ------
        Add random slope parsing; currently random slopes are not supported
    """

    def __init__(self, junction_measure, transcript_measure, covariates, group_varname,
                 diff_intercept_by_measure=True,
                 min_groupvar=1e-8,
                 optimizer='bfgs',
                 *args, **kwargs):
        """

        Parameters
        ----------
        junction_measure
        transcript_measure
        covariates
        group_varname
        diff_intercept_by_measure
        min_groupvar : float
            if group variance is smaller than the minimum value, remove the random effect and re-run
        args
        kwargs
        """
        super().__init__(junction_measure=junction_measure,
                         transcript_measure=transcript_measure,
                         covariates=covariates,
                         diff_intercept_by_measure=diff_intercept_by_measure,
                         *args,
                         **kwargs)
        self.__viable_tests = ['t-test', 'Wald']
        assert group_varname in covariates.covariate.columns, ValueError('Group variable not found in given covariates')
        self.min_groupvar = min_groupvar
        self._group_varname = group_varname
        self._group_cov_idx = np.where(self.covariates.covariate.columns == self._group_varname)[0][0]
        self.optimizer = optimizer.lower()
        assert self.optimizer in ('bfgs', 'lbfgs', 'cg'), ValueError('Unknown optimizer: %s' % self.optimizer)

    def optimize(self, data, test_index=None, *args, **kwargs):
        """

        Parameters
        ----------
        data
        test_index : None
            currently has no effect
        args
        kwargs

        Returns
        -------

        """
        from statsmodels.regression.mixed_linear_model import MixedLM
        assert data is not None and len(data) > 0
        if test_index is not None:
            test_index = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, y = self._combine_measures(data)
            group = x[:, self._group_cov_idx]
            x = np.delete(x, self._group_cov_idx, axis=1)
            mod = MixedLM(endog=y, exog=x, groups=group)
            try:
                res = mod.fit(method=self.optimizer, maxiter=2000)
                # variance is divided by scale:
                # see: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.html
                # why this is the case??
                group_var = res.params[x.shape[1]::] * res.scale
                if np.isscalar(group_var):
                    group_var = np.array([group_var])
            except np.linalg.LinAlgError:
                # Singular Matrix
                group_var = np.array([0])
            if test_index is None:
                test_index = self.covariates.columns.tolist()
                a = test_index.pop(self._group_cov_idx)
                test_index = test_index + ["_junc_diff_"] if (
                        'x.tpm' in data and 'x.junc' in data and self.diff_intercept_by_measure is True) else test_index
                test_index.append(a)

            if all(group_var > self.min_groupvar) and res.converged is True:
                opt_val = float(res.summary().tables[0].loc[3, 3])
                beta_mle = res.params[:x.shape[1]]
                beta_mle = np.concatenate([beta_mle, group_var], axis=0)
                mle_var = np.concatenate([res.bse_fe, res.bse_re]) ** 2
                t_stats = res.tvalues
                pvals = res.pvalues
                t_test = {'coef': beta_mle, 'var': mle_var, 't': t_stats, 'pvals': pvals}
                t_test = pd.DataFrame(t_test, index=test_index)
                return t_test, opt_val
            else:
                warnings.warn(
                    'Removed random effect due to smaller than min_groupvar; examine your model specification',
                    stacklevel=2)
                return super().optimize(data=(x, y), test_index=test_index[:-1])

    def predict(self, eid, data="both"):
        if data == "both":
            use_jct = True
            use_txr = True
        elif data == "jct":
            use_jct = True
            use_txr = False
        elif data == "txr":
            use_jct = False
            use_txr = True
        else:
            raise ValueError("Unknown data identifier: %s" % data)
        assert eid in self.stats_tests
        data, use_diff_intercept = self.get_data(eid, use_jct=use_jct, use_txr=use_txr)
        beta = self.stats_tests[eid]['coef']
        beta = beta.drop(self._group_varname)
        if use_diff_intercept is False and "_junc_diff_" in beta:
            beta = beta.drop("_junc_diff_", axis=0, inplace=False)
        beta = beta.to_numpy()
        fit = {}
        if 'y.junc' in data:
            z = np.matmul(np.delete(data['x.junc'], self._group_cov_idx, axis=1), beta)
            fit_junc = sigmoid(z)
            obs_junc = data['y.junc'] / data['N.junc']
            # fit['fit.junc'] = fit_junc
            # fit['obs.junc'] = obs_junc
            fit['junc'] = self.covariates.iloc[data['sample_index.junc']]
            fit['junc'] = fit['junc'].assign(obs=obs_junc, fit=fit_junc)
        if 'y.tpm' in data:
            z = np.matmul(np.delete(data['x.tpm'], self._group_cov_idx, axis=1), beta)
            fit_tpm = sigmoid(z)
            obs_tpm = sigmoid(data['y.tpm'])
            # fit['fit.tpm'] = fit_tpm
            # fit['obs.tpm'] = obs_tpm
            fit['tpm'] = self.covariates.iloc[data['sample_index.tpm']]
            fit['tpm'] = fit['tpm'].assign(obs=obs_tpm, fit=fit_tpm)
        return fit

    def _filter_machine_precision(self):
        for eid in self.stats_tests:
            reg_table = self.stats_tests[eid]
            reg_table.at[
                (reg_table['coef'].abs() < self._MIN_ABS_COEF) & (
                        reg_table.index != self._group_varname), 'pvals'] = 1.
            reg_table.at[
                (reg_table['coef'].abs() > self._MAX_ABS_COEF) & (
                        reg_table.index != self._group_varname), 'pvals'] = 1.

    def load_regression_table(self, outfp, fast_load=False):
        super(JemmLinearMixedModel, self).load_regression_table(outfp=outfp, fast_load=fast_load)
        outdf = pd.read_table(outfp)
        for i in range(outdf.shape[0]):
            row = outdf.iloc[i]
            eid = row['darts_id']
            cov_names = [str(x) for x in row['covariate_names'].split(',')]
            qvals = np.array([float(x) for x in row['covariate_qval'].split(',')])
            old_qvals = {cov_names[i]: qvals[i] for i in range(len(qvals))}
            self.stats_tests[eid].at[self._group_varname, 'qvals'] = old_qvals[self._group_varname]


# Alias
Jemm = JointExonModel
