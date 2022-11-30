# -*- coding: utf-8 -*-

"""This module provides construstors and manipulations for covariate matrix
"""

# Author: zzjfrank
# Date : Sep. 3, 2020

from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import scipy.stats
import itertools


class Contrasts:
    def __init__(self, name, levels):
        self.name = name
        self.levels = levels

    def __add__(self, other):
        assert type(other) is Contrasts
        if type(self.name) is not list:
            sname = [self.name]
            slevels = [self.levels]
        else:
            sname = self.name
            slevels = self.levels
        if type(other.name) is not list:
            oname = [other.name]
            olevels = [other.levels]
        else:
            oname = other.name
            olevels = other.levels
        return Contrasts(
                name = sname + oname,
                levels = slevels + olevels
            )

    def apply(self, df):
        if type(self.name) is list:
            for name, levels in zip(self.name, self.levels):
                df_w_contrast = df.loc[df[name].isin(levels)]
                df = df_w_contrast
        else:
            df_w_contrast = df.loc[ df[self.name].isin(self.levels) ]
        return df_w_contrast


class Covariate:
    def __init__(self, fp, contrasts, main_effects, interaction_effects=None, index_col=0, col_to_keep=None, factor_conversion="auto", sep=",", verbose=False):
        """

        Parameters
        ----------
        fp : str, or pandas.DataFrame
        contrasts : jemm.covariate.Contrasts
        main_effects : list of str
        interaction_effects : list of str
        index_col : int
        col_to_keep : list
        factor_conversion : str or dict
        sep : str
        verbose : bool
        """
        self._verbose = verbose
        if type(fp) is str:
            self.meta = pd.read_table(fp, index_col=index_col, sep=sep)
        elif type(fp) is pd.DataFrame:
            self.meta = fp
        else:
            raise TypeError('fp must be str or pd.DataFrame; got %s' % type(fp))
        if col_to_keep is not None:
            self.meta = self.meta[col_to_keep]
        # this is a bit dangerous but have to do..
        self.meta = self.meta.dropna(axis=1)
        self.contrasts = contrasts
        self.meta = self.contrasts.apply(self.meta)
        self.factor_conversion = self._auto_factor_conversion() if factor_conversion == "auto"  else \
                factor_conversion
        self._main_effects = main_effects
        self._interaction_effects = interaction_effects or []
        self.covariate, interaction_to_col = self.convert_factor_to_num(
            meta=self.meta,
            main_effects=self._main_effects,
            interaction_effects=self._interaction_effects,
            factor_conversion=self.factor_conversion,
            verbose=self._verbose)
        self._interaction_to_col = interaction_to_col
        self.covariate.insert(0, "intercept", 1)

    @property
    def index(self):
        return self.covariate.index

    @property
    def columns(self):
        return self.covariate.columns

    @property
    def shape(self):
        return self.covariate.shape

    @property
    def iloc(self):
        return self.covariate.iloc

    @property
    def loc(self):
        return self.covariate.loc

    @property
    def formula(self):
        main = " + ".join(self._main_effects)
        intr = " + ".join(self._interaction_effects)
        f = main + " + " + intr if len(intr) else main
        return "y = %s" % f

    def match_index(self, *args):
        arg_index = np.intersect1d(*args) if len(args) > 1 else args[0]
        self.covariate = self.covariate.loc[self.covariate.index.isin(arg_index)]

    def to_numpy(self):
        return self.covariate.to_numpy()

    def get_design_mat(self, covariates=None):
        if covariates is None:
            main_eff = self._main_effects
            intr_eff = self._interaction_effects
        else:
            main_eff = [x for x in covariates if x in self._main_effects]
            intr_eff = [x for x in covariates if x in self._interaction_effects]
        cols = []
        for x in main_eff:
            if x in self.factor_conversion:
                cols += [t for _, t in self.factor_conversion[x].items()]
            else:
                cols += [x]
        cols += [t for x in intr_eff for t in self._interaction_to_col[x]]
        design_mat = self.covariate[cols]
        return design_mat

    @staticmethod
    def convert_factor_to_num(meta, main_effects, interaction_effects, factor_conversion, min_var_allowed=1e-5, max_interaction_cor_allowed=0.99, verbose=False):
        """Static method for converting a metadata to a covariate matrix

        A design/covariate matrix can be separated into two parts: the main effects and the interaction effects.
        Under the hood, the main effects are further separately considered as 1) continuous variables, and 2) factor/categorical
        variables.
        The interaction effects are simply the multiplication of two or more main effects

        Parameters
        ----------
        meta : pandas.DataFrame
        main_effects : list of str
        interaction_effects : list of str
            Different terms are separated by bar (|). For example, interaction of x1 and x2 is "x1|x2".
        factor_conversion : dict
        min_var_allowed : float
            The minimum variance for a covariate. We don't want any constants, because an intercept term will be added automatically.
            Default is 1e-5.
        max_interaction_cor_allowed : float
            The maximum correlation between an interaction term and its main terms allowed. Default value is 0.99; i.e. we keep all
            interactions whose pearson correlation coefficient is <0.99 with any of its main effects.
        verbose : bool
            verbose mode. Default is False.
        """
        # cols is the dict that will be filled-up and convert to covariates
        cols = {}
        for x in main_effects:
            col = meta[x]
            if x in factor_conversion:
                for k, v in factor_conversion[x].items():
                    new_col = np.zeros(col.shape)
                    idx = (col==k)
                    new_col[idx] = 1
                    var = np.var(new_col)
                    if var < min_var_allowed:
                        if verbose is True: print("skipped main effect %s because var=%.4f < %.4f, disallowed"%(v, var, min_var_allowed))
                        continue
                    cols[v] = new_col
            else:
                var = np.var(col)
                if var < min_var_allowed:
                    if verbose is True: print("skipped main effect %s because var=%.4f < %.4f, disallowed"%(x, var, min_var_allowed))
                    continue
                cols[x] = col

        # create a dict that maps interaction name (e.g. x1|x2) to factorized 
        # children (e.g. [x1@True|x2@True, x1@True|x2@False, ...])
        interaction_to_col = defaultdict(list)
        for xx in interaction_effects:
            ele = xx.split("|")
            ele = [ list(factor_conversion[x][_] for _ in factor_conversion[x]) if x in factor_conversion else [x] for x in ele   ]
            factorized_combinations = [a for a in itertools.product(*ele)]
            for col_comb in factorized_combinations:
                col = np.product([cols[x] for x in col_comb], axis=0)
                colname = "|".join(col_comb)
                is_correlated_intr = False
                for x in col_comb:
                    pcor = scipy.stats.pearsonr(cols[x], col)[0]
                    if pcor > max_interaction_cor_allowed:
                        is_correlated_intr = True
                        break
                if is_correlated_intr:
                    if verbose is True: print("skipped interaction %s because main-effect pcor=%.4f > %.4f, disallowed"%(colname, pcor, max_interaction_cor_allowed))
                    continue
                cols[colname] = col
                interaction_to_col[xx].append(colname)

        covariates = pd.DataFrame.from_dict(cols)
        covariates.index = meta.index
        return covariates, interaction_to_col

    def _auto_factor_conversion(self, max_factor_level=3):
        factor_conversion = {}
        for x in self.meta:
            obs_vals = self.meta[x].unique()
            if pd.api.types.is_string_dtype(self.meta[x]):
                is_factor = True
            elif len(obs_vals) < max_factor_level:
                is_factor = True
            else:
                is_factor = False
            if is_factor:
                obs_vals = sorted(obs_vals)[1:]
                factor_conversion[x] = OrderedDict({v:"%s@%s"%(x, v) for v in obs_vals})
        return factor_conversion





