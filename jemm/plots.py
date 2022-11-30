"""This module hosts plotting and visualization functions
"""

# Author : zzjfrank
# Date   : Aug. 31, 2020


import matplotlib.pyplot as plt
import seaborn as sns
import six
import scipy.stats as ss
import numpy as np


def facet_boxplot(fit, regression_table, eid=None, x="tp", y="obs", contrast_col="final", facet_col=None, save_fname=None, font_size=12):
    """
    Examples
    --------
    Plot a facet boxplots ::
        facet_boxplot(
            fit=jmm.predict(eid),
            regression_table=jmm.stats_tests[eid],
            eid=eid,
            contrast_col='final@Late',
            x='tp@28',
            y='obs',
            facet_col='Sex@M',
        )
    """
    plt.clf()
    plt.tight_layout()
    fit_keys = sorted([k for k in fit])
    if facet_col is None:
        facet_col = "intercept"
    ncols = len(fit[fit_keys[0]][facet_col].unique())+2
    nrows = len(fit_keys)
    fig, axs = plt.subplots(nrows=len(fit_keys), ncols=ncols,
            sharey=True,
            figsize=(6*ncols, 6*nrows))
    facets = fit[fit_keys[0]][facet_col].unique()
    assert len(facets) < 15, "Too many facets (>=15)"
    tables = fit_keys
    for k in range(len(tables)):
        fit_ = fit[tables[k]]
        for i, f in enumerate(facets):
            ax = axs[k,i] if len(tables)>1 else axs[i]
            ax.set_ylim(0, 1)
            subdf = fit_.loc[fit_[facet_col]==f]
            sns.boxplot(x=x, y=y, hue=contrast_col, data=subdf, ax=ax, showfliers=False)
            sns.stripplot(x=x, y=y, hue=contrast_col, data=subdf, ax=ax,
                    size=4,
                    color="black",
                    edgecolor="black", dodge=True)
            ax.set_title(label="%s|%s=%s"%(tables[k], facet_col, f))
            ax.get_legend().remove()
        # scatter plot of fit vs obs
        scatter_ax = axs[k,ncols-2] if len(tables)>1 else axs[ncols-2]
        scatter_ax.set_xlim(scatter_ax.get_ylim())
        sns.scatterplot(x="fit", y="obs", hue=contrast_col, style=facet_col, data=fit_, ax=scatter_ax)
        sns.regplot(x="fit", y="obs", scatter=False, data=fit_, truncate=False, ax=scatter_ax, ci=None)
        scatter_ax.plot([0,1], [0,1], linestyle="--", color="grey")
        pearsonr = ss.pearsonr(x=fit_['fit'], y=fit_['obs'])
        scatter_ax.set_title("%s, r=%.2f, p=%.2f"%(tables[k], pearsonr[0], pearsonr[1]))
    bbox=[0.3, 0, 0.9, 1]
    table_ax = axs[0, ncols-1] if len(tables)>1 else axs[ncols-1]
    table_ax.axis("off")
    mpl_table = table_ax.table(cellText=regression_table.round(3).values, bbox=bbox, colLabels=regression_table.columns, rowLabels=regression_table.index)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    if eid is not None:
        table_ax.set_title(eid)

    # beatify the table; 
    # code is from https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure 
    header_color='#40466e'
    header_columns = 0
    row_colors=['#f1f1f2', 'w']
    edge_color='w'
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(fontstretch="extra-condensed", weight='semibold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    if save_fname is not None: plt.savefig(save_fname)
    plt.close()
    return fig

def beta_barplot(regression_table, cols_to_plot, eid, ax=None):
    subdf = regression_table.loc[cols_to_plot]
    subdf['Conditions'] = subdf.index
    subdf['std'] = 2*np.sqrt(subdf['var'])
    subdf['color'] = 'red'
    subdf.loc[subdf['coef']<0, 'color'] = 'blue'
    if ax is None:
        plt.close()
        fig, ax = plt.subplots(1,1, figsize=(4, 1*len(cols_to_plot)))
    subdf.plot.barh(x="Conditions", y='coef', xerr='std', color=subdf['color'], capsize=3, ax=ax)
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_title(eid)
    ax.set_yticklabels(["%s\npadj=%.3f"%(subdf['Conditions'][i], subdf['qvals'][i]) for i in range(subdf.shape[0])])
    return ax


def volcano_plot(x, y, xcut=None, ycut=None, labels=None, data=None, ax=None):
    """Plot a volcano plot in a data frame

    Parameters
    ----------
    x : str
    y : str
    labels : str
    data : pandas.DataFrame

    Returns
    --------
    matplotlib.pyplot.Axes
    """
    plot_data = data.copy()
    plot_data['color'] = 'black'
    plot_data.at[(plot_data[x]>= xcut)&(plot_data[y]>=ycut), 'color'] = 'red'
    plot_data.at[(plot_data[x]<= -xcut)&(plot_data[y]>=ycut), 'color'] = 'blue'
    ax = sns.scatterplot(x=x, y=y, hue='color', hue_order=['black', 'red', 'blue'],
                    palette=['black', 'red', 'blue'],
                    alpha=0.8,
                    ax=ax, data=plot_data)
    ax.axhline(ycut, linestyle='--', color='black', alpha=0.5)
    ax.axvline(xcut, linestyle='--', color='black', alpha=0.5)
    jitter_ = (plot_data[x].max() - plot_data[x].min())/50
    if labels is not None:
        for i in range(plot_data.shape[0]):
            lab = plot_data.iloc[i][labels]
            if lab == '':
                continue
            ax.text(plot_data.iloc[i][x] - jitter_, plot_data.iloc[i][y], lab,
                    horizontalalignment='right',
                    verticalalignment='bottom',)
    ax.legend().remove()
    return ax


