'''Utility functions for TJI Analyses

Author: Everett Wetchler (everett.wetchler@gmail.com)
'''


import os


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns


#################################################################
# I. General Utilities
#################################################################


class PlotSaver(object):
    def __init__(self, plot_dir, plot_prefix, saving=True):
        self.saving = saving
        self.plot_dir = plot_dir
        self.plot_prefix = plot_prefix
        if not self.plot_prefix.endswith('_'):
            self.plot_prefix += '_'
        self.plot_count = 0
        if self.saving:
            self.clear_plots()

    def saveplot(self, fig, name):
        if self.saving:
            self.plot_count += 1
            filename = os.path.join(
                self.plot_dir, self.plot_prefix + "%02d_%s.png" % (self.plot_count, name))
            fig.savefig(filename)

    def clear_plots(self):
        prev_plots = [f for f in os.listdir(self.plot_dir) if f.startswith(self.plot_prefix)]
        print("Removing %d past plots" % len(prev_plots))
        for f in prev_plots:
            os.remove(os.path.join(self.plot_dir, f))

    def disable(self):
        self.saving = False

    def enable(self):
        self.saving = True


def test_summary(p, parens=True, r=None, chi=False, fisher=False):
    '''Create a string summarizing the results of a statistical test.'''
    if r is not None:
        test_name = 'pearson r' + r'$^2$' + ' = %.2f' % (r ** 2)
    elif chi:
        test_name = 'χ2 test'
    elif fisher:
        test_name = 'fisher exact test'
    else:
        raise Exception("Must specify one or {r, chi, fisher}")
    test_name = ', %s' % test_name
        
    fmt = "(%s%s)" if parens else "%s%s"
    if p < .001:
        p_str = "p < .001"
    elif p < .01:
        p_str = "p < .01"
    else:
        p_str = "p = %.3f" % p
    return fmt % (p_str, test_name)


def insert_col_after(df, s, name, after):
    '''Insert a new column into a dataframe, after a particular column.'''
    cols = list(df.columns)
    i = cols.index(after)
    newcols = cols[:(i+1)] + [name] + cols[(i+1):]
    df[name] = s
    return df[newcols]


def insert_col_before(df, s, name, before):
    '''Insert a new column into a dataframe, before a particular column.'''
    cols = list(df.columns)
    i = cols.index(before)
    newcols = cols[:i] + [name] + cols[i:]
    df[name] = s
    return df[newcols]


def insert_col_front(df, s, name):
    '''Insert a new column into a dataframe, in front as the first column.'''
    cols = list(df.columns)
    newcols = [name] + cols
    df[name] = s
    return df[newcols]


#################################################################
# II. EDA Utilities
#################################################################


# Summary statistics, missing data


def _truncate_str(s, n=15):
    '''Returns string s truncated to n characters, with elipsis as needed...'''
    if isinstance(s, str):
        return s if len(s) <= n else s[:(n - 3)] + '...'
    return s


def percentify_x(ax, decimals=None):
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=decimals))


def percentify_y(ax, decimals=None):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=decimals))


def smart_value_counts(s, top=10, normalize=True, percent=False,
                       sort_index=False, truncate=True):
    '''Similar to s.value_counts() in pandas, but with rollups and pcts.'''
    vc = s.value_counts(ascending=False, normalize=normalize)
    if sort_index:
        vc = vc.sort_index()
    if truncate:
        vc.index = [_truncate_str(i) for i in vc.index]
    out = vc.iloc[:top]
    if vc.sum() > out.sum():
        out.loc['[OTHER] (%d types)' % (len(vc) - len(out))] = vc.sum() - out.sum()
    na = s.isnull().sum()
    if normalize:
        na /= len(s)
    if na > 0:
        out = out.append(pd.Series([na], index=['[NA (%.0f%%)]' % abs(na * 100)]))
    if percent:
        return out.apply(lambda p: '%.2f%%' % abs(p * 100))
    return out


COLOR_HIST = (0.30, 0.45, 0.69) # Blue
COLOR_BAR = (0.33, 0.66, 0.41)  # Green
COLOR_NA = (0.39, 0.71, 0.80)  # Cyan
ALPHA_BAR = 0.8


def smart_hist(s, ax, rugplot=False, color=COLOR_HIST,
               color_na=COLOR_NA, alpha=ALPHA_BAR):
    s = s.astype(float)
    s.hist(ax=ax, color=color, weights=np.zeros(s.count()) + 1. / len(s),
           alpha=alpha)
    ax.set_yticklabels(['%.0f%%' % (t * 100) for t in ax.get_yticks()])
    na = 1.0 - float(s.count()) / len(s)
    if na > 0:
        ax.plot([min(ax.get_xticks()), max(ax.get_xticks())], [na, na],
                color=color_na, label='NA (%.0f%%)' % (na * 100),
                alpha=0.8 if na else 0,
                linestyle='--')
        ax.legend(loc='upper right')
    if rugplot:
        sns.rugplot(s, color=color, alpha=1.0, ax=ax)


def smart_bar(s, ax, color=COLOR_BAR, color_na=COLOR_NA, alpha=ALPHA_BAR,
              sort_index=False, truncate=True):
    s = smart_value_counts(s, truncate=truncate)
    s[::-1].plot(ax=ax, kind='barh', color=color, alpha=alpha)
    ax.set_xticklabels(['%.0f%%' % (t * 100) for t in ax.get_xticks()])


def dist_plot(s, ax, color=COLOR_BAR, color_na=COLOR_NA, alpha=ALPHA_BAR,
              sort_index=False):
    pass


# Correlations


def scatter(df, x_col, y_col, labels=None):
    fig, ax = plt.subplots(1)
    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=40)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if labels is not None:
        for i, lab in enumerate(labels):
            x, y = (df[x_col].iloc[i], df[y_col].iloc[i])
            ax.annotate(lab, xy=(x, y), xytext = (x - 60, y + 20), fontsize=12)
    return fig, ax


def scatter_colored(df, x_col, y_col, color_var):
    fig, ax = plt.subplots(1)
    colors = sns.color_palette('hls', len(df[color_var].value_counts()))
    for i, pair in enumerate(sorted(list(df.groupby(color_var)))):
        name, group = pair
        x, y = group[x_col], group[y_col]
        if x.count() and y.count():
            ax.scatter(x, y, alpha=0.6, s=40, color=colors[i], label=name)
    ax.legend()
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig, ax


def univariate_corrs_bar(df, to_predict, method='pearson'):
    if not isinstance(to_predict, list):
        to_predict = [to_predict]
    corr = df.corr(method=method.lower())[to_predict]
    corr.drop(to_predict, axis=0, inplace=True)
    corr = corr.sort_index()[::-1]
    fig, ax = plt.subplots(1)
    fig.set_size_inches(12, 8)
    corr.plot(ax=ax, kind='barh')
    ax.set_title('Linear univariate correlations (Pearson R), N=%d' % len(df))
    ax.set_xlabel('R (%s)' % method.capitalize())
    ax.set_yticklabels(
        ['%s (%.0f%% NA)' % (
            c.get_text(),
            100 * (1 - float(df[c.get_text()].count())
                   / len(df[c.get_text()])))
            for c in ax.get_yticklabels()],
        fontsize=14)
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    fig.subplots_adjust(left=0.4)
    return fig, ax


def add_fit(ax, a, b, orders=1, colors=None, fig=None):
    if not isinstance(orders, list):
        orders = [orders]
    if not colors:
        colors = ['gray']
        if len(orders) > 0:
            colors = sns.color_palette()[1:]
        if len(orders) > colors:
            colors = sns.color_palette('hls', len(orders))
    for i, o in enumerate(orders):
        col = colors[i]
        coeffs = np.polyfit(a, b, o)
        f = np.poly1d(coeffs)
        pred = f(a)
        r, p = pearsonr(pred, b)
        if o == 1 and coeffs[0] < 0:
            r = -r
        x = np.linspace(min(a), max(a), 200)
        y = f(x)
        lab = 'Fit line' if o == 1 else 'Fit deg-%d' % o
        ax.plot(x, y, color=col, alpha=0.7, linestyle='--',
                linewidth=2, label='%s (R2 = %.2f, p = %.2f)' % (lab, r, p))
    ax.legend(loc='upper right')
