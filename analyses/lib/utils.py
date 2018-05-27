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
# General Utilities
#################################################################


class PlotSaver(object):
    def __init__(self, plot_dir, plot_prefix, saving=True, numbering=True):
        self.saving = saving
        self.numbering = numbering
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
            num = ('%02d_' % self.plot_count) if self.numbering else ''
            filename = os.path.join(
                self.plot_dir, self.plot_prefix + num + name + '.png')
            fig.savefig(filename)

    def clear_plots(self):
        prev_plots = [f for f in os.listdir(self.plot_dir) if f.startswith(self.plot_prefix)]
        print('Removing %d past plots' % len(prev_plots))
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
        raise Exception('Must specify one or {r, chi, fisher}')
    test_name = ', %s' % test_name
        
    fmt = '(%s%s)' if parens else '%s%s'
    if p < .001:
        p_str = 'p < .001'
    elif p < .01:
        p_str = 'p < .01'
    else:
        p_str = 'p = %.3f' % p
    return fmt % (p_str, test_name)


def insert_col_after(df, s, name, after):
    '''Insert a new column into a dataframe, after a particular column.'''
    df = df.copy()
    cols = list(df.columns)
    i = cols.index(after)
    newcols = cols[:(i+1)] + [name] + cols[(i+1):]
    df[name] = s
    return df[newcols]


def insert_col_before(df, s, name, before):
    '''Insert a new column into a dataframe, before a particular column.'''
    df = df.copy()
    cols = list(df.columns)
    i = cols.index(before)
    newcols = cols[:i] + [name] + cols[i:]
    df[name] = s
    return df[newcols]


def insert_col_front(df, s, name):
    '''Insert a new column into a dataframe, in front as the first column.'''
    df = df.copy()
    cols = list(df.columns)
    newcols = [name] + cols
    df[name] = s
    return df[newcols]


#################################################################
# EDA Utilities
#################################################################


def shortyear_xticks(ax):
    ax.set_xticklabels(["'%02d" % (t % 100) for t in ax.get_xticks()])


def _truncate_str(s, n=15):
    '''Returns string s truncated to n characters, with elipsis as needed...'''
    if isinstance(s, str):
        return s if len(s) <= n else s[:(n - 3)] + '...'
    return s


def percentify_x(ax, decimals=None):
    '''Convert decimals in x-axis tick labels to percents (e.g. 0.05 -> 5%)'''
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=decimals))


def percentify_y(ax, decimals=None):
    '''Convert decimals in y-axis tick labels to percents (e.g. 0.05 -> 5%)'''
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=decimals))


def smart_value_counts(s, top=10, normalize=True, percent=False,
                       sort_index=False, truncate=True):
    '''Similar to s.value_counts() in pandas, but rolling up rare values.'''
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
