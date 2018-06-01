'''Utility functions for TJI Analyses

Author: Everett Wetchler (everett.wetchler@gmail.com)
'''


import os


import matplotlib as mpl
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
    def __init__(self, plot_dir, plot_prefix, saving=True, numbering=True, dpi=None):
        self.saving = saving
        self.numbering = numbering
        self.plot_dir = plot_dir
        self.plot_prefix = plot_prefix
        self.dpi = dpi
        if not self.plot_prefix.endswith('_'):
            self.plot_prefix += '_'
        self.plot_count = 0
        if self.saving:
            self.clear_plots()

    def saveplot(self, fig, name, dpi=None):
        if self.saving:
            self.plot_count += 1
            num = ('%02d_' % self.plot_count) if self.numbering else ''
            filename = os.path.join(
                self.plot_dir, self.plot_prefix + num + name + '.png')
            print('Saving plot to', filename)
            fig.savefig(filename, dpi = dpi or self.dpi)

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


def alter_last_bar(ax, alpha=0.5, hatch='//', hatchalpha=0.2):
    '''Change the look of the last bar in a bar chart to indicate incomplete data.'''
    bars = [c for c in ax.get_children() if isinstance(c, mpl.patches.Rectangle)]
    xmax = max(b.xy[0] for b in bars)
    for b in bars:
        if b.xy[0] == xmax:
            b.set_hatch(hatch)
            b.set_edgecolor((1, 1, 1) + (hatchalpha,))
            b.set_facecolor(b.get_facecolor()[:3] + (alpha,))


def pie_plot(df, title='', donut=False,
             suptitle_size=30, axtitle_size=24, label_size=16,
             figsize=None, min_pct_for_label=3,
             **kwargs):
    # Normalize to plot fractions
    df = df.div(df.sum())

    # Plot sections clockwise, unless the user says otherwise
    if 'counterclock' not in kwargs:
        kwargs['counterclock'] = False
        
    # User asked for a donut, but didn't specify
    # an inner radius size. Pick a good default.
    if donut == True:
        donut = 0.7

    # By default, add white gaps between donut sections
    if donut:
        kwargs['wedgeprops'] = kwargs.get('wedgeprops', {})
        if 'linewidth' not in kwargs['wedgeprops']:
            kwargs['wedgeprops']['linewidth'] = 3
            kwargs['wedgeprops']['edgecolor'] = 'white'
    
    axes = df.plot(
        kind='pie', subplots=True, figsize=figsize or (6 * len(df.columns), 6),
        autopct='%.0f%%', startangle=180,
        textprops={'color': 'black', 'fontsize': label_size, 'fontweight': 'bold'},
        pctdistance=max(0.6, 0.5 + donut/2),
        **kwargs)

    for ax, t in zip(axes, df.columns):
        ax.set_ylabel('')
        ax.get_legend().set_visible(False)
        ax.axis("equal")

        if donut > 0:
            my_circle=plt.Circle((0,0), donut, color='white')
            ax.add_artist(my_circle)
            t = '\n'.join(t.rsplit(' ', 1))
            ax.text(0, 0, t, verticalalignment='center', horizontalalignment='center',
                    fontweight='bold', size=axtitle_size)
            ax.set_title('')
        else:
            ax.set_title(t, fontweight='bold', size=axtitle_size, pad=75)

        i = 0
        for child in ax.get_children():
            if isinstance(child, plt.Text) and '%' in child.get_text():
                child.set_fontweight('bold')
                child.set_color('white')
                child.set_family('monospace')
                child.set_fontsize(label_size)
                color = sns.color_palette()[i]
                pct = float(child.get_text().strip('%'))
                if pct < min_pct_for_label:
                    child.set_visible(False)
                i += 1

    fig = plt.gcf()
    if title:
        fig.suptitle(title, y=.95, fontweight='bold', fontsize=suptitle_size,
                     bbox=dict(facecolor='none', edgecolor='black', pad=10))
    plt.subplots_adjust(wspace=.5, left=.1, right=.9, top=.8 if donut else .6, bottom=.05 if donut else 0.1)
    return fig, ax
