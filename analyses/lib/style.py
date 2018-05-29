'''Tools to style plots according to the TJI brand.

Author: Everett Wetchler (everett.wetchler@gmail.com)
'''


import matplotlib as mpl
import seaborn as sns


#################################################################
# Font and graph sizing
#################################################################


mpl.rcParams.update({
  'font.size': 14,
  'axes.titlesize': 'xx-large',
  'axes.labelsize': 'x-large',
  'xtick.labelsize': 'large',
  'ytick.labelsize': 'large',
  'legend.fancybox': True,
  'legend.fontsize': 'large',
  'legend.frameon': True,
  'legend.framealpha': 0.7,
  'figure.figsize': ['9', '6'],
  'lines.linewidth' : 4,
  'lines.solid_joinstyle': 'miter',
})


#################################################################
# Set up color palette
#################################################################


TJI_BLUE = '#0B5D93'
TJI_RED = '#CE2727'
TJI_DEEPBLUE = '#252939'
TJI_YELLOW = '#F1AB32'
TJI_BRIGHTYELLOW = '#FFFD00'
TJI_PURPLE = '#4D3B6B'
TJI_DARKGRAY = '#3F3F40'
TJI_DEEPPURPLE = '#2D1752'
TJI_DEEPRED = '#872729'
TJI_TEAL = '#50E3C2'
TJI_PALETTE = sns.color_palette([
    TJI_BLUE, TJI_RED, TJI_YELLOW, TJI_DEEPBLUE,
    TJI_PURPLE, TJI_TEAL, TJI_DEEPRED, TJI_DEEPPURPLE,
])
sns.set_palette(TJI_PALETTE)
