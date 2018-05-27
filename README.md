# Texas Justic Initiative: Analysis Library

To learn more about TJI, visit our website at www.texasjusticeinitiative.org

## About this repo

This is a general repository for all presentable files conducting analysis of TJI-related data. This is not a place for preprocessing steps (see [tji/data-processing](https://github.com/texas-justice-initiative/data-processing)). All data used for analysis should be read directly from data.world (see TJI's data.world account [here](https://data.world/tji)).

If you are not a coder, feel free to skim the analyses here for their summaries (at the top) and charts (embedded - skim for them). You can also browse the [plots/](https://github.com/texas-justice-initiative/analysis/tree/master/plots) directory to jump right into the pile of results.

## Data Scientists:

If you are a coder, feel free to reproduce and alter the code for your own theories. If you identify bugs or concerns with our work, please file an [issue](https://github.com/texas-justice-initiative/analysis/issues) so we can ensure that our results our accurate.

## TJI Team

* To conduct your own analysis, clone the [analyses/TJI_ANALYSIS_TEMPLATE.ipynb](https://github.com/texas-justice-initiative/analysis/analyses/TJI_ANALYSIS_TEMPLATE.ipynb) file and work from there.

## Overview of analyses contained

#### I. Descriptive analyses
* [Who makes up the Texas police force?](https://github.com/texas-justice-initiative/analysis/blob/master/analyses/officer_population_descriptive_analysis.ipynb)
* [Summary of civilians shot by police](https://github.com/texas-justice-initiative/analysis/blob/master/analyses/ois_descriptive_analysis_civilians_shot.ipynb)
* [Summary of police shot by civilians](https://github.com/texas-justice-initiative/analysis/blob/master/analyses/ois_descriptive_analysis_officers_shot.ipynb)
* [Summary of deaths in custody](https://github.com/texas-justice-initiative/analysis/blob/master/analyses/cdr_explore.ipynb)

#### II. Correlational investigations
* [What kinds of officers are involved in shootings?](https://github.com/texas-justice-initiative/analysis/blob/master/analyses/ois_which_officers.ipynb)
* [What determines if a civilian will survive a shooting?](https://github.com/texas-justice-initiative/analysis/blob/master/analyses/ois_who_survives_shootings.ipynb)

## About the datasets

* See our [data-processing github repository](https://github.com/texas-justice-initiative/data-processing) for detailed descriptions of all datasets TJI uses, plus the code we use in data wrangling and cleaning.
* See our [data.world repository](https://data.world/tji) for the actual datasets, which are distributed across several data.world projects ([Officer Involved Shootings](https://data.world/tji/officer-involved-shootings) | [Custodial Deaths](https://data.world/tji/tx-deaths-in-custody-2005-2015) | [Other Data](https://data.world/tji/auxiliary-datasets))
