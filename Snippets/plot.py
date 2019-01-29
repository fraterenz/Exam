import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from machine_learning.commons import check_serie, check_dataframe


def describe_data(serie: pd.Series, title, nb_bins=None, kde=True, figsize=(10, 6)) -> pd.DataFrame:
    """ Returns describe and plots of the Serie: plots are a boxplot and a histogram

    :type kde: bool
    :type nb_bins: int
    :type title: str
    :type serie: pd.Series
    :param serie: serie to be described
    :param title: title of the figure
    :param nb_bins: nb of bins of hist
    :param kde: wether or not to put a contour line in the hist
    :return: described dataframe with plots
    """
    check_serie(serie)
    df = pd.DataFrame(data=serie)
    f, axes = plt.subplots(1, 2, figsize=figsize)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.boxplot(data=df, ax=axes[0])
    sns.distplot(df, color="b", kde=kde, ax=axes[1], bins=nb_bins)
    plt.tight_layout()
    plt.grid(True)
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=.9)
    plt.show()
    return df.describe().T


def pairwise_plot(df: pd.DataFrame, spearman=True) -> pd.DataFrame:
    """
    pairwise plot 
    by default spearman correlation
    """
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.pairplot(df)
    plt.suptitle('Pairwise plot', fontsize=20)
    plt.subplots_adjust(top=.9)
    plt.show()
    if spearman:
        return df.corr(method='spearman')
    return df.corr(method='pearson')


def regression_plot(df: pd.DataFrame, col1: str, col2: str, title: str, figsize=(7, 5)):
    """
    ScatterPlot data give dataframe df, col1 name, col2 name, title, and a linear regression model fit
    return spearman corrrelation by default
    """
    check_dataframe(df)
    correlation_pearson = df[[col1, col2]].corr(method='pearson').iloc[0, 1]
    correlation_spearman = df[[col1, col2]].corr(method='spearman').iloc[0, 1]
    f, axes = plt.subplots(1, 1, figsize=figsize)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.regplot(df[col1], df[col2], scatter_kws={'s': 3, 'color': 'blue'}, line_kws={'color': 'red'}, ax=axes)
    f.suptitle(title)
    plt.show()

    print('Pearson correlation coefficient is: {:.2}   \n'.format(correlation_pearson))
    print('Spearman correlation coefficient is: {:.2}   \n'.format(correlation_spearman))

    return correlation_pearson, correlation_spearman


def cumulative_distribution(serie: pd.Series):
    check_serie(serie)
    # cumulative distribution
    maxTagsCounts = int(serie.max())

    step = 100
    xs = pd.Series(range(1,maxTagsCounts,step))

    # count elements in pdTagsCounts.counts that are greater than x
    gratherThanData = xs.apply(lambda x: (serie.counts[serie.counts>=x]).count())

    f, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax1)
    plt.plot(xs,gratherThanData)
    plt.ylabel('question counts',fontsize=16)
    plt.xlabel('# tags',fontsize=16)

    # log log plot of cumulative distribution
    plt.sca(ax2)
    plt.plot(xs,gratherThanData)
    plt.xlabel('# tags',fontsize=16)
    plt.yscale('log'); plt.xscale('log')

    f.set_size_inches(18, 5)
    f.suptitle('numbers of tags per question counts', fontsize=20)
    plt.show()
