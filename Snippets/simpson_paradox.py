import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from machine_learning.commons import check_dataframe


def simpson_correlation(df: pd.DataFrame, field_to_partition: str, partition_list: list, cols_correlated: list) -> dict:
    """
    field_to_partition: column in dataframe df eg. 'State'
    partition_list: list county_data[field_to_partition].unique().tolist(): county_data['State'].unique().tolist()
    cols_correlated: ['SelfEmployed', 'IncomePerCap']
    ele_to_plot='Connecticut'
    """

    def plot_correlation():
        f, axes = plt.subplots(1, 2, figsize=(12, 8))
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        sns.scatterplot(x=df[cols_correlated[0]], y=df[cols_correlated[1]], ax=axes[0])
        sns.barplot(x=corrs['spearman'], y=corrs['spearman'].index.values, ax=axes[1])
        plt.title('Spearman correlation')
        plt.tight_layout()
        plt.grid(True)
        plt.suptitle('Simpson paradox', fontsize=20)
        plt.subplots_adjust(top=.9, wspace=.3)
        plt.show()

    check_dataframe(df)
    # check that partition_list is in df
    assert len([ele for ele in partition_list if ele]) == len(partition_list)

    methods = ['pearson', 'spearman']
    corrs: dict = dict.fromkeys(methods)
    for method in methods:
        correlations: list = []
        for ele in partition_list:
            correlations.append(df[cols_correlated][df[field_to_partition] == ele].corr(method=method).iloc[0, 1])
        corrs[method]: pd.Series = pd.Series(correlations, index=partition_list)
        corrs[method].dropna(axis=0, inplace=True)

    plot_correlation()
    print('Correlations: \n')
    print('\t Pearson: ')
    print('\n {}'.format(corrs['pearson']))
    print('\t Spearman:')
    print('\n {}'.format(corrs['spearman']))

    return corrs
