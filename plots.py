import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import warnings
from pandas.api.types import CategoricalDtype


def filter_na_quantile(data):
    print('Previous Shape', data.shape)
    # Drop nan from CEVAE
    data = data.dropna(axis=0)
    print('New Shape after remove NAN', data.shape)
    data.reset_index(inplace=True, drop=True)

    # Drop the top 10 worse results from all methods
    filter_quantile = data.groupby(['Method']).quantile(0.85)['MAE']
    filter_quantile = pd.DataFrame(filter_quantile)
    filter_quantile.reset_index(inplace=True)
    print(filter_quantile)

    drop_rows = []
    for i in range(data.shape[0]):
        th = filter_quantile[filter_quantile['Method'] == data['Method'][i]]
        th = th['MAE'].values
        if data['MAE'][i] >= th:
            drop_rows.append(i)
    print(drop_rows)

    data.drop(index=drop_rows, inplace=True)
    print('New shape after removing worse 10%', data.shape)
    return data


def read_all_configs(listfile, path, dataname='gwas'):
    table = pd.DataFrame()
    for i, config in enumerate(listfile['config'].values):
        # print(path ,'output_' , dataname ,'_' , config , '.csv')
        _table = pd.read_csv(path + 'output_' + dataname + '_' + config + '.csv').iloc[:, 1:]
        _table['config'] = listfile['config'][i]
        _table['sample_size'] = listfile['sample_size'][i]
        _table['treatments'] = listfile['treatments'][i]
        _table['confounders'] = listfile['confounders'][i]
        table = pd.concat([table, _table], axis=0)
    # table.reset_index(drop=True,inplace=True)
    table['type'] = 'baseline'
    table['type'][table['Method'] == 'M3E2'] = 'proposed'
    table['Method'] = table['Method'].str.replace("Dragonnet", 'Drag.')
    table = table[table['Method'] != 'TrueTreat']
    table.reset_index(drop=True, inplace=True)

    # table['sample_size'] = table['sample_size'].astype(str)
    # table['treatments'] = table['treatments'].astype(str)
    # table['confounders'] = table['confounders'].astype(str)
    return table


def read_one_config(path, dataname='copula', config='config2a'):
    data = pd.read_csv(path + 'output_' + dataname + '_' + config + '.csv').iloc[:, 1:]
    data['Method'] = data['Method'].str.replace("Dragonnet", 'Drag.')
    data['type'] = 'Baselines'
    data['type'][data['Method'] == 'M3E2'] = 'Proposed'

    TrueEffect = data[data['Method'] == 'TrueTreat']
    TrueEffect.drop_duplicates(keep='first', inplace=True)

    data = data[data['Method'] != 'TrueTreat']
    print(data.groupby(['Method']).mean())

    return data, TrueEffect


def plot_treat(data, TrueEffect, ax, treat, colors, order, x_label_letter, x_label_number, seed=6, isGwas=True,
               size=9, ymin=-0.5, ymax=1.5):
    data_ = data.set_index('Method').loc[order]
    if isGwas:
        data_ = data_[data_['seed_data'] == seed]
        true = TrueEffect[TrueEffect['seed_data'] == seed][treat].values[0]
        print(true)
    else:
        TrueEffect.reset_index(drop=True, inplace=True)
        true = TrueEffect[treat][0] #* (-1)
    data_.reset_index(drop=False, inplace=True)
    data_[treat] = data_[treat].round(2)
    sns.swarmplot(x="Method", y=treat,
                  hue='type',
                  data=data_,
                  palette=sns.color_palette(colors),
                  ax=ax,
                  size=size)
    ax.set_xticklabels(ax.get_xticklabels())
    ax.legend([], [], frameon=False)
    ax.axhline(y=true, color='r', linestyle='-')
    ax.set(xlabel=x_label_letter + ' τ' + x_label_number, ylabel='')
    ax.axis(ymin=ymin, ymax=ymax)

    return ax


def plot_barplot(ax, data, order, colors, title, ylabel='MAE'):
    sns.barplot(ax=ax, x='Method', y='MAE', palette=sns.color_palette(colors),
                hue='type', dodge=False, order=order, data=data)
    ax.set(xlabel='', ylabel=ylabel)
    ax.set_title(title, fontsize=14)
    ax.legend([], [], frameon=False)
    return ax


def plot_lines(data, listconfigs, ax, colors, x, ymin=0.10, ymax=0.26,
               xlabel='x. bla', ylabel='MAE', addLegend=False):
    cat_type = CategoricalDtype(categories=pd.unique(data[x]).astype(str), ordered=True)
    data[x] = data[x].astype(str)
    data[x] = data[x].astype(cat_type)

    data = data[data['config'].isin(listconfigs)]
    # if len(style_order)==0:
    sns.lineplot(data=data, x=x, y="MAE", ax=ax, hue='Method',
                 style='Method', markers=True, palette=sns.color_palette(colors),
                 legend='brief', markersize=12)
    # else:
    #    sns.lineplot(data=data, x=x, y="MAE", ax=ax, hue='Method',
    #                 style='Method', markers=True, palette=sns.color_palette(colors),
    #                 legend='brief', markersize=12, style_order=style_order)
    ax.axis(ymin=ymin, ymax=ymax)
    ax.set(xlabel=xlabel)
    if addLegend:
        ax.legend(ncol=2)
    else:
        ax.legend([], [], frameon=False)
