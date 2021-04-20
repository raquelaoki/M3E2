import random
import pandas as pd
import numpy as np
import sys
import yaml
import time
from sklearn.model_selection import train_test_split
import torch

sys.path.insert(0, 'src/')
#sys.path.insert(0, 'bartpy/')  # https://github.com/JakeColtman/bartpy
sys.path.insert(0, 'ParKCa/src/')
#from ParKCa.src.train import *
from CompBioAndSimulated_Datasets.simulated_data_multicause import *
import model_m3e2 as m3e2


def main(config_path, seed_models, seed_data):
    """Start: Parameters Loading"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    params = config['parameters']

    # Fix numpy seed for reproducibility
    np.random.seed(seed_models)
    # Fix random seed for reproducibility
    random.seed(seed_models)
    # Fix Torch graph-level seed for reproducibility
    torch.manual_seed(seed_models)

    if 'gwas' in params['data']:

        params_b = {'DA': {'k': [15]},
                    'CEVAE': {'num_epochs': 100, 'batch': 200, 'z_dim': 10}}

        params["n_treatments"] = trykey(params, 'n_treatments', 5)
        prop = params["n_treatments"] / (params["n_treatments"] + params['n_covariates'])

        sdata_gwas = gwas_simulated_data(prop_tc=prop,
                                         pca_path='/content/CompBioAndSimulated_Datasets/data/tgp_pca2.txt',
                                         seed=seed_data,
                                         n_units=params['n_sample'],
                                         n_causes=params["n_treatments"] + params['n_covariates'],
                                         true_causes=params["n_treatments"])
        X, y, y01, treatement_columns, treatment_effects, group = sdata_gwas.generate_samples()
        # Train and Test split use the same seed
        params['baselines'] = trykey(params, 'baselines', False)
        if params['baselines']:
            baselines_results, exp_time, f1_test = baselines(params['baselines_list'], pd.DataFrame(X), y01, params_b,
                                                             TreatCols=treatement_columns, timeit=True,
                                                             seed=seed_models)
        else:
            baselines_results, exp_time, f1_test = baselines(['noise'], pd.DataFrame(X), y01, params_b,
                                                             TreatCols=treatement_columns, timeit=True,
                                                             seed=seed_models)

        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y01, test_size=0.33, random_state=seed_models)
        print('... Target - proportion of 1s', np.sum(y01) / len(y01))
        # Split X1, X2 on GWAS: case with no clinicla variables , X2 = X
        X1_cols = []
        X2_cols = range(X.shape[1] - len(treatement_columns))

        data_nnl = m3e2.data_nn(X_train.values, X_test.values, y_train, y_test, treatement_columns,
                                treatment_effects[treatement_columns], X1_cols, X2_cols)
        loader_train, loader_val, loader_test, num_features = data_nnl.loader(params['suffle'], params['batch_size'],
                                                                              seed_models)
        params['pos_weights'] = data_nnl.treat_weights
        params['pos_weight_y'] = trykey(params, 'pos_weight_y', 1)
        params['hidden1'] = trykey(params, 'hidden1', 64)
        params['hidden2'] = trykey(params, 'hidden2', 8)
        cate_m3e2, f1_test_ = m3e2.fit_nn(loader_train, loader_val, loader_test, params, treatement_columns,
                                          num_features,
                                          X1_cols, X2_cols)
        print('... CATE')
        baselines_results['M3E2'] = cate_m3e2
        exp_time['M3E2'] = time.time() - start_time
        f1_test['M3E2'] = f1_test_
        output = organize_output(baselines_results.copy(), treatment_effects[treatement_columns], exp_time, f1_test)
    if 'copula' in params['data']:
        params_b = {'DA': {'k': [5]},
                    'CEVAE': {'num_epochs': 100, 'batch': 200, 'z_dim': 5}}

        sdata_copula = copula_simulated_data(seed=seed_data, n=params['n_sample'], s=params['n_covariates'])
        X, y, y01, treatement_columns, treatment_effects = sdata_copula.generate_samples()

        if params['baselines']:
            baselines_results, exp_time, f1_test = baselines(params['baselines_list'], pd.DataFrame(X), y01, params_b,
                                                             TreatCols=treatement_columns, timeit=True,
                                                             seed=seed_models)
        else:
            baselines_results, exp_time, f1_test = baselines(['noise'], pd.DataFrame(X), y01, params_b,
                                                             TreatCols=treatement_columns, timeit=True,
                                                             seed=seed_models)
        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y01, test_size=0.33, random_state=seed_models)
        X1_cols = []
        X2_cols = range(X.shape[1] - len(treatement_columns))
        # TODO: add other baselines here to run everything on the same train/testing sets

        data_nnl = m3e2.data_nn(X_train, X_test, y_train, y_test, treatement_columns,
                                treatment_effects, X1_cols, X2_cols)
        loader_train, loader_val, loader_test, num_features = data_nnl.loader(params['suffle'], params['batch_size'],
                                                                              seed_models)
        params['pos_weights'] = data_nnl.treat_weights
        params['pos_weight_y'] = trykey(params, 'pos_weight_y', 1)
        params['hidden1'] = trykey(params, 'hidden1', 6)
        params['hidden2'] = trykey(params, 'hidden2', 6)

        cate_m3e2, f1_test_ = m3e2.fit_nn(loader_train, loader_val, loader_test, params, treatement_columns,
                                          num_features,
                                          X1_cols, X2_cols)
        print('... CATE')
        cate = pd.DataFrame({'CATE_M3E2': cate_m3e2, 'True_Effect': treatment_effects})
        baselines_results['M3E2'] = cate_m3e2
        exp_time['M3E2'] = time.time() - start_time
        f1_test['M3E2'] = f1_test_
        output = organize_output(baselines_results.copy(), treatment_effects[treatement_columns], exp_time, f1_test)
    if 'gwas' not in params['data'] and 'copula' not in params['data']:
        print(
            "ERRROR! \nDataset not recognized. \nChange the parameter data in your config.yaml file to gwas or copula.")

    name = 'output_' + params['data'][0] + '_' + params['id'] + '.csv'
    output['seed_data'] = seed_data
    output['seed_models'] = seed_models

    return output, name


def trykey(params, key, default):
    try:
        return params[key]
    except KeyError:
        params[key] = default
        return params[key]


def baselines(BaselinesList, X, y, ParamsList, seed=63, TreatCols=None, id='', timeit=False):
    """
    input:
        X, colnamesX: potential causes and their names
        Z, colnamesZ: confounders and their names
        y: 01 outcome
        causes: name of the potential causes
    """

    if TreatCols is None:
        TreatCols = list(range(X.shape[1]))

    # check if binary treatments
    X01 = X.copy()
    for col in TreatCols:
        a = X01.iloc[:, col]
        if not ((a == 0) | (a == 1)).all():
            mean_v = np.mean(X01.iloc[:, col])
            X01.iloc[:, col] = [1 if i > mean_v else 0 for i in X01.iloc[:, col]]
        else:
            pass

    X_train, X_test, y_train, y_test, X_train01, X_test01 = train_test_split(X, y, X01,
                                                                             test_size=0.33, random_state=seed)
    coef_table = pd.DataFrame(columns=['causes'])
    coef_table['causes'] = ['T' + str(i) for i in range(len(TreatCols))]
    times, f1_test = {}, {}

    if 'DA' in BaselinesList:
        start_time = time.time()
        from deconfounder import deconfounder_algorithm as DA
        ParamsList['DA']['k'] = trykey(ParamsList['DA'], 'k', 15)  # if exploring multiple latent sizes
        for k in ParamsList['DA']['k']:
            if len(ParamsList['DA']['k']) > 1:
                coln = 'DA_' + str(id) + str(k)
            else:
                coln = 'DA'
            model_da = DA(X_train, X_test, y_train, y_test, k, print_=False)
            ParamsList['DA']['class_weight'] = trykey(ParamsList['DA'], 'class_weight', {0: 1, 1: 1})
            coef, coef_continuos, roc, f1_test['DA'] = model_da.fit(class_weight=ParamsList['DA']['class_weight'])
            coef_table[coln] = coef_continuos[TreatCols]
        times['DA'] = time.time() - start_time
        print('\nDone!')

    if 'BART' in BaselinesList:
        start_time = time.time()
        from bart import BART as BART
        model_bart = BART(X_train01, X_test01, y_train, y_test)
        ParamsList['BART']['n_trees'] = trykey(ParamsList['BART'], 'n_trees', 50)
        ParamsList['BART']['n_burn'] = trykey(ParamsList['BART'], 'n_burn', 100)
        model_bart.fit(n_trees=ParamsList['BART']['n_trees'], n_burn=ParamsList['BART']['n_burn'], print_=False)
        print('...... predictions')
        coef_table['BART'], f1_test['BART'] = model_bart.cate(TreatCols, print_=False)
        times['BART'] = time.time() - start_time
        print('\nDone!')

    if 'CEVAE' in BaselinesList:
        print('\n\n Learner: CEVAE')
        start_time = time.time()
        from cevae import CEVAE as CEVAE
        print('Note: Treatments should be the first columns of X')
        ParamsList['CEVAE']['epochs'] = trykey(ParamsList['CEVAE'], 'epochs', 100)
        ParamsList['CEVAE']['batch'] = trykey(ParamsList['CEVAE'], 'batch', 200)
        ParamsList['CEVAE']['z_dim'] = trykey(ParamsList['CEVAE'], 'z_dim', 5)

        confeatures, binfeatures = [], []
        for col in range(X_train01.shape[1]):
            a = X_train01.iloc[:, col]
            if not ((a == 0) | (a == 1)).all():
                confeatures.append(col)
            else:
                binfeatures.append(col)

        print('... length con and bin features', len(confeatures), len(binfeatures))
        model_cevae = CEVAE(X_train01, X_test01, y_train, y_test, TreatCols,
                            binfeats=binfeatures, contfeats=confeatures,
                            epochs=ParamsList['CEVAE']['epochs'],
                            batch=ParamsList['CEVAE']['batch'],
                            z_dim=ParamsList['CEVAE']['z_dim'])
        coef_table['CEVAE'], f1_test['CEVAE'] = model_cevae.fit_all(print_=False)
        times['CEVAE'] = time.time() - start_time
        print('\nDone!')

    if not timeit:
        return coef_table
    else:
        return coef_table, times, f1_test


def organize_output(experiments, true_effect, exp_time=None, f1_scores=None):
    """
    Important: experiments, experiments times and f1 scores should be in the same order
    Parameters
    ----------
    experiments
    true_effect
    exp_time

    Returns
    -------
    """
    Treatments = experiments['causes']
    experiments.set_index('causes', inplace=True)
    experiments['TrueTreat'] = true_effect
    Treatments_cate = np.transpose(experiments)
    BaselinesNames = experiments.columns
    mae = []
    for col in BaselinesNames:
        dif = np.abs(experiments[col] - experiments['TrueTreat'])
        mae.append(np.nanmean(dif))
    output = pd.DataFrame({'Method': BaselinesNames, 'MAE': mae})
    exp_time['TrueTreat'] = 0
    f1_scores['TrueTreat'] = 0
    if f1_scores is not None:
        output['F1_Test'] = [f1_scores[m] for m in output['Method'].values]
    if exp_time is not None:
        output['Time(s)'] = [exp_time[m] for m in output['Method'].values]

    out = pd.DataFrame(Treatments_cate, columns=Treatments)
    out.reset_index(inplace=True, drop=True)

    return pd.concat((output, out), 1)


colab = False
notebook = True
arg = {'config_path': 'config1.yaml',
       'seed_models': 10,
       'seed_data': 5,
       }
if colab:
    arg['path'] = '/content/'
    arg['config_path'] = arg['path']+arg['config_path']
else:
    arg['path'] = ''

if __name__ == "__main__":
    start_time = time.time()
    if notebook:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
        for j in range(arg['seed_data']):
            print('Data',j)
            for i in range(arg['seed_models']):
                print('Models',i)
                if i == 0 and j == 0:
                    output, name = main(config_path=arg['config_path'], seed_models=i, seed_data=j)
                else:
                    output_, name = main(config_path=arg['config_path'], seed_models=i, seed_data=j)
                    output = pd.concat([output, output_], 0, ignore_index=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
        for j in range(sys.argv[3]):
            print('Data', j)
            for i in range(sys.argv[2]):
                print('Models', i)
                if i == 0:
                    output, name = main(config_path=sys.argv[1], seed_models=i, seed_data=j)
                else:
                    output_, name = main(config_path=sys.argv[1], seed_models=i, seed_data=j)
                    output = pd.concat([output, output_], 0, ignore_index=True)

    output.to_csv(name)
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))
