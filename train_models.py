import logging
import numpy as np
import pandas as pd
import random
import sys
import time
import torch
import yaml

from sklearn.model_selection import train_test_split

# Local Imports
from CompBioAndSimulated_Datasets.simulated_data_multicause import *
import model_m3e2 as m3e2
from resources.cevae import CEVAE as CEVAE
from resources.deconfounder import deconfounder_algorithm as DA
from resources.dragonnet import dragonnet

logger = logging.getLogger(__name__)


def main(config_path, seed_models, seed_data, path_save_model=''):
    """Start: Parameters Loading"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    params = config['parameters']

    # # Fix numpy seed for reproducibility
    # np.random.seed(seed_models)
    # # Fix random seed for reproducibility
    # random.seed(seed_models)
    # # Fix Torch graph-level seed for reproducibility
    # torch.manual_seed(seed_models)

    if params['is_binary_treatment']:
        params['pos_weight_t'] = np.repeat(1, params['n_treatments'])
    else:
        params['pos_weight_t'] = params['pos_weights']

    params['baselines'] = params.get('baselines', False)
    if 'gwas' in params['data']:
        # params = {'DA': {'k': [15]},
        #             'CEVAE': {'num_epochs': 100, 'batch': 200, 'z_dim': 10, 'binarytarget': True},
        #             'Dragonnet': {'u1': 200, 'u2': 100, 'u3': 1},
        #             'HiCI': {'batch': 250, 'type_target': 'binary', 'gamma': 0.05,
        #                      'hidden1': 64, 'hidden2': 32, 'loss_weight': [5, 0.05, 0.3]}}
        params["n_treatments"] = params.get('n_treatments', 5)
        prop = params["n_treatments"] / (params["n_treatments"] + params['n_covariates'])

        sdata_gwas = gwas_simulated_data(prop_tc=prop,
                                         seed=seed_data,
                                         n_units=params['n_sample'],
                                         n_causes=params["n_treatments"] + params['n_covariates'],
                                         true_causes=params["n_treatments"])
        X, y, y01, treatement_columns, true_effect, group = sdata_gwas.generate_samples()
        # Train and Test split use the same seed
        if params['baselines']:
            results, experiment_time, score = baselines(BaselinesList=params['baselines_list'],
                                                        X=pd.DataFrame(X),
                                                        y=y01,
                                                        ParamsList=params,
                                                        TreatCols=treatement_columns,
                                                        timeit=True,
                                                        seed=seed_models)
        else:
            results, experiment_time, score = baselines(['noise'],
                                                        TreatCols=treatement_columns, timeit=True,
                                                        seed=seed_models)


        # Split X1, X2 on GWAS: case with no clinicla variables , X2 = X
        Xlow_cols = []
        Xhigh_cols = range(X.shape[1] - len(treatement_columns))
        results = proposed_method(X=X,
                                 y=y,
                                 ParamsList=params,
                                 seed=seed_models,
                                 TreatCols=treatement_columns,
                                 true_effect=true_effect,
                                 Xlow_cols=Xlow_cols,
                                 Xhigh_cols=Xhigh_cols,
                                 results=results)
    elif 'copula' in params['data']:
        # params = {'DA': {'k': [5]},
        #             'CEVAE': {'num_epochs': 100, 'batch': 200, 'z_dim': 5, 'binarytarget': True},
        #             'Dragonnet': {'u1': 10, 'u2': 5, 'u3': 1},
        #             'HiCI': {'batch': 250, 'type_target': 'continous', 'gamma': 0.1,
        #                      'hidden1': 10, 'hidden2': 4, 'loss_weight': [1, 0.2, 0.01]}
        #             }

        sdata_copula = copula_simulated_data(seed=seed_data, n=params['n_sample'], s=params['n_covariates'])
        X, y, y01, treatement_columns, true_effect = sdata_copula.generate_samples()

        if params['baselines']:
            results, experiment_time, score = baselines(BaselinesList=params['baselines_list'],
                                                        X=pd.DataFrame(X),
                                                        y=y01,
                                                        ParamsList=params,
                                                        TreatCols=treatement_columns,
                                                        timeit=True,
                                                        seed=seed_models)
        else:
            results, experiment_time, score = baselines(['noise'],
                                                        TreatCols=treatement_columns, timeit=True,
                                                        seed=seed_models)

        Xlow_cols = []
        Xhigh_cols = range(X.shape[1] - len(treatement_columns))
        results = proposed_method(X=X,
                                 y=y,
                                 ParamsList=params,
                                 seed=seed_models,
                                 TreatCols=treatement_columns,
                                 true_effect=true_effect,
                                 Xlow_cols=Xlow_cols,
                                 Xhigh_cols=Xhigh_cols,
                                 results=results)
    elif 'ihdp' in params['data']:
        sdata_ihdp = ihdp_data(id=seed_data)
        X, y, treatement_columns, true_effect = sdata_ihdp.generate_samples()

        # params = {'DA': {'k': [5]},
        #             'CEVAE': {'num_epochs': 150, 'batch': 50, 'z_dim': 5, 'binarytarget': False},
        #             'Dragonnet': {'u1': 200, 'u2': 100, 'u3': 1}}  # same as paper

        if params['baselines']:
            results, experiment_time, score = baselines(BaselinesList=params['baselines_list'],
                                                        X=X,
                                                        y=y,
                                                        ParamsList=params,
                                                        TreatCols=treatement_columns,
                                                        timeit=True,
                                                        seed=seed_models)
        else:
            results, experiment_time, score = baselines(['noise'],
                                                        TreatCols=treatement_columns, timeit=True,
                                                        seed=seed_models)

        Xlow_cols = range(X.shape[1] - len(treatement_columns))  # []
        Xhigh_cols = None  # []#range(X.shape[1] - len(treatement_columns))
        results = proposed_method(X=X,
                                 y=y,
                                 ParamsList=params,
                                 seed=seed_models,
                                 TreatCols=treatement_columns,
                                 true_effect=true_effect,
                                 Xlow_cols=Xlow_cols,
                                 Xhigh_cols=Xhigh_cols,
                                 results=results)

    if 'gwas' not in params['data'] and 'copula' not in params['data'] and 'ihdp' not in params[
        'data'] and 'bcch' not in params['data']:
        logger.debug(
            "ERRROR! \nDataset not recognized. \nChange the parameter data in your config.yaml file to gwas or copula.")

    filename = 'output_' + params['data'][0] + '_' + params['id'] + '.csv'
    results['seed_data'] = seed_data
    results['seed_models'] = seed_models

    return results, filename


def proposed_method(X, y, ParamsList, seed, results,
                    TreatCols=None, true_effect=None,
                    Xlow_cols=None, Xhigh_cols=None):
    start_time = time.time()
    print('...Running M3E2')
    # Fix numpy seed for reproducibility
    np.random.seed(seed)
    # Fix random seed for reproducibility
    random.seed(seed)
    # Fix Torch graph-level seed for reproducibility
    torch.manual_seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed_models)

    data_nnl = m3e2.data_nn(X_train=X_train.values,
                            X_test=X_test.values,
                            y_train=y_train,
                            y_test=y_test,
                            treatments_columns=TreatCols,
                            treatment_effec=true_effect,
                            Xlow_cols=Xlow_cols,
                            Xhigh_cols=Xhigh_cols)
    loader_train, loader_val, loader_test, num_features = data_nnl.loader(shuffle=ParamsList['suffle'],
                                                                          batch=ParamsList['batch_size'],
                                                                          seed=seed)
    ParamsList['hidden1'] = ParamsList.get('hidden1', 6)
    ParamsList['hidden2'] = ParamsList.get('hidden2', 6)
    params['pos_weight_y'] = params.get('pos_weight_y', 1)
    ParamsList['pos_weights'] = np.repeat(ParamsList['pos_weights'], len(TreatCols))
    ate_m3e2, score_ = m3e2.fit_nn(loader_train=loader_train,
                                   loader_val=loader_val,
                                   loader_test=loader_test,
                                   params=ParamsList,
                                   treatement_columns=TreatCols,
                                   Xlow_cols=Xlow_cols,
                                   Xhigh_cols=Xhigh_cols,
                                   use_bias_y=True)
    logger.debug('... ate')
    results['M3E2'] = ate_m3e2
    experiment_time['M3E2'] = time.time() - start_time
    score['M3E2'] = score_
    output = organize_output(output_table=results.copy(),
                             true_effect=true_effect,
                             experiment_time=experiment_time,
                             score=score,
                             gwas=False)
    return output


def baselines(BaselinesList, X, y, ParamsList, seed=63, TreatCols=None, timeit=True):
    """
    input:
        X, colnamesX: potential causes and their names
        Z, colnamesZ: confounders and their names
        y: 01 outcome
        causes: name of the potential causes
    """

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
    results_table = pd.DataFrame(columns=['causes'])
    results_table['causes'] = ['T' + str(i) for i in range(len(TreatCols))]
    times, score = {}, {}

    if 'DA' in BaselinesList:
        # Fix numpy seed for reproducibility
        np.random.seed(seed)
        # Fix random seed for reproducibility
        random.seed(seed)
        # Fix Torch graph-level seed for reproducibility
        torch.manual_seed(seed)
        start_time = time.time()
        print('...Running DA')
        ParamsList['da_k'] = ParamsList.get('da_k', 15)  # if exploring multiple latent sizes
        for k in ParamsList['da_k']:
            if len(ParamsList['da_k']) > 1:
                coln = 'DA_' + str(k)
            else:
                coln = 'DA'
            model_da = DA(X_train, X_test, y_train, y_test, k, print_=False)
            ParamsList['da_class_weight'] = ParamsList.get('da_class_weight', {0: 1, 1: 1})
            coef, coef_continuos, roc, score['DA'] = model_da.fit(class_weight=ParamsList['da_class_weight'])
            results_table[coln] = coef_continuos[TreatCols]
        times['DA'] = time.time() - start_time
        logger.debug('\nDone!')

    if 'Dragonnet' in BaselinesList:
        # Fix numpy seed for reproducibility
        np.random.seed(seed)
        # Fix random seed for reproducibility
        random.seed(seed)
        # Fix Torch graph-level seed for reproducibility
        torch.manual_seed(seed)
        start_time = time.time()
        print('...Running Dragonnet')
        model_dragon = dragonnet(X_train=X_train01,
                                           X_test=X_test01,
                                           y_train=y_train,
                                           y_test=y_test,
                                           treatments_columns=TreatCols)
        model_dragon.fit_all(is_targeted_regularization=False,
                             u1=ParamsList['Dragonnet_u1'],
                             u2=ParamsList['Dragonnet_u2'],
                             u3=ParamsList['Dragonnet_u3'],
                             epochs_adam=ParamsList['max_epochs'],
                             epochs_sgd=ParamsList['max_epochs'],
                             batch_size=ParamsList['batch_size']
                             )
        ate = model_dragon.ate()
        results_table['Dragonnet'], score['Dragonnet'] = ate[0], model_dragon.score
        times['Dragonnet'] = time.time() - start_time
        logger.debug('\nDone!')

    if 'CEVAE' in BaselinesList:
        # Fix numpy seed for reproducibility
        np.random.seed(seed)
        # Fix random seed for reproducibility
        random.seed(seed)
        # Fix Torch graph-level seed for reproducibility
        torch.manual_seed(seed)
        start_time = time.time()
        print('...Running CEVAE')
        logger.debug('Note: Treatments should be the first columns of X')
        ParamsList['CEVAE_z_dim'] = ParamsList.get('CEVAE_z_dim', 5)

        confeatures, binfeatures = [], []
        for col in range(X_train01.shape[1]):
            a = X_train01.iloc[:, col]
            if not ((a == 0) | (a == 1)).all():
                confeatures.append(col)
            else:
                binfeatures.append(col)

        logger.debug('... length con and bin features', len(confeatures), len(binfeatures))
        model_cevae = CEVAE(X_train=X_train01,
                            X_test=X_test01,
                            y_train=y_train,
                            y_test=y_test,
                            treatments_columns=TreatCols,
                            binfeats=binfeatures,
                            contfeats=confeatures,
                            epochs=ParamsList['max_epochs'],
                            batch=ParamsList['batch_size'],
                            z_dim=ParamsList['CEVAE_z_dim'],
                            binarytarget=ParamsList['is_binary_target'])
        logger.debug('DONE INITIALIZATION')
        results_table['CEVAE'], score['CEVAE'] = model_cevae.fit_all(print_=False)
        times['CEVAE'] = time.time() - start_time
        logger.debug('\nDone!')

    if 'noise' in BaselinesList:
        results_table['noise'] = np.repeat(0, len(treatement_columns))
        score['noise'] = np.repeat(0, len(treatement_columns))

    else:
        raise NotImplementedError('This method is not supported!')

    if not timeit:
        return results_table
    else:
        return results_table, times, score


def organize_output(output_table, true_effect, experiment_time=None, score=None):
    """
    Important: output_table, output_table times and f1 scores should be in the same order
    Parameters
    ----------
    score
    output_table
    true_effect
    experiment_time

    Returns
    -------
    """
    Treatments = output_table['causes']
    output_table.set_index('causes', inplace=True)
    output_table['TrueTreat'] = true_effect

    Treatments_ate = np.transpose(output_table)
    BaselinesNames = output_table.columns
    mae = []
    for col in BaselinesNames:
        dif = np.abs(output_table[col] - output_table['TrueTreat'])
        mae.append(np.nanmean(dif))
    output = pd.DataFrame({'Method': BaselinesNames, 'MAE': mae})
    experiment_time['TrueTreat'] = 0
    score['TrueTreat'] = 0
    if score is not None:
        output['score'] = [score[m] for m in output['Method'].values]
    if experiment_time is not None:
        output['Time(s)'] = [experiment_time[m] for m in output['Method'].values]

    out = pd.DataFrame(Treatments_ate, columns=Treatments)
    out.reset_index(inplace=True, drop=True)

    return pd.concat((output, out), 1)


colab = False
notebook = False
arg = {'config_path': 'config2.yaml',
       'seed_models': 3,
       'seed_data': 1,
       }
if colab:
    arg['path'] = '/content/'
    arg['config_path'] = arg['path'] + arg['config_path']
else:
    arg['path'] = ''

if __name__ == "__main__":
    start_time = time.time()
    if notebook:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
        for j in range(arg['seed_data']):
            print('Data', j)
            for i in range(arg['seed_models']):
                print('Models', i)
                if i == 0 and j == 0:
                    output, name = main(config_path=arg['config_path'], seed_models=i, seed_data=j)
                else:
                    output_, name = main(config_path=arg['config_path'], seed_models=i, seed_data=j)
                    output = pd.concat([output, output_], 0, ignore_index=True)
        output_path = arg['path']
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
        for j in range(int(sys.argv[3])):
            print('Data', j)
            output = pd.DataFrame()
            for i in range(int(sys.argv[2])):
                print('Models', i)
                #if i == 0 and j == 0:
                #    output, name = main(config_path=sys.argv[1], seed_models=i, seed_data=j)
                #else:
                output_, filename = main(config_path=sys.argv[1], seed_models=i, seed_data=j)
                output = pd.concat([output, output_], 0, ignore_index=True)
        output_path = 'output/'

    output.to_csv(output_path + filename)
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))
