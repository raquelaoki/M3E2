import random
import pandas as pd
import numpy as np
import sys
import yaml
import time
from sklearn.model_selection import train_test_split
import torch

sys.path.insert(0, 'src/')
sys.path.insert(0, 'bartpy/')  # https://github.com/JakeColtman/bartpy
sys.path.insert(0, 'bartpy/')
sys.path.insert(0, 'ParKCa/src/')
from ParKCa.src.train import *
from CompBioAndSimulated_Datasets.simulated_data_multicause import *
import model_m3e2 as m3e2


# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torch import optim, Tensor


def main(config_path):
    """Start: Parameters Loading"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    params = config['parameters']

    try:
        SEED = params["SEED"]
    except KeyError:
        SEED = 2

    # Fix numpy seed for reproducibility
    np.random.seed(SEED)
    # Fix random seed for reproducibility
    random.seed(SEED)
    # Fix Torch graph-level seed for reproducibility
    torch.manual_seed(SEED)

    if 'gwas' in params['data']:
        sdata_gwas = gwas_simulated_data(prop_tc=0.05,
                                         pca_path='/content/CompBioAndSimulated_Datasets/data/tgp_pca2.txt')
        X, y, y01, treatement_columns, treatment_effects, group = sdata_gwas.generate_samples()
        X_train, X_test, y_train, y_test = train_test_split(X, y01, test_size=0.33, random_state=SEED)
        print('... Target - proportion of 1s', np.sum(y01) / len(y01))
        # Split X1, X2 on GWAS
        X1_cols = []
        X2_cols = range(X.shape[1] - len(treatement_columns))
        # TODO: add other baselines here to run everything on the same train/testing sets

        data_nnl = m3e2.data_nn(X_train.values, X_test.values, y_train, y_test, treatement_columns,
                                treatment_effects[treatement_columns], X1_cols, X2_cols,
                                units_exp=params['units_exp'])
        loader_train, loader_val, loader_test, num_features = data_nnl.loader(params['suffle'], params['batch_size'],
                                                                              SEED)
        params['pos_weights'] = data_nnl.treat_weights
        cate_m3e2 = m3e2.fit_nn(loader_train, loader_val, loader_test, params, treatement_columns, num_features)
        print('... CATE')
        cate = pd.DataFrame({'CATE_M3E2': cate_m3e2, 'True_Effect': treatment_effects[treatement_columns]})
        print(cate)
        dif = cate_m3e2 - treatment_effects[treatement_columns]
        print('MAE', np.abs(dif).mean())

def trykey(params,key,default):
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
        a = X01.iloc[:,col]
        if not ((a == 0) | (a == 1)).all():
            mean_v = np.mean(X01.iloc[:,col])
            X01.iloc[:,col] = [1 if i > mean_v else 0 for i in X01.iloc[:,col]]
        else:
            pass

    X_train, X_test, y_train, y_test, X_train01, X_test01 = train_test_split(X, y, X01,
                                                                             test_size=0.33, random_state=seed)
    coef_table = pd.DataFrame(columns=['causes'])
    coef_table['causes'] = ['T' + str(i) for i in range(len(TreatCols))]
    times = {}

    if 'DA' in BaselinesList:
        print('\n\nBaseline: DA')
        start_time = time.time()
        from deconfounder import deconfounder_algorithm as DA
        ParamsList['DA']['k'] = trykey(ParamsList['DA'],'k',15) # if exploring multiple latent sizes
        for k in ParamsList['DA']['k']:
            if len(ParamsList['DA']['k']) > 1:
                coln = 'DA_' + str(id) + str(k)
            else:
                coln = 'DA'
            model_da = DA(X_train, X_test, y_train, y_test, k)
            ParamsList['DA']['class_weight'] = trykey(ParamsList['DA'], 'class_weight', {0:1,1:1})
            coef, coef_continuos, roc = model_da.fit(class_weight=ParamsList['DA']['class_weight'])
            coef_table[coln] = coef_continuos[0:len(TreatCols)]
        times['DA'] = time.time() - start_time
        print('Done!')

    if 'BART' in BaselinesList:
        print('\n\nLearner: BART')
        start_time = time.time()
        from bart import BART as BART
        model_bart = BART(X_train01, X_test01, y_train, y_test)
        ParamsList['BART']['n_trees'] = trykey(ParamsList['BART'], 'n_trees', 50)
        ParamsList['BART']['n_burn'] = trykey(ParamsList['BART'], 'n_burn', 100)
        model_bart.fit(n_trees=ParamsList['BART']['n_trees'], n_burn=ParamsList['BART']['n_burn'])
        print('...... predictions')
        coef_table['BART'] = model_bart.cate(TreatCols)
        times['BART'] = time.time() - start_time
        print('Done!')

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
        cate = model_cevae.fit_all()
        coef_table['CEVAE'] = cate
        times['CEVAE'] = time.time() - start_time
        print('Done!')

    if not timeit:
        return coef_table
    else:
        return coef_table, times


def organize_output(experiments, true_effect, exp_time):
    experiments['TrueTreat'] = true_effect
    experiments.set_index('causes', inplace=True)
    BaselinesNames = experiments.columns

    mae = []
    for col in BaselinesNames:
        mae.append(np.sum(np.abs(experiments[col] - experiments['TrueTreat'])) / experiments.shape[0])

    output = pd.DataFrame({'Method': BaselinesNames, 'MAE': mae})
    exp_time['TrueTreat'] = 0
    output['Time(s)'] = [exp_time[m] for m in output['Method'].values]
    print(experiments, '\n', output)
    return experiments, output


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
    main(config_path=sys.argv[1])
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))
