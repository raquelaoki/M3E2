import random
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, 'src/')
sys.path.insert(0, 'bartpy/')  # https://github.com/JakeColtman/bartpy
from model_m3e2 import *
from sklearn.model_selection import train_test_split
import torch
sys.path.insert(0, 'bartpy/')
sys.path.insert(0, 'ParKCa/src/')
from ParKCa.src.train import *
from CompBioAndSimulated_Datasets.simulated_data_multicause import *
import model_m3e2 as m3e2
import yaml
import time
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader, TensorDataset
#from torch import optim, Tensor


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
        sdata_gwas = gwas_simulated_data(prop_tc=0.05, pca_path='/content/CompBioAndSimulated_Datasets/data/tgp_pca2.txt')
        X, y, y01, treatement_columns, treatment_effects, group = sdata_gwas.generate_samples()
        X_train, X_test, y_train, y_test = train_test_split(X, y01, test_size=0.33, random_state=SEED)
        # Split X1, X2 on GWAS
        X1_cols = []
        X2_cols = range(X.shape[1]-len(treatement_columns))
        # TODO: add other baselines here to run everything on the same train/testing sets

        data_nnl = m3e2.data_nn(X_train.values, X_test.values, y_train, y_test, treatement_columns,
                                treatment_effects[treatement_columns], X1_cols, X2_cols)
        loader_train, loader_val, loader_test, num_features = data_nnl.loader(params['suffle'],params['batch_size'],SEED)
        cate_test = m3e2.fit_nn(loader_train, loader_val, loader_test,params,treatement_columns, num_features)
        print(cate_test, treatment_effects[treatement_columns])


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
    main(config_path=sys.argv[1])
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))

