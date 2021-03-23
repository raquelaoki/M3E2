import random
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'src/')
sys.path.insert(0,'bartpy/') #https://github.com/JakeColtman/bartpy
from data_simulation import *
from model_m3e2 import *

from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

def main(config_path):
    """Start: Parameters Loading"""
    with open(config_path) as f:
        config = yaml.load_all(f, Loader=yaml.FullLoader)
        for p in config:
            params = p["parameters"]

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


    loader_train, loader_val, loader_test, n_treat , y_continuous = data_gwas()
    X, y , T = next(iter(train_loader))
    model = M3E2(data = 'gwas', num_treat = n_treat, num_exp = 5 )
    criterion = nn.BCEWithLogitsLoss()
    #criterion = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params["cw_census"][i]).to(device)) for i in range(num_tasks)]
    lr = 1e-4 #Learning Rate
    wd = 0.01
    if torch.cuda.is_available():
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params["gamma"])

    loss_ = []
    best_val_AUC = 0
    best_epoch = 0


#todo: install torch and setting for gpu

def data_gwas(params = None):
    params['n_sample'] = 1000
    params['n_covariates'] = 100
    params["shuffle"] = True
    params["batch_size"] = 250
    SEED = 1
    #dataset
    gwas_data = gwas_simulated_data(params['n_sample'] , params['n_covariates'], SEED, prop_tc = 0.05)
    y, tc, X, col = gwas_data.generate_samples()
    X = pd.DataFrame(X).sample(frac=1.0).values
    T = X[:,col]
    X1 = np.delete(X,col,1)
    print('\nTotal: ',X.shape,'\nCovariates: ', X1.shape, '\nTreatments: ',T.shape)
    #X = pd.DataFrame(X).sample(frac=1.0).values
    #y = y.astype('float')
    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(X1, y, T, test_size=0.33, random_state=SEED)
    X_val, X_test, y_val, y_test, T_val, T_test = train_test_split(X_test, y_test, T_test, test_size=0.5, random_state=SEED)

    ''' Creating TensorDataset to use in the DataLoader '''
    dataset_train = TensorDataset(Tensor(X_train), Tensor(y_train),Tensor(T_train))
    dataset_test = TensorDataset(Tensor(X_test), Tensor(y_test),Tensor(T_test))
    dataset_val = TensorDataset(Tensor(X_val), Tensor(y_val),Tensor(T_val))

    ''' Required: Create DataLoader for training the models '''
    loader_train = DataLoader(dataset_train, shuffle=params["shuffle"], batch_size=params["batch_size"])
    loader_val = DataLoader(dataset_validation,shuffle=params["shuffle"],batch_size=validation_data.shape[0])
    loader_test = DataLoader(dataset_test, shuffle=False, batch_size=test_data.shape[0])

    y_continuous = False
    n_treat = T_test.shape[1]
    return loader_train, loader_val, loader_test, n_treat , y_continuous


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
    main(config_path=sys.argv[1])
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))
