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

    #dataset
    gwas_data = gwas_simulated_data(1000, 100, 8, prop_tc = 0.05)
    y, tc, X, col = gwas_data.generate_samples()
    X = pd.DataFrame(X).sample(frac=1.0).values
    y = y.astype('float')


#todo: install torch and setting for gpu
#todo: python data




if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
    main(config_path=sys.argv[1])
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))
