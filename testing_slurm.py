import random
import pandas as pd
import numpy as np
import sys
import yaml
import time
import torch


def main(config_path, seed_models, seed_data):
    """Start: Parameters Loading"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    params = config['parameters']

    output = params
    output['new_entrey'] = 'THIS IS NEW'

    return output

colab = True
arg = {'config_path': '/content/config2.yaml',
       'seed_models': 10,
       'seed_data': 5,
       }

if __name__ == "__main__":
    start_time = time.time()
    if colab:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
        name = 'output/NEW_SLRUM_TEST_FILE.csv'
        output = pd.DataFrame({'a':[1,2], 'b':[3,4]})
        output.to_csv(name)
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    import torch
    from torch import Tensor

    print('Available devices ', torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())
    cuda = torch.device(0)
    a = [0, 1, 2]
    b = Tensor(a).cuda()
    print(a, b)
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))
