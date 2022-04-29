# M3E2

Implementation of _M3E2: Multi-gate Mixture-of-experts for
Multi-treatment Effect Estimation_.

## File structure: 
* [model_m3e2.py](model_m3e2.py): Pytorch implementation of our proposed method; 
* [train_models.py](train_models.py): Run the experiments (M3E2 and baselines)
* [resources/.](resources): Code associated to our baselines;
* [sh/.](sh) and [config/.](config): Contain the settings explored;
* [output./](output): csv files with experimental results;
* [M3E2_plots.ipynb](M3E2_plots.ipynb): make the plots;
* [plots/.](plots): png images from experimental results;


## Running Experiments

1. Check [here](ComputeCanada.md) for instructions on how to use a cluster to run and submit all the experiments;
2. To run locally or on Colab, please check TODO. 

## Intructions: 
1. IF using CEVAE as baseline, the current implementation only supports 
## TODO: 
3. Add small unit test
4. Add unit test