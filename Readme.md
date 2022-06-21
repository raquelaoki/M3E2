# M3E2: Multi-gate Mixture-of-experts for Multi-treatment Effect Estimation

Author: Raquel Aoki
Date: 2022/06/21

M3E2 is an estimator for multiple treatments. Hence, instead of a single treetment T, there is a set of treatments 
$\mathcal{T} = \{T_0, T_1,...,T_K\}$.  

This repository contains the implementation of the baselines adopted in the paper. Some of these baselines were designed 
for single treatment applications. Please check the paper for more details about the methods, adaption, and results [1]. 


### Baselines implemented: 
- Dragonnet
- CEVAE
- Deconfounder Algorithm


We tried to use the implementation available of these methods is possible. 
In some cases, we created a fork with the adoption required for the 
multi-treatment setting.


## File structure: 
* [model_m3e2.py](model_m3e2.py): Pytorch implementation of our proposed method; 
* [train_models.py](train_models.py): Run the experiments (M3E2 and baselines)
* [resources/.](resources): Code associated to our baselines;
* [M3E2_plots.ipynb](M3E2_plots.ipynb): make the plots;


## Running Experiments

2. To run locally or on Colab, please check TODO.

## Note: 
1. If using CEVAE as baseline, the current implementation only supports CPU usage. 



## References

[1] Aoki, Raquel, Yizhou Chen, and Martin Ester. "M3E2: Multi-gate Mixture-of-experts for Multi-treatment Effect 
Estimation." arXiv preprint arXiv:2112.07574 (2021). [link](https://arxiv.org/abs/2112.07574)