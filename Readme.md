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

```shell
!python train_models.py config/config_testing/debug.yaml 1 1
```
## Instructions
To train the models, we use the file [train_models.py](train_models.py), which takes three parameters: 
the config files in \textit{yaml} format,
the number of dataset replications, and the number of model replications. 

The folder [config](config/.) contains all the config files adopted by our experiments,
which contain dataset generation parameters, such as dataset name, sample size, covariates size, and the number of treatments. 
The config file also contains model parameters, such as batch size, number of epochs, optimization parameters, size of hidden layers,
type of target and treatment, baselines available to run, and loss weights. The number of datasets and model replications 
are also used as seeds. For example, if the number of dataset replications is 4, we adopt the seeds 1, 2, 3, and 4 
to generate each dataset. 

For our proposed method, M3E2, the user needs to define the following hyperparameters:
* (Required) Number of experts ($num\_exp$): the ideal number depends on the complexity of the treatments and outcome. The current implementation requires the user to test different sizes manually. Most applications active the best results with 4 to 12 experts.
* $expert$: Define the architecture of the experts. The user can pass a $torch.nn.Module$ as an expert. The default option has one linear layer with 4 units. The number of units of the default option can also be changed with the parameter $units\_exp$.
* $X_{low}$ and $X_{high}$: Default option is $X_{high}=X$ and $X_{low}=\{\} $. The default factor model is an autoencoder with two layers of size $hidden1$ (default 64) and $hidden2$ (default 8). 
* $type\_treatment$ and $type\_target$: the implemented options are \textit{binary} and \textit{continuous}. These values define the type of loss function. We assume all treatments have the same type (all binary or all continuous); therefore, only one value needs to be defined.
 * $loss\_target$,$loss\_da$, $loss\_treat$, $loss\_reg$: losses weights for the target/outcome, autoencoder, treatments, and regularization, respectively. The default value is 1. 


As previously mentioned, the user must also define optimization parameters as any other neural network
(learning rate, batch size, number of epochs, etc.). 


## Note
1. If using CEVAE as baseline, the current implementation only supports CPU usage. 


## References

[1] Aoki, Raquel, Yizhou Chen, and Martin Ester. "M3E2: Multi-gate Mixture-of-experts for Multi-treatment Effect 
Estimation." arXiv preprint arXiv:2112.07574 (2021). [link](https://arxiv.org/abs/2112.07574)