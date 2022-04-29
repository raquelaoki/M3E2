# Compute Canada  

This document explains how to setup Compute Canada env and
run experiments. 


## Virtualenv on Compute Canada
Setting up the virtual env ([reference](https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment))
```bash
module load python/3.7

virtualenv --no-download env
source env/bin/activate

#It takes some time ~3h
pip install --no-index -r requirements.txt

# For interactive jupyter notebook:
echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter notebook --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/notebook.sh
chmod u+x $VIRTUAL_ENV/bin/notebook.sh
jupyter nbextension install --py jupyterlmod --sys-prefix #dont work
jupyter nbextension enable --py jupyterlmod --sys-prefix
jupyter serverextension enable --py jupyterlmod --sys-prefix
```
Running an interactive section of jupyter notebook:

1. To go your project's folder on Compute Canada.
2. Run:
```bash
source env/bin/activate
salloc --time=1:0:0 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=4000M --account=rrg-ester srun $VIRTUAL_ENV/bin/notebook.sh
```
3. In your local machine, run (example, ports might change): 
```bash
ssh -L 8888:cdr767.int.cedar.computecanada.ca:8888 USER@cedar.computecanada.ca
```
Note: ports might differ, check before running if you are using the correct ones. 

4. Open the link shown on Compute Canada on your local browser.

## Runing Experiments

* Inside sh/ there are several .sh files. These files control the number of dataset repetitions and model repetitions. 
* For each .sh file, there is a .yaml file inside the config/ folder. 

One could either submit the .sh file to a cluster, or run locally: 
```shell
python train_models.py config/config1.yaml 8 4
```
where 8 would be the number of model repetitions, and 4 the number of datasets repettions (total of 4 x 8 runs of settings inside config1.yaml)