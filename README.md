# M3E2
This project was developed using Compute Canada.

## Virtualenv on Compute Canada
Setting up the virtual env ([reference](https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment))
```bash
module load python/3.7

virtualenv --no-download env
source env/bin/activate

#It takes some time ~1h
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
salloc --time=1:0:0 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=1024M --account=def-ester srun $VIRTUAL_ENV/bin/notebook.sh
```
3. In your local machine, run: 
```bash
ssh -L 8888:cdr767.int.cedar.computecanada.ca:8888 USER@cedar.computecanada.ca
```
Note: ports might differ, check before running if you are using the correct ones. 
4. Open the link shown on Compute Canada on your local browser.

TODO: module load cuda torch (test if required)

## Datasets
- Simulated datasets are available [here](https://github.com/raquelaoki/CompBioAndSimulated_Datasets)

## Baselines
- The Deconfouder Algorithm, BART, and CEVAE were previously implemented [here](https://github.com/raquelaoki/ParKCa). We are using the same implementation. 
- Dragonned and Hi-CI: TODO
