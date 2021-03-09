# M3E2
Spring 21

Virtualenv 
```bash
module load python/3.6
virtualenv env
source env/bin/activate
pip install jupyter
echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter notebook --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/notebook.sh
chmod u+x $VIRTUAL_ENV/bin/notebook.sh


pip install jupyterlmod
jupyter nbextension install --py jupyterlmod --sys-prefix #dont work
jupyter nbextension enable --py jupyterlmod --sys-prefix
jupyter serverextension enable --py jupyterlmod --sys-prefix

salloc --time=1:0:0 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=1024M --account=def-ester srun $VIRTUAL_ENV/bin/notebook.sh

```

