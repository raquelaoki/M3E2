#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-ester
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024M

echo 'Starting ....'
echo 'not using #SBATCH --gres=gpu:p100:1'

module load python/3.8 cuda cudnn

SOURCEDIR= $HOME/M3E2

source $HOME/M3E2/env/bin/activate

python testing_slurm.py

echo 'SUCESS, NOW LOOK FOR FILE'

#SBATCH -o /slurm/output.%a.out