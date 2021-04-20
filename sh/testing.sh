#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-ester
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024M

echo 'Starting ....'

module load python/3.8 cuda cudnn

SOURCEDIR=/M3E2

source /env/bin/activate

python testing_slurm.py

