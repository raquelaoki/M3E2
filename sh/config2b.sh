#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:1

echo 'Starting ....'

module load --ignore-cache python/3.7 cuda cudnn

SOURCEDIR= ~/M3E2

source env/bin/activate

python train_models.py config/config2b.yaml 8 4

echo 'DONE!'

