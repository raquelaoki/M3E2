#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000M
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:1

echo 'Starting ....'

module load --ignore-cache python/3.7 cuda cudnn

SOURCEDIR= ~/M3E2

source env/bin/activate

python train_models.py config/config2i.yaml 8 4

echo 'DONE!'

