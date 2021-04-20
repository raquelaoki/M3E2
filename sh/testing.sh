#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=rrg-ester
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024M


echo 'Starting ....'
echo 'not using #SBATCH --gres=gpu:p100:1'
echo 'not using #SBATCH --output=slurm/%x-%j.out'

module load python/3.7 cuda cudnn

SOURCEDIR= ~/projects/rrg-ester/raoki/M3E2

source env/bin/activate
python testing_slurm.py

echo 'SUCESS, NOW LOOK FOR FILE'

