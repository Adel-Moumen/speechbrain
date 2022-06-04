#!/bin/bash
#SBATCH --job-name=CV-it-LiGRU-v100
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_3090:2
#SBATCH --mem=12G
#SBATCH --time=192:00:00

source /etc/profile.d/conda.sh
conda activate ligru 

srun python3 /users/amoumen/speechbrain/recipes/CommonVoice/ASR/seq2seq/train.py /users/amoumen/speechbrain/recipes/CommonVoice/ASR/seq2seq/exps/ligru_vanilla_train_it.yaml --data_folder=/local_disk/idyie/amoumen/cv-corpus-8.0-2022-01-19/it
