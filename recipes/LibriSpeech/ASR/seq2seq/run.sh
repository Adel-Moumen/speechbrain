#!/bin/bash
#SBATCH --nodes=1    
#SBATCH --gpus-per-node=v100l:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G  
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=rrg-ravanelm
#SBATCH --array=0-8%1
#SBATCH --job-name=ls_train   
#SBATCH --output=ls_train_%A_%a.out 
#SBATCH --error=ls_train_%A_%a.err 

# sbatch --job-name=ls_gru --output=ls_gru_%A_%a.out --error=ls_gru_%A_%a.err /home/adelmou/icassp-2024-exploding/speechbrain/recipes/LibriSpeech/ASR/seq2seq/run.sh no_soft_regularisation/gru.yaml results/no_soft_regularisation/gru

# sbatch --job-name=ls_lstm --output=ls_lstm_%A_%a.out --error=ls_lstm_%A_%a.err /home/adelmou/icassp-2024-exploding/speechbrain/recipes/LibriSpeech/ASR/seq2seq/run.sh no_soft_regularisation/lstm.yaml results/no_soft_regularisation/lstm

# Exit if any command fails and if an undefined variable is used
set -eu
# echo of launched commands
set -x
 
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hparams_file> <remove_csv_folder>"
    exit 1
fi
    
# Command-line arguments
HPARAMS_FILE="$1"
REMOVE_CSV_FOLDER="$2"

module load python/3.9.6
source $HOME/icassp/bin/activate

# Copy dataset to SLURM temporary directory
echo "Copying dataset to SLURM_TMPDIR..."
scp -r $HOME/projects/def-ravanelm/datasets/librispeech $SLURM_TMPDIR/
echo "Dataset copied successfully."

cd $SLURM_TMPDIR/

# Extract dataset
echo "Extracting dataset..."
for file in librispeech/*.tar.gz; do
    tar -zxf $file
done
echo "Dataset extracted."

cd $HOME/icassp-2024-exploding/speechbrain/recipes/LibriSpeech/ASR/seq2seq/
# Check if the folder exists, if so, remove the csv files 
# because the path of audio files is not correct anymore
# are we are changing from compute node
if [ -d "$REMOVE_CSV_FOLDER" ]; then
    echo "Removing *.csv files in $REMOVE_CSV_FOLDER..."
    find "$REMOVE_CSV_FOLDER" -name "*.csv" -type f -delete
    echo "Removed successfully."
else
    echo "Folder $REMOVE_CSV_FOLDER does not exist. Skipping removal."
fi

# Launch the training script using distributed training
echo "Starting training..."
python -m torch.distributed.launch --nproc_per_node=2 train.py "hparams/$HPARAMS_FILE"  --data_folder=$SLURM_TMPDIR/LibriSpeech --distributed_launch
echo "Training completed."
