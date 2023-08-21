#!/bin/bash
#SBATCH --nodes=1    
#SBATCH --gpus-per-node=v100l:2
#SBATCH --cpus-per-task=30
#SBATCH --mem=20G  
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=cv_train   
#SBATCH --output=cv_train_%A_%a.out 
#SBATCH --error=cv_train_%A_%a.err 

# sbatch --job-name=cv_gru --output=cv_gru_%A_%a.out --error=cv_gru_%A_%a.err /home/adelmou/icassp-2024-exploding/speechbrain/recipes/CommonVoice/ASR/seq2seq/run.sh no_soft_regularisation/gru.yaml results/no_soft_regularisation/gru

# sbatch --job-name=cv_lstm --output=cv_lstm_%A_%a.out --error=cv_lstm_%A_%a.err /home/adelmou/icassp-2024-exploding/speechbrain/recipes/CommonVoice/ASR/seq2seq/run.sh no_soft_regularisation/lstm.yaml results/no_soft_regularisation/lstm

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

module purge 

module load python/3.9.6
source $HOME/icassp/bin/activate

module load ffmpeg 

# Copy dataset to SLURM temporary directory
echo "Copying dataset to SLURM_TMPDIR..."
scp -r $HOME/projects/def-ravanelm/datasets/common_voice_14.0/cv-corpus-14.0-2023-06-23-fr.tar.gz $SLURM_TMPDIR/
echo "Dataset copied successfully."

cd $SLURM_TMPDIR/

# Extract dataset
echo "Extracting dataset..."
tar -zxf cv-corpus-14.0-2023-06-23-fr.tar.gz
echo "Dataset extracted."

echo "Resampling + formatting..."
python $HOME/icassp-2024-exploding/speechbrain/recipes/CommonVoice/ASR/seq2seq/resample.py --input_folder=cv-corpus-14.0-2023-06-23/fr/clips/ --output_folder=cv-corpus-14.0-2023-06-23/fr/clips-v2/ --output_sr=16000 --file_ext="mp3" --n_jobs=29
echo "Done..."


cd $HOME/icassp-2024-exploding/speechbrain/recipes/CommonVoice/ASR/seq2seq/
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
python -m torch.distributed.launch --nproc_per_node=2 train.py "hparams/$HPARAMS_FILE"  --data_folder=$SLURM_TMPDIR/cv-corpus-14.0-2023-06-23/fr --distributed_launch --debug
echo "Training completed."
