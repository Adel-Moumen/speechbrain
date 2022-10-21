#!/bin/sh

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_BNCI2014001 # name of the orion experiment
OUTPUT_FOLDER=$2 # results/EEGNet_BNCI2014001_seed_variability_moabb
HPOPT_CONFIG_FILE=hparams/orion/hparams_tpe.yaml   #hparam file for orion

export _MNE_FAKE_HOME_DIR='/network/scratch/r/ravanelm/' # change with your own folder (needed for mne)
export ORION_DB_ADDRESS=/network/scratch/r/ravanelm/tpe_EEGNet_BNCI2014001.pkl # This is the database where orion will save the results
export ORION_DB_TYPE=PickledDB

# Running orion
cd ../..

orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=250  \
	./run_experiments_seed_variability.sh hparams/EEGNet_BNCI2014001_seed_variability.yaml \
	/localscratch/eeg_data $OUTPUT_FOLDER 1 1 'random_seed' 1 acc valid_metrics.pkl false true \
<<<<<<< HEAD
  --number_of_epochs~"uniform(100, 1000,discrete=True)" \
	--avg_models~"uniform(1, 15,discrete=True)" \
=======
	--number_of_epochs~"uniform(100, 1000,discrete=True)" \
	--avg_models~"uniform(1, 20,discrete=True)" \
>>>>>>> 8628ab5a5f85fb9e865503d71ee62f5cb879f492
	--step_size~"choices([50,100,150])" \
	--batch_size~"choices([32,64,128])" \
	--lr~"choices([0.01,0.005, 0.001,0.0005, 0.0001])" \
	--tmax~"uniform(1.0, 4.0)" \
	--fmin~"uniform(0.1, 5)" \
	--fmax~"uniform(20, 50)" \
	--cnn_temporal_kernels~"uniform(4, 64,discrete=True)" \
	--cnn_temporal_kernelsize~"uniform(8, 64,discrete=True)" \
	--cnn_spatial_depth_multiplier~"uniform(1, 4,discrete=True)" \
	--cnn_spatial_pool~"uniform(1, 4,discrete=True)" \
	--cnn_septemporal_depth_multiplier~"uniform(1, 4,discrete=True)" \
	--cnn_septemporal_kernelsize~"uniform(4, 32,discrete=True)" \
	--cnn_septemporal_pool~"uniform(1, 8,discrete=True)" \
	--dropout~"uniform(0.0, 0.50)" \
<<<<<<< HEAD
	--snr_low~"uniform(0.0, 15.0)" \
	--snr_high~"uniform(15.0, 30.0)" \
=======
	--snr_pink_low~"uniform(0.0, 15.0)" \
	--snr_pink_high~"uniform(15.0, 30.0)" \
	--snr_white_low~"uniform(0.0, 15.0)" \
	--snr_white_high~"uniform(15.0, 30.0)" \
	--snr_muscular_low~"uniform(0.0, 15.0)" \
	--snr_muscular_high~"uniform(15.0, 30.0)" \
>>>>>>> 8628ab5a5f85fb9e865503d71ee62f5cb879f492
	--repeat_augment~"uniform(1, 3,discrete=True)" \
	--n_augmentations~"uniform(0, 10,discrete=True)"


