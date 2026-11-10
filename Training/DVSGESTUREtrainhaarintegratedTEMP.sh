#!/bin/bash

# Conda environment name
CONDA_ENV="thesis"

# Paths
LOG_DIR="../Logs/DVSGESTURE/Training/IntegratedHaarSNN"

# Create log directory if it does not exist 
mkdir -p "$LOG_DIR"

# Initialize Conda for this non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate Conda environment
conda activate "$CONDA_ENV"

# Encodings

ENCDATESTAMP=$(date +"%m%d%y")
ENCTIMESTAMP=$(date +"%H%M")
#  python trainDVSGESTURE.py 0 3 200 -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/IntegratedWaveletSNN/SpikeTrain/20261020_151758_checkpoint_epoch_300.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/IntegratedWaveletSNN/SpikeTrain/20261020_151758_hist_checkpoint_epoch_300.pt \
#      > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_0.log" \
#      2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_0.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 1 3 200 -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/IntegratedWaveletSNN/VoxelGrids/20261021_051317_checkpoint_epoch_300.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/IntegratedWaveletSNN/VoxelGrids/20261021_051317_hist_checkpoint_epoch_300.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_1.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_1.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

# python trainDVSGESTURE.py 2 3 250 -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/IntegratedWaveletSNN/DCT/20261021_185411_checkpoint_epoch_250.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/IntegratedWaveletSNN/DCT/20261021_185411_hist_checkpoint_epoch_250.pt \
#     > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_2.log" \
#     2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_2.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

# python trainDVSGESTURE.py 3 1 25 -g \
#     > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_3.log" \
#     2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_3.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

# python trainDVSGESTURE.py 4 3 200 -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/IntegratedWaveletSNN/VoxelGrids/20261021_051317_checkpoint_epoch_300.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/IntegratedWaveletSNN/VoxelGrids/20261021_051317_hist_checkpoint_epoch_300.pt \
#     > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_4.log" \
#     2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_4.err"

