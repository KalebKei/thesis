#!/bin/bash

# Conda environment name
CONDA_ENV="thesis"

# Paths
LOG_DIR="../Logs/DVSGESTURE/Training/SNN"

# Create log directory if it does not exist 
mkdir -p "$LOG_DIR"

# Initialize Conda for this non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate Conda environment
conda activate "$CONDA_ENV"

# Encodings

ENCDATESTAMP=$(date +"%m%d%y")
ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 0 0 150 -d -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/SpikeTrain/20260912_005437_checkpoint_epoch_350.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/SpikeTrain/20260912_005437_hist_checkpoint_epoch_350.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_0.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_0.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 1 0 200 -d -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/VoxelGrids/20260912_113659_checkpoint_epoch_275.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/VoxelGrids/20260912_113659_hist_checkpoint_epoch_275.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_1.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_1.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 2 0 250 -d -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/DCT/20260912_233924_checkpoint_epoch_325.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/DCT/20260912_233924_hist_checkpoint_epoch_325.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_2.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_2.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 3 0 350 -d -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/TruncatedDCT/20260913_114409_checkpoint_epoch_275.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/TruncatedDCT/20260913_114409_hist_checkpoint_epoch_275.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_3.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_3.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 4 0 350 -d -g -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/AggressiveDCT/20260913_235255_checkpoint_epoch_326.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/AggressiveDCT/20260913_235255_hist_checkpoint_epoch_326.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_4.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_4.err"

