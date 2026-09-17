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

python trainDVSGESTURE.py 0 0 100 -d -g -vp 0.95 -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/SpikeTrain/20260914_203216_checkpoint_epoch_500.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/SpikeTrain/20260914_203216_hist_checkpoint_epoch_500.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_0.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_0.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 1 0 100 -d -g -vp 0.95 -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/VoxelGrids/20260915_055614_checkpoint_epoch_475.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/VoxelGrids/20260915_055614_hist_checkpoint_epoch_475.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_1.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_1.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 2 0 100 -d -g -vp 0.95 -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/DCT/20260915_191119_checkpoint_epoch_575.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/DCT/20260915_191119_hist_checkpoint_epoch_575.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_2.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_2.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 3 0 100 -d -g -vp 0.95 -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/TruncatedDCT/20260916_135348_checkpoint_epoch_625.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/TruncatedDCT/20260916_135348_hist_checkpoint_epoch_625.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_3.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_3.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 4 0 100 -d -g -vp 0.95 -mf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/AggressiveDCT/20260917_083231_checkpoint_epoch_676.pt -mhf /home/kaleb/Thesis/thesis/Training/ModelCheckpoints/DVSGESTURE/SNN/AggressiveDCT/20260917_083231_hist_checkpoint_epoch_676.pt \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_4.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_4.err"

