#!/bin/bash

# Conda environment name
CONDA_ENV="thesis"

# Paths
LOG_DIR="../Logs/DVSGESTURE/Training/FrontEndWaveletSNN/"

# Create log directory if it does not exist 
mkdir -p "$LOG_DIR"

# Initialize Conda for this non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate Conda environment
conda activate "$CONDA_ENV"

# Encodings

ENCDATESTAMP=$(date +"%m%d%y")
ENCTIMESTAMP=$(date +"%H%M")
 python trainDVSGESTURE.py 0 1 25 -g -vp 0.90 -mf /home/kalebkei/ThesisWork/Training/ModelCheckpoints/DVSGESTURE/FrontEndWaveletSNN/SpikeTrain/20260924_010542_checkpoint_epoch_86.pt -mhf /home/kalebkei/ThesisWork/Training/ModelCheckpoints/DVSGESTURE/FrontEndWaveletSNN/SpikeTrain/20260924_010542_hist_checkpoint_epoch_86.pt \
     > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_0.log" \
     2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_0.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 1 1 25 -g -vp 0.90 \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_1.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_1.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 2 1 25 -g -vp 0.90 \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_2.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_2.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 3 1 25 -g -vp 0.90 \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_3.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_3.err"

# ENCDATESTAMP=$(date +"%m%d%y")
# ENCTIMESTAMP=$(date +"%H%M")

python trainDVSGESTURE.py 4 1 25 -g -vp 0.90 \
    > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_4.log" \
    2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_4.err"

