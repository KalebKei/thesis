#!/bin/bash

# Conda environment name
CONDA_ENV="thesis"

# Paths
LOG_DIR="../Logs/DVSGESTURE/Training/HaarSNN"

# Create log directory if it does not exist 
mkdir -p "$LOG_DIR"

# Initialize Conda for this non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate Conda environment
conda activate "$CONDA_ENV"

# Encodings
for i in {0..4}; do
    ENCDATESTAMP=$(date +"%m%d%y")
    ENCTIMESTAMP=$(date +"%H%M")

    python trainDVSGESTURE.py "$i" 1 50 -d -g \
        > "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_${i}.log" \
        2> "$LOG_DIR/${ENCDATESTAMP}_${ENCTIMESTAMP}_full_${i}.err"
done

