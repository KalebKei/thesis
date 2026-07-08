#!/bin/bash

# paths
VENV_PATH="../.venv"
LOG_DIR="../Logs/Training"

# activate env
source "$VENV_PATH/bin/activate"


# encoding 1: spike train
ENC1DATESTAMP=$(date +"%m%d%y")
ENC1TIMESTAMP=$(date +"%H%M")
python trainwavelet.py > "$LOG_DIR/HaarVoxelGrids/voxelgrid_${ENC1DATESTAMP}_${ENC1TIMESTAMP}.log" 2> "$LOG_DIR/HaarVoxelGrids/voxelgrid_${ENC1DATESTAMP}_${ENC1TIMESTAMP}.err"

# encoding 2: voxel grids
# ENC2DATESTAMP=$(date +"%m%d%y")
# ENC2TIMESTAMP=$(date +"%H%M")
# python trainvoxelgrid.py > "$LOG_DIR/VoxelGridsvoxelgrid_${ENC2DATESTAMP}_${ENC2TIMESTAMP}.log" 2> "$LOG_DIR/VoxelGridsvoxelgrid_${ENC2DATESTAMP}_${ENC2TIMESTAMP}.err"

# encoding 3: DCT
# ENC3DATESTAMP=$(date +"%m%d%y")
# ENC3TIMESTAMP=$(date +"%H%M")
# python traindct.py > "$LOG_DIR/DCT/dct_${ENC3DATESTAMP}_${ENC3TIMESTAMP}.log" 2> "$LOG_DIR/DCT/dct_${ENC3DATESTAMP}_${ENC3TIMESTAMP}.err"

# encoding 4: Truncated DCT
# ENC4DATESTAMP=$(date +"%m%d%y")
# ENC4TIMESTAMP=$(date +"%H%M")
# python traintruncdct.py > "$LOG_DIR/TruncatedDCT/truncdct_${ENC4DATESTAMP}_${ENC4TIMESTAMP}.log" 2> "$LOG_DIR/TruncatedDCT/truncdct_${ENC4DATESTAMP}_${ENC4TIMESTAMP}.err"

# encoding 5: Aggressive DCT
# ENC5DATESTAMP=$(date +"%m%d%y")
# ENC5TIMESTAMP=$(date +"%H%M")
# python trainaggrdct.py > "$LOG_DIR/AggressiveDCT/aggrdct_${ENC5DATESTAMP}_${ENC5TIMESTAMP}.log" 2> "$LOG_DIR/AggressiveDCT/aggrdct_${ENC5DATESTAMP}_${ENC5TIMESTAMP}.err"