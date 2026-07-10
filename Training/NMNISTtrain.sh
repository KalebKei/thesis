#!/bin/bash

# paths
VENV_PATH="../.venv"
LOG_DIR="../Logs/NMNIST/Training/SNN"

# activate env
source "$VENV_PATH/bin/activate"

# encodings
for i in {0..4}; do
    ENCDATESTAMP=$(date +"%m%d%y")
    ENCTIMESTAMP=$(date +"%H%M")
    python trainNMNIST.py $i 0 4 -d > "$LOG_DIR/Full_${ENCDATESTAMP}_${ENCTIMESTAMP}.log" 2> "$LOG_DIR/Full_${ENCDATESTAMP}_${ENCTIMESTAMP}.err"
done