#!/bin/bash
# Run a pipeline command on the shared GPU server (uses only the free GPU 1 and 16 of 48 cores).
# usage: bash srv_run.sh <logname> <python args...>     e.g. bash srv_run.sh v9 run_all.py --tag v9nov --download
R=/dev/shm/pm25_work
cd "$(dirname "$(readlink -f "$0")")"
source $R/venv/bin/activate
mkdir -p $R/triton $R/tmp $R/xdg $R/skrub
export TRITON_CACHE_DIR=$R/triton TORCHINDUCTOR_CACHE_DIR=$R/inductor TMPDIR=$R/tmp XDG_CACHE_HOME=$R/xdg MPLCONFIGDIR=$R/tmp
export SKRUB_DATA_DIRECTORY=$R/skrub CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=$R/hf CUDA_VISIBLE_DEVICES=1 LGB_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
LOG=data/processed/srv_$1.log; shift
nohup python -u "$@" > $LOG 2>&1 &
echo "started pid $! -> $LOG"
