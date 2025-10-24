#!/bin/bash
if [[ "$1" != "" && "$1" -le 8 ]]; then
    NGPU=$1
else
    echo "Defaulting to 1 GPU."
    NGPU=1
fi
TIME=$((120-30*$NGPU))
HOURS=$(($TIME/60))
MINUTES=$(($TIME%60))
CORES_PER_GPU=32
CORES=$(($CORES_PER_GPU * $NGPU))
printf "Requesting %d GPUs (+ %d CPU-cores) for %d minutes (%02d:%02d:00)\n\n" $NGPU $CORES $TIME $HOURS $MINUTES
srun -N 1 -n 1 -c $CORES --gres=gpu:a100:$NGPU --qos=devel -p dgx -t $TIME --pty bash




#mkdir -p /scratch/hpc-prf-haqc/haikai/miniconda3
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /scratch/hpc-prf-haqc/haikai/miniconda3/miniconda.sh
#bash /scratch/hpc-prf-haqc/haikai/miniconda3/miniconda.sh -b -u -p /scratch/hpc-prf-haqc/haikai/miniconda3

# rm ~/miniconda3/miniconda.sh
