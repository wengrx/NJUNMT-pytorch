#!/usr/bin/env bash

# 如果要使用多GPU，请将下面注释的内容添加到.bashrc中

#export CUDA_HOME="/home/public_data/softwares/cuda_tools/cuda-9.0"
#export PATH="$CUDA_HOME/bin:$PATH"
#export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
#export CPATH="$CUDA_HOME/include:$CPATH"
#export LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBRARY_PATH"
#alias realpath="readlink -f"
#
## added by Anaconda3 installer
#export PATH="/home/public_data/public_env/anaconda3/envs/weihr/bin:$PATH"
#
## NCCL
#export NCCL_HOME="/home/weihr/nccl_2.3.5-2+cuda9.0_x86_64/"
#export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
#export HOROVOD_CUDA_HOME="$CUDA_HOME"
#export HOROVOD_NCCL_HOME="$NCCL_HOME"
#export HOROVOD_GPU_ALLREDUCE=NCCL


export CUDA_VISIBLE_DEVICES=0,1

mpirun -np 2 \
    -H localhost:2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python -m src.bin.train \
    --model_name "transformer" \
    --reload \
    --config_path "./configs/transformer_base_config.yaml" \
    --log_path "./log" \
    --saveto "./save/" \
    --use_gpu --multi_gpu