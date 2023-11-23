#!/bin/bash

NCCL_HOME="/mnt/petrelfs/xuekui/tmp/nccl_2.15.5-1+cuda11.8_x86_64"
MPC_HOME="/mnt/petrelfs/share/gcc/mpc-0.8.1"
MPFR_HOME="/mnt/petrelfs/share/gcc/mpfr-2.4.2"
GMP_HOME="/mnt/petrelfs/share/gcc/gmp-4.3.2"
MPI_HOME="/mnt/petrelfs/share/openmpi"

export NCCL_HOME=$NCCL_HOME
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NCCL_HOME/include:$MPC_HOME/include:$MPFR_HOME/include:$GMP_HOME/include:$MPI_HOME/include
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$NCCL_HOME/include:$MPC_HOME/include:$MPFR_HOME/include:$GMP_HOME/include:$MPI_HOME/include
export LIBRARY_PATH=$LIBRARY_PATH:$NCCL_HOME/lib:$MPC_HOME/lib:$MPFR_HOME/lib:$GMP_HOME/lib:$MPI_HOME/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_HOME/lib:$MPC_HOME/lib:$MPFR_HOME/lib:$GMP_HOME/lib:$MPI_HOME/lib
export NCCL_INCLUDE_DIR=$NCCL_HOME/include
export NCCL_LIB_DIR=$NCCL_HOME/lib
export NCCL_VERSION=2
export NCCL_LAUNCH_MODE=GROUP

export PATH="/mnt/petrelfs/share/gcc/gcc-9.3.0/bin:$MPI_HOME/bin:$PATH" 

HF_MODEL=/mnt/petrelfs/xuekui/code/gpt-neox-genggui001/model_dir/pulse_v11_123b_gpt4_hf

# rm -rf $HF_MODEL/int4_awq_triton_tp1_1_a100
rm -rf $HF_MODEL/int4_awq_triton_tp2_2_a100


# python3 -u -m lmdeploy.serve.turbomind.deploy \
#     --model-name med_puyu \
#     --model-path $HF_MODEL/int4_awq \
#     --model-format awq \
#     --group-size 128 \
#     --tp 1 \
#     --dst_path $HF_MODEL/int4_awq_triton_tp1_1_a100


python3 -u -m lmdeploy.serve.turbomind.deploy \
    --model-name med_puyu \
    --model-path $HF_MODEL/int4_awq \
    --model-format awq \
    --group-size 128 \
    --tp 2 \
    --dst_path $HF_MODEL/int4_awq_triton_tp2_2_a100



# HF_MODEL=/mnt/petrelfs/xuekui/code/Megatron-DeepSpeed-Llama/model_dir/tigerbot-13b-base_v9_gpt4_hf

# python3 -u -m lmdeploy.serve.turbomind.deploy \
#     --model-name med_puyu \
#     --model-path $HF_MODEL/int4_awq \
#     --model-format awq \
#     --group-size 128 \
#     --tp 1 \
#     --dst_path $HF_MODEL/int4_awq_triton_tp1


