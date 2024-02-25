#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# HF_MODEL=/mnt/petrelfs/xuekui/code/Megatron-DeepSpeed-Llama/model_dir/puyu_123b_pretrain-v9-gpt4-hf
# CALIB_JSONL_PATH=/mnt/petrelfs/xuekui/code/Megatron-DeepSpeed-Llama/qua/puyu_123b_pretrain-v9-gpt4.jsonl

# HF_MODEL=/mnt/petrelfs/xuekui/code/Megatron-DeepSpeed-Llama/model_dir/tigerbot-13b-base_v9_gpt4_hf
# CALIB_JSONL_PATH=/mnt/petrelfs/xuekui/code/Megatron-DeepSpeed-Llama/qua/tigerbot-13b-v9-gpt4.jsonl

# HF_MODEL=/mnt/petrelfs/xuekui/code/gpt-neox-genggui001/model_dir/pulse_v11_123b_gpt4_hf
# CALIB_JSONL_PATH=/mnt/petrelfs/xuekui/code/gpt-neox-genggui001/qua/XuanYuan_70B.jsonl

# rm -rf $HF_MODEL/int4_awq
# mkdir -p $HF_MODEL/int4_awq

# python3 -u -m lmdeploy.lite.apis.calibrate_use_jsonl \
#   --model /mnt/petrelfs/xuekui/.cache/huggingface/hub/models--Duxiaoman-DI--XuanYuan-70B/snapshots/f73650aef17707804d9f6e97fd3a46437a8e08db \
#   --calib_jsonl_path /mnt/petrelfs/xuekui/code/gpt-neox-genggui001/qua/XuanYuan_70B.jsonl \
#   --calib_samples 48 \
#   --calib_seqlen 8192 \
#   --work_dir /mnt/petrelfs/xuekui/pretrain_weights/nlp/XuanYuan-70B-smooth


python3 -u -m lmdeploy.lite.apis.smooth_layers \
  --model /mnt/petrelfs/xuekui/.cache/huggingface/hub/models--Duxiaoman-DI--XuanYuan-70B/snapshots/f73650aef17707804d9f6e97fd3a46437a8e08db \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir /mnt/petrelfs/xuekui/pretrain_weights/nlp/XuanYuan-70B-smooth

