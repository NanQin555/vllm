# !/bin/bash

# set env for profiler
export VLLM_TORCH_PROFILER_DIR=../vllm_profile
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1

python ./vllm_infer.py