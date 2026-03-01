#!/bin/bash
# Wrapper that sets required CUDA/cuDNN library paths for the prosim conda env
# (System has cuDNN 8; PyTorch 2.4.1 requires cuDNN 9 from pip nvidia packages)

NVIDIA_USER_LIBS="/home/msc-auto/.local/lib/python3.8/site-packages/nvidia"
NCCL_LIB="/home/msc-auto/miniconda3/lib/python3.9/site-packages/nvidia/nccl/lib"

export LD_LIBRARY_PATH="\
${NVIDIA_USER_LIBS}/cuda_runtime/lib:\
${NVIDIA_USER_LIBS}/nvtx/lib:\
${NVIDIA_USER_LIBS}/cuda_cupti/lib:\
${NVIDIA_USER_LIBS}/nvjitlink/lib:\
${NVIDIA_USER_LIBS}/cuda_nvrtc/lib:\
${NVIDIA_USER_LIBS}/cublas/lib:\
${NVIDIA_USER_LIBS}/curand/lib:\
${NVIDIA_USER_LIBS}/cufft/lib:\
${NVIDIA_USER_LIBS}/cudnn/lib:\
${NVIDIA_USER_LIBS}/cusparse/lib:\
${NCCL_LIB}:\
${LD_LIBRARY_PATH}"

exec /home/msc-auto/miniconda3/envs/prosim/bin/python "$@"
