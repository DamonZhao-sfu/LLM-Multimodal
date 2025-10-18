# LLM-Multimodal

export HF_HOME=/scratch/hpc-prf-haqc/haikai/hf-cache
export VLLM_CACHE_DIR=/scratch/hpc-prf-haqc/haikai/vllm-cache
export TORCH_COMPILE_CACHE=$VLLM_CACHE_DIR/torch_compile_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=/scratch/hpc-prf-haqc/vllm-compile-cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/hpc-prf-haqc/vllm-compile-cache

vllm serve "llava-hf/llava-1.5-7b-hf" --gpu-memory-utilization 0.5 >  vllm_server.log 2>&1 &