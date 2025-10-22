# LLM-Multimodal

export HF_HOME=/scratch/hpc-prf-haqc/haikai/hf-cache
export VLLM_CACHE_DIR=/scratch/hpc-prf-haqc/haikai/vllm-cache
export TORCH_COMPILE_CACHE=$VLLM_CACHE_DIR/torch_compile_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=/scratch/hpc-prf-haqc/vllm-compile-cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/hpc-prf-haqc/vllm-compile-cache

vllm serve "llava-hf/llava-1.5-7b-hf" --gpu-memory-utilization 0.5 >  vllm_server.log 2>&1 &


cd vllm
CUDA_VISIBLE_DEVICES="6,7" VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server --model /data/models/llava-1.5-7b-hf --trust-remote-code --port 8005 --tensor-parallel-size 2 --disable-log-requests --allowed-local-media-path /home/haikai/ --max_num_seqs 256 --max_num_batched_tokens=64768 --chat-template llavatemplate.jinja > vllm_server.log 2>&1 &