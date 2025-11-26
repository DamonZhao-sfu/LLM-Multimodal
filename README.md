# LLM-Multimodal

export HF_HOME=/scratch/hpc-prf-haqc/haikai/hf-cache
export VLLM_CACHE_DIR=/scratch/hpc-prf-haqc/haikai/vllm-cache
export TORCH_COMPILE_CACHE=$VLLM_CACHE_DIR/torch_compile_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=/scratch/hpc-prf-haqc/vllm-compile-cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/hpc-prf-haqc/vllm-compile-cache

# Load and run the model:
vllm serve # Load and run the model:

# Load and run the model:
vllm serve "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct" --gpu-memory-utilization 0.8 --enable-mm-embeds --trust_remote_code True>  vllm_server.log 2>&1 &

vllm serve "llava-hf/llava-onevision-qwen2-7b-ov-hf" --gpu-memory-utilization 0.8 --enable-mm-embeds >  vllm_server.log 2>&1 &

vllm serve "llava-hf/llava-1.5-7b-hf" --gpu-memory-utilization 0.8  >  vllm_server.log 2>&1 &

vllm serve "llava-hf/llava-v1.6-mistral-7b-hf" --gpu-memory-utilization 0.8 --enable-mm-embeds >  vllm_server.log 2>&1 &

CUDA_VISIBLE_DEVICES="0,1" vllm serve "Qwen/Qwen2.5-VL-7B-Instruct" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  > vllm_server.log 2>&1 &


cd vllm
CUDA_VISIBLE_DEVICES="6,7" VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server --model /data/models/llava-1.5-7b-hf --trust-remote-code --port 8000 --tensor-parallel-size 2 --disable-log-requests --allowed-local-media-path /home/haikai/ --max_num_seqs 256 --max_num_batched_tokens=64768 --chat-template llavatemplate.jinja > vllm_server.log 2>&1 &