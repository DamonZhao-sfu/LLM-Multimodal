from vllm import LLM
import os
import torch

if __name__ == "__main__":
    # Initialize the LLM only if this is the main process
    # llava-hf/llava-1.5-7b-hf
    # llava-hf/llava-v1.6-mistral-7b-hf
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf", 
            trust_remote_code=True,
            download_dir=os.environ["HF_HOME"])

    prompt = "USER: <image>\nWhat is this image?\nASSISTANT:"
    #model_config = llm.model_config

    
    # case 2
    #image_embeds = torch.zeros((16, model_config.hf_config.text_config.hidden_size))
    
    # case 3
    image_embeds = [torch.zeros(16, 1024)]
    

    # 2. Wrap it in MultiModalEmbedding
    # This tells vLLM: "This is an embedding, do not use HF image processor"
    out = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image_embeds}, 
    })
    print(out[0].outputs[0].text)
    
 
# 1
#     Traceback (most recent call last):
#   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/multimodal/processing.py", line 1064, in call_hf_processor
#     output = hf_processor(**data, **allowed_kwargs, return_tensors="pt")
#   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/site-packages/transformers/models/llava_onevision/processing_llava_onevision.py", line 165, in __call__
#     image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
#   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/site-packages/transformers/image_processing_utils.py", line 51, in __call__
#     return self.preprocess(images, **kwargs)
#   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/site-packages/transformers/models/llava_onevision/image_processing_llava_onevision.py", line 696, in preprocess
#     images = make_flat_list_of_images(images)
#   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/site-packages/transformers/image_utils.py", line 245, in make_flat_list_of_images
#     raise ValueError(f"Could not make a flat list of images from {images}")
# ValueError: Could not make a flat list of images from [tensor([0.2584, 0.0609, 0.9136,  ..., 0.8697, 0.4567, 0.5591]), tensor([0.4842, 0.0263, 0.5300,  ..., 0.5066, 0.2075, 0.8987]), tensor([0.3425, 0.8657, 0.9972,  ..., 0.6739, 0.0388, 0.8260]), tensor([0.9120, 0.3767, 0.1720,  ..., 0.7541, 0.1369, 0.3183]), tensor([0.7738, 0.6794, 0.7538,  ..., 0.7612, 0.5966, 0.6114]), tensor([0.7852, 0.2506, 0.2160,  ..., 0.7851, 0.3964, 0.9316]), tensor([0.6908, 0.9116, 0.6687,  ..., 0.1566, 0.1714, 0.9841]), tensor([0.6019, 0.0050, 0.1403,  ..., 0.4697, 0.4494, 0.3051]), tensor([0.1942, 0.3804, 0.3525,  ..., 0.2697, 0.1140, 0.9054]), tensor([0.9904, 0.1340, 0.1460,  ..., 0.1298, 0.8521, 0.7786]), tensor([0.1024, 0.2608, 0.3714,  ..., 0.7538, 0.2897, 0.3285]), tensor([0.5446, 0.7627, 0.4683,  ..., 0.8070, 0.2907, 0.9207]), tensor([0.5798, 0.3689, 0.0264,  ..., 0.6598, 0.8576, 0.9197]), tensor([0.9155, 0.3978, 0.4749,  ..., 0.9469, 0.9829, 0.8905]), tensor([0.8760, 0.8434, 0.6236,  ..., 0.8607, 0.2392, 0.5023]), tensor([0.6209, 0.8098, 0.4869,  ..., 0.4380, 0.7663, 0.7008])]
 
 
    
# 2
# ```
# (EngineCore_DP0 pid=3454063) Process EngineCore_DP0:
# (EngineCore_DP0 pid=3454063) Traceback (most recent call last):
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
# (EngineCore_DP0 pid=3454063)     self.run()
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/multiprocessing/process.py", line 108, in run
# (EngineCore_DP0 pid=3454063)     self._target(*self._args, **self._kwargs)
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/engine/core.py", line 846, in run_engine_core
# (EngineCore_DP0 pid=3454063)     raise e
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/engine/core.py", line 835, in run_engine_core
# (EngineCore_DP0 pid=3454063)     engine_core.run_busy_loop()
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/engine/core.py", line 862, in run_busy_loop
# (EngineCore_DP0 pid=3454063)     self._process_engine_step()
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/engine/core.py", line 891, in _process_engine_step
# (EngineCore_DP0 pid=3454063)     outputs, model_executed = self.step_fn()
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/engine/core.py", line 345, in step
# (EngineCore_DP0 pid=3454063)     model_output = future.result()
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/concurrent/futures/_base.py", line 451, in result
# (EngineCore_DP0 pid=3454063)     return self.__get_result()
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
# (EngineCore_DP0 pid=3454063)     raise self._exception
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/executor/uniproc_executor.py", line 79, in collective_rpc
# (EngineCore_DP0 pid=3454063)     result = run_method(self.driver_worker, method, args, kwargs)
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/serial_utils.py", line 479, in run_method
# (EngineCore_DP0 pid=3454063)     return func(*args, **kwargs)
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/worker/worker_base.py", line 369, in execute_model
# (EngineCore_DP0 pid=3454063)     return self.worker.execute_model(scheduler_output, *args, **kwargs)
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
# (EngineCore_DP0 pid=3454063)     return func(*args, **kwargs)
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/worker/gpu_worker.py", line 552, in execute_model
# (EngineCore_DP0 pid=3454063)     output = self.model_runner.execute_model(
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
# (EngineCore_DP0 pid=3454063)     return func(*args, **kwargs)
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/worker/gpu_model_runner.py", line 2743, in execute_model
# (EngineCore_DP0 pid=3454063)     ) = self._preprocess(
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/worker/gpu_model_runner.py", line 2318, in _preprocess
# (EngineCore_DP0 pid=3454063)     self._execute_mm_encoder(scheduler_output)
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/worker/gpu_model_runner.py", line 1997, in _execute_mm_encoder
# (EngineCore_DP0 pid=3454063)     sanity_check_mm_encoder_outputs(
# (EngineCore_DP0 pid=3454063)   File "/scratch/hpc-prf-haqc/haikai/vllm/vllm/v1/worker/utils.py", line 196, in sanity_check_mm_encoder_outputs
# (EngineCore_DP0 pid=3454063)     assert all(e.ndim == 2 for e in mm_embeddings), (
# (EngineCore_DP0 pid=3454063) AssertionError: Expected multimodal embeddings to be a sequence of 2D tensors, but got tensors with shapes [torch.Size([1, 16, 1152])] instead. This is most likely due to incorrect implementation of the model's `embed_multimodal` method.

# ```

3
    # (EngineCore_DP0 pid=3444746)     assert all(e.ndim == 2 for e in mm_embeddings), ( (EngineCore_DP0 pid=3444746) AssertionError: Expected multimodal embeddings to be a sequence of 2D tensors, but got tensors with shapes [torch.Size([1, 16, 1152])] instead. This is most likely due to incorrect implementation of the model's `embed_multimodal` method.



4

    #  File "/scratch/hpc-prf-haqc/haikai/conda3/envs/spark/lib/python3.10/site-packages/transformers/image_transforms.py", line 434, in normalize
    #raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}") ValueError: mean must have 1 elements if it is an iterable, got 3
    


# It is safe (and often recommended) to keep environment setup global 
# so spawned processes inherit these settings if they import the module.
# scratch_path = "/scratch/hpc-prf-haqc"
# user_path = f"{scratch_path}/haikai"

# os.environ["HF_HOME"] = f"{user_path}/hf-cache"
# os.environ["VLLM_CACHE_DIR"] = f"{user_path}/vllm-cache"
# os.environ["TORCH_COMPILE_CACHE"] = f"{user_path}/vllm-cache/torch_compile_cache"
# os.environ["TRANSFORMERS_CACHE"] = f"{user_path}/hf-cache"

# os.environ["TRITON_CACHE_DIR"] = f"{scratch_path}/vllm-compile-cache"
# os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{scratch_path}/vllm-compile-cache"
