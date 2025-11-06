
# import torch
# from vllm import LLM, SamplingParams

# MODEL_PATH = "/home/nanqinw/models/Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"

# torch.cuda.set_per_process_memory_fraction(1.0, 0)

# if __name__ == "__main__":
#     # 模拟 4 张 GPU
#     llm = LLM(
#         model=MODEL_PATH,
#         tensor_parallel_size=4,
#         engine_args={"use_ray": False} 
#     )

#     prompt_list = [f"Hello from virtual rank {i}!" for i in range(4)]
#     sampling_params = SamplingParams(max_tokens=50)

#     outputs = llm.generate(prompt_list, sampling_params)

#     for i, output in enumerate(outputs):
#         print("="*50)
#         print(f"[Virtual Rank {i}] Prompt: {output.prompt!r}")
#         print(f"[Virtual Rank {i}] Generated: {output.outputs[0].text!r}")
