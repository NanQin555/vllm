from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
    "Where I am?",
    "The A is"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

Qwen3_8B = "/home/nanqinw/models/Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
llm = LLM(model=Qwen3_8B)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")