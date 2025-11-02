

def inferModel(model_path):
    from vllm import LLM, SamplingParams
    prompts = [
        "Hello, my name is",
        "The future of AI is",
        "Where I am?",
        "The A is"
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model=model_path)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def printModelInfoTransformers(model_path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    print(model)  

def printModelInfoVLLM(model_path):
    from vllm import LLM
    import torch
    from vllm.model_executor.models.qwen import QWenModel
    llm = LLM(model=model_path, dtype=torch.float16)
    # print(llm.llm_engine.model_config) 
    model = llm.llm_engine.model_executor.driver_worker.model
    print(model.model.layers[0])


Qwen3_8B = "/home/nanqinw/models/Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
def main():
    printModelInfoTransformers(Qwen3_8B)
    # printModelInfoVLLM(Qwen3_8B)
    # inferModel(Qwen3_8B)

if __name__ == "__main__":
    main()