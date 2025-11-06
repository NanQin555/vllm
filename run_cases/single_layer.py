import torch
import vllm
from vllm import LLM
from vllm.model_executor.models.qwen import QWenModel

def runSingleLayer(model_path, layer_id=10):
    llm = LLM(model=model_path, dtype=torch.float16)

    model = QWenModel(vllm_config=llm.llm_engine.model_config).cuda().half()

    hidden_size = llm.llm_engine.model_config.hidden_size
    x = torch.randn(1, 16, hidden_size, device="cuda", dtype=torch.float16)

    layer = model.h[layer_id]

    out = layer(x)[0]
    print(f"Layer {layer_id} output shape:", out.shape)

    torch.cuda.synchronize()
    for _ in range(3): 
        _ = layer(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = layer(x)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Average forward time: {(end-start)/10:.4f} s")

def printModelInfoTransformers(model_path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    print(model)  

def printModelInfoVLLM(model_path):
    llm = vllm.LLM(model=model_path, dtype=torch.float16)
    print(llm.llm_engine.model_config) 

Qwen3_8B = "/home/nanqinw/models/Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
def main():
    printModelInfoTransformers(Qwen3_8B)
    # printModelInfoVLLM(Qwen3_8B)
    # runSingleLayer(Qwen3_8B, 10)
    # print(vllm.__version__)


if __name__ == "__main__":
    main()