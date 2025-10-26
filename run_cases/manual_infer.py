#!/usr/bin/env python3

import os
import argparse
import time
import math
import torch

# NVTX 标注（可选）
try:
    import torch.cuda.nvtx as nvtx
    HAVE_NVTX = True
except Exception:
    HAVE_NVTX = False

# ---- 下面两行是典型 vLLM 的导入方式；不同版本可能略有差别 ----
# Try to import LLMEngine / LLM. 如果找不到请根据你的仓库结构修改导入。
try:
    from vllm.engine.llm_engine import LLMEngine
    from vllm.engine.arg_utils import EngineArgs
    HAVE_VLLM_ENGINE = True
except Exception:
    HAVE_VLLM_ENGINE = False

try:
    # some vllm versions expose a top-level LLM wrapper
    from vllm import LLM
    HAVE_VLLM = True
except Exception:
    HAVE_VLLM = False

# ---- Async communication stub ----
# Replace this with your project's communication primitives (e.g. vllm.parallel_utils.communication)
def async_all_reduce_stub(tensor):
    """
    示例异步 all_reduce stub。真实使用时请替换为 vllm 的通信实现或 torch.distributed.all_reduce(..., async_op=True)
    返回一个 work-like object with .wait() 或 None（同步实现）
    """
    if not torch.distributed.is_initialized():
        # 单机单卡 or 未初始化 distributed：同步返回 None
        return None
    # 演示：使用 PyTorch async_op（注意：需要正确初始化 torch.distributed）
    try:
        work = torch.distributed.all_reduce(tensor, async_op=True)
        return work
    except Exception:
        return None

# ---- Utilities: sampling ----
def topk_sample(logits, k=50):
    """ logits: 1D tensor of token logits. 返回 token index (int) """
    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return int(indices[idx])

def greedy_sample(logits):
    return int(torch.argmax(logits).item())

# ---- Main driver ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or model id")
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable-overlap", type=int, default=0,
                        help="1 to enable communication-computation overlap stub")
    parser.add_argument("--sampling", choices=["greedy", "topk"], default="greedy")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--ddp_rank", type=int, default=0, help="If distributed: rank")
    parser.add_argument("--ddp_world_size", type=int, default=1, help="If distributed: world size")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    # 如果需要分布式，请在外部或这里初始化（示例单节点多进程需要环境变量）
    if args.ddp_world_size > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://",
                world_size=args.ddp_world_size, rank=args.ddp_rank
            )
            torch.cuda.set_device(device.index if device.type == "cuda" else 0)
        print(f"[info] torch.distributed initialized: rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()}")

    # --------------------------
    # 1) 初始化 vLLM engine / model runner
    # --------------------------
    model = args.model
    engine = None
    model_runner = None

    if HAVE_VLLM_ENGINE:
        print("[info] Creating LLMEngine via EngineArgs (LLMEngine.from_engine_args)...")
        # NOTE: EngineArgs 是示例，你可能需要根据本地 vLLM 修改字段
        ea = EngineArgs(model=model, tensor_parallel_size=1)
        engine = LLMEngine.from_engine_args(ea)
        # 下面两行的路径名会因版本不同而变，请根据你本地结构调整
        model_executor = engine.model_executor
        model_runner = model_executor.model_runner
        tokenizer = engine.tokenizer
    elif HAVE_VLLM:
        print("[info] Using top-level LLM wrapper (vllm.LLM)...")
        llm = LLM(model=model)
        # 取内部引用（版本差异可能需要适配）
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            raise RuntimeError("Could not get engine from LLM instance; adapt script to your vllm version.")
        model_executor = engine.model_executor
        model_runner = model_executor.model_runner
        tokenizer = engine.tokenizer
    else:
        raise RuntimeError("Cannot find vllm imports. Please run this script inside vllm repo env and adapt imports if needed.")

    # --------------------------
    # 2) tokenization + prefill（一次性处理 prompt）
    # --------------------------
    prompt = args.prompt
    print(f"[info] Prompt: {prompt!r}")

    # tokenization: 你可能需要换成 engine.tokenize(...) 的具体方法
    try:
        enc = tokenizer.encode(prompt)
        # token ids -> torch tensor on device
        input_ids = torch.tensor([enc], dtype=torch.long, device=device)
    except Exception as e:
        # fallback: if tokenizer is a huggingface tokenizer
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(model)
            enc = hf_tok(prompt, return_tensors="pt")["input_ids"].to(device)
            input_ids = enc
            tokenizer = hf_tok
        except Exception:
            raise RuntimeError("Tokenization failed; adapt tokenizer usage for your vLLM setup.") from e

    # Prefill: 执行一次 prompt prefill，建立 KV cache
    print("[info] Running prefill...")
    # 不同版本接口不同，这里给两个备选：
    prefill_ok = False
    # Option A: model_runner 提供 run_prefill / prefill API
    if hasattr(model_runner, "run_prefill") or hasattr(model_runner, "prefill"):
        fn = getattr(model_runner, "run_prefill", None) or getattr(model_runner, "prefill", None)
        # 调用并等待返回
        res = fn(input_ids)
        prefill_ok = True
    # Option B: model_executor 提供 execute_model(prefill=True,...)
    elif hasattr(model_executor, "execute_model"):
        try:
            res = model_executor.execute_model(prefill=True, input_ids=input_ids)
            prefill_ok = True
        except TypeError:
            # 接口签名不同 — 请根据本地版本调整
            pass

    if not prefill_ok:
        raise RuntimeError("Prefill API not found on model_runner/model_executor. Adapt script for your vLLM version.")

    print("[info] Prefill done. Entering manual decode loop.")

    # --------------------------
    # 3) 手动 step-by-step decode loop
    # --------------------------
    max_new = args.max_new_tokens
    generated_tokens = []
    # 如果 model_runner 提供 step 接口（示例名：run_decode_one_step / decode_step）
    step_fn = None
    if hasattr(model_runner, "run_decode_one_step"):
        step_fn = model_runner.run_decode_one_step
    elif hasattr(model_runner, "decode_step"):
        step_fn = model_runner.decode_step
    elif hasattr(model_executor, "decode_step"):
        step_fn = model_executor.decode_step
    else:
        # 可能需要直接调用 model_executor.execute_model with decode flag; 这里做最保守的 fallback
        raise RuntimeError("No decode step API found; adapt to your vLLM version (look for run_decode_one_step/decode_step).")

    for step in range(max_new):
        # Optional NVTX mark (便于 Nsight timeline 定位)
        if HAVE_NVTX:
            nvtx.range_push(f"decode_step_{step}")

        # Create CUDA events for timing
        if device.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_evt.record()

        # ------------- Optionally start async communication BEFORE compute -------------
        # This demonstrates the pattern: 如果你的 next step 需要从别的 rank 收到 shard，
        # 你可以在这里发起 async collective（返回 work），随后执行一些可并行计算，
        # 再在需要时调用 work.wait()
        async_work = None
        if args.enable_overlap:
            # 这里示例使用 stub；替换为 vllm 内部通信函数以获得真实效果
            # 通常你会对一个 buffer 发起 all_gather/reduce_scatter 等
            dummy_tensor = torch.zeros(1, device=device)
            async_work = async_all_reduce_stub(dummy_tensor)

        # ------------- 执行 decode step（得到 logits / token）-------------
        # run decode step -> returns logits for next token (API dependent)
        out = step_fn()  # adapt if step_fn requires args
        # out 的结构视 vLLM 版本而定；常见为 dict 包含 "logits" 或 "next_token_logits"
        # 下面给出几种常见情况的解析逻辑：
        logits = None
        if isinstance(out, dict):
            if "logits" in out:
                logits = out["logits"]
            elif "next_token_logits" in out:
                logits = out["next_token_logits"]
            elif "logits_for_next_token" in out:
                logits = out["logits_for_next_token"]
        elif torch.is_tensor(out):
            logits = out
        else:
            # 视具体返回值格式调整
            raise RuntimeError("Unexpected decode step return type; inspect `out` structure.")

        # logits 形状通常是 (vocab,) 或 (1, vocab)
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)

        # 如果之前发起了 async 所需结果，在这里 wait
        if async_work is not None:
            try:
                async_work.wait()
            except Exception:
                # 如果 work 不支持 wait（stub），忽略
                pass

        # 记录时间
        if device.type == "cuda":
            end_evt.record()
            torch.cuda.synchronize()
            el_ms = start_evt.elapsed_time(end_evt)
        else:
            el_ms = None

        # ------------- 从 logits 采样 / 选择 token -------------
        if args.sampling == "greedy":
            next_token_id = greedy_sample(logits)
        else:
            next_token_id = topk_sample(logits, k=args.topk)

        # 更新 KV cache 与 internal state：大多数实现会提供 update_kv_cache 或者由 run_decode_one_step 自动处理
        # 如果需要手动更新，请在这里调用相应接口（示例 placeholder）
        if hasattr(model_runner, "update_kv_cache"):
            model_runner.update_kv_cache(next_token_id)
        # 否则 assume step_fn 已自动把 token 应用到 cache

        generated_tokens.append(next_token_id)

        # 转换 token 为文字并输出（根据 tokenizer）
        try:
            token_str = tokenizer.decode([next_token_id])
        except Exception:
            # HF tokenizer decode
            try:
                token_str = tokenizer._convert_id_to_token(next_token_id)
            except Exception:
                token_str = f"<{next_token_id}>"

        print(f"[step {step}] token_id={next_token_id} token={token_str!r} compute_ms={el_ms}")

        if HAVE_NVTX:
            nvtx.range_pop()

        # 可选早停（如遇到 EOS）
        # 这里假设 tokenizer 提供 eos token id：
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and next_token_id == eos_id:
            print("[info] EOS token generated, stopping.")
            break

    # ------------- post-processing: 输出完整文本 -------------
    try:
        full = tokenizer.decode(generated_tokens)
    except Exception:
        # 如果 decode 失败，逐 token 拼接
        full = "".join([str(t) for t in generated_tokens])
    print("=== Generated text ===")
    print(full)
    print("======================")

    # cleanup
    if args.ddp_world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
