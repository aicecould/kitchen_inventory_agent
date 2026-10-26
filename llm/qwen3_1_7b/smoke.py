"""Exercise actual GPU generation and save the observed result."""

import json
from pathlib import Path
from time import perf_counter

import torch

from llm.qwen3_1_7b.runtime import MODEL_ID, QwenRuntime

if __name__ == "__main__":
    runtime = QwenRuntime()
    prompt = "请用中文简短回答：鸡蛋和西红柿可以做什么菜？"
    started = perf_counter()
    text, input_tokens, output_tokens, finish = runtime.generate(
        [{"role": "user", "content": prompt}], max_tokens=96, temperature=0
    )
    result = {
        "model": MODEL_ID, "torch": torch.__version__,
        "device": str(runtime.device),
        "gpu": (torch.xpu.get_device_name(0) if runtime.device.type == "xpu"
                else torch.cuda.get_device_name(0)),
        "parameter_devices": sorted({str(p.device) for p in runtime.model.parameters()}),
        "dtype": str(next(runtime.model.parameters()).dtype),
        "prompt": prompt, "response": text,
        "prompt_tokens": input_tokens, "completion_tokens": output_tokens,
        "finish_reason": finish, "elapsed_seconds": round(perf_counter() - started, 3),
    }
    if not text.strip():
        raise RuntimeError("Model generated an empty response")
    output = json.dumps(result, ensure_ascii=False, indent=2)
    run_dir = Path(__file__).resolve().parents[2] / ".run" / "qwen3_1_7b"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "deployment-result.json").write_text(output + "\n")
    print(output)
