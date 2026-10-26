"""Local Qwen inference with mandatory GPU placement and no CPU fallback."""

from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent

import torch

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_DIR = ROOT / "models" / "Qwen3-1.7B"


def gpu_device() -> torch.device:
    # Initialize the SYCL backend before device enumeration. On this WSL driver,
    # PyTorch 2.14's pre-init Sysman enumeration segfaults in pyzes.zesInit.
    if getattr(torch.version, "xpu", None) is not None:
        torch.xpu.init()
    if torch.xpu.is_available():
        return torch.device("xpu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    raise RuntimeError("GPU unavailable: CPU inference is forbidden.")


class QwenRuntime:
    def __init__(self):
        self.device = gpu_device()
        # Transformers also probes XPU during import; initialize the GPU first.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False
        )
        # Loading weights into host memory is not inference. All forward passes,
        # parameters, buffers, inputs and KV caches are placed on the GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, attn_implementation="sdpa",
        ).to(self.device).eval()
        if any(p.device != self.device for p in self.model.parameters()):
            raise RuntimeError("Model parameters must all reside on the GPU")
        if any(b.device != self.device for b in self.model.buffers()):
            raise RuntimeError("Model buffers must all reside on the GPU")
        self.lock = Lock()

    def generate(self, messages, max_tokens=256, temperature=0.7, tools=None):
        with self.lock, torch.inference_mode():
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt", return_dict=True,
                tools=tools,
            ).to(self.device)
            prompt_tokens = inputs["input_ids"].shape[-1]
            context_limit = min(32768, self.model.config.max_position_embeddings)
            if prompt_tokens + max_tokens > context_limit:
                raise ValueError(f"Input plus output exceeds the {context_limit}-token service limit")
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature, "top_p": 0.8, "top_k": 20}
                   if temperature > 0 else {}),
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated = outputs[0, prompt_tokens:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return text, prompt_tokens, len(generated), (
                "stop" if generated[-1].item() == self.tokenizer.eos_token_id else "length"
            )
