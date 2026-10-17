# encoders/flux3_klein.py
import os
import time
from typing import Any, Dict, List
import torch
from transformers import AutoTokenizer

import config
from .base import BaseEncoder

# Instruct template strictly matching native ComfyUI Flux3Tokenizer / Qwen3VLTokenizer (without think block)
FLUX3_KLEIN_PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class Flux3KleinEncoder(BaseEncoder):
    """
    Text conditioning encoder for FLUX.3 Klein (5.6B DiT), utilizing the Qwen3-VL-4B-Instruct architecture.

    Architecture & Parameter Notes:
      - Unlike FLUX.2 Klein (which has separate 4B and 9B DiT releases paired with Qwen3-4B / Qwen3-8B text encoders),
        FLUX.3 Klein features a single 5.6B DiT release paired with a 4B parameter Qwen3-VL text encoder.
      - Uses FLUX3_KLEIN_PROMPT_TEMPLATE (pure Instruct template without <think> block).
      - Right-pads token sequence to a multiple of 80 with pad token 151643 (<|endoftext|>).
      - Forces RoPE position_ids = 1 for padding tokens (as DiT attends to padding positions).
      - Taps 8 intermediate layers [4, 8, 12, 16, 20, 24, 28, 32].
      - Concatenates layers: movedim(1, 2).flatten(2) -> [batch, seq_len, 20480].
      - Pooled representation: [batch, 768] zeros tensor.
      - Accelerated direct Unified Memory (MPS) loading.
    """

    def __init__(self, model_path: str, max_length: int = 512):
        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        self.device = config.DEVICE
        self.dtype = config.DTYPE
        self.max_length = max_length
        self.target_layers = [4, 8, 12, 16, 20, 24, 28, 32]
        self.pad_token_id = 151643
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[Flux3Klein] Directory not found: {self.model_path}")

        print(f"[Flux3Klein] Loading tokenizer from '{self.model_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id != self.pad_token_id:
            self.tokenizer.pad_token_id = self.pad_token_id

        t0 = time.time()
        print(f"[Flux3Klein] Staging model weights with C++ empty-tensor loader ({self.dtype})...")

        # Resolve model class: prioritize direct Qwen3VLForConditionalGeneration import to avoid
        # auto-mapping import side-effects in third-party packages.
        model_cls = None
        try:
            from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
            model_cls = Qwen3VLForConditionalGeneration
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText
                model_cls = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModel
                model_cls = AutoModel

        try:
            # Solid production loader: low_cpu_mem_usage=True uses meta-tensor initialization.
            # Bypasses device_map accelerate dispatch hooks to prevent Python 3.13 Metal segfaults.
            self.model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[Flux3Klein] Staging fallback ({e}); retrying standard loader...")
            self.model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                local_files_only=True,
                trust_remote_code=True,
            )

        print(f"[Flux3Klein] Transferring weights to {self.device}...")
        self.model = self.model.to(device=self.device)
        self.model.eval()

        if self.device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        print(f"[Flux3Klein] Model successfully loaded and resident on {self.device} in {elapsed:.2f}s.")

    def unload(self) -> None:
        """Explicitly dismantles model objects to free unified memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

    def encode(
        self,
        prompt: str,
        negative_prompt: str = "",
        images_vl: List[torch.Tensor] = None,
        keep_vision: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        raw_text = prompt.strip() if prompt else ""

        # Format prompt with official FLUX.3 instruct template (no think block)
        formatted_prompt = FLUX3_KLEIN_PROMPT_TEMPLATE.format(raw_text)

        # Tokenize without automatic padding first to apply exact 80-token block padding
        token_ids = self.tokenizer.encode(
            formatted_prompt,
            add_special_tokens=False
        )

        # Right-pad to a multiple of 80 tokens, matching ComfyUI Flux3Tokenizer
        pad_len = (-len(token_ids)) % 80
        padded_token_ids = token_ids + [self.pad_token_id] * pad_len
        attention_mask = [1] * len(token_ids) + [0] * pad_len

        input_ids_tensor = torch.tensor([padded_token_ids], dtype=torch.long, device=self.device)
        attn_mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self.device)

        # Transformers 4.57+ / ComfyUI pins RoPE position for padding tokens to 1
        # kwargs["position_ids"] = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
        position_ids_tensor = (attn_mask_tensor.cumsum(-1) - 1).masked_fill(attn_mask_tensor == 0, 1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attn_mask_tensor,
                position_ids=position_ids_tensor,
                output_hidden_states=True,
            )

            all_hidden_states = outputs.hidden_states

            # Extract 8-layer tapped outputs for indices [4, 8, 12, 16, 20, 24, 28, 32]
            tapped_states = [all_hidden_states[idx] for idx in self.target_layers]

            # Replicate native ComfyUI tensor reshaping logic:
            # torch.stack [batch, 8, seq_len, 2560] -> movedim(1, 2) -> flatten(2) -> [batch, seq_len, 20480]
            stacked = torch.stack(tapped_states, dim=1)
            conditioning = stacked.movedim(1, 2).flatten(2)

            # Standard 768-dim pooled dummy representation
            pooled_output = torch.zeros(
                (conditioning.shape[0], 768),
                dtype=self.dtype,
                device=self.device
            )

            conditioning = conditioning.to(dtype=torch.bfloat16).cpu()
            pooled_output = pooled_output.to(dtype=torch.bfloat16).cpu()

        if self.device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return {
            "conditioning": [
                [
                    conditioning,
                    {
                        "pooled_output": pooled_output
                    }
                ]
            ]
        }
