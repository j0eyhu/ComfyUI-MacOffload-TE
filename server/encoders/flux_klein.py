# encoders/flux_klein.py
import os
import time
from typing import Any, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import config
from .base import BaseEncoder

# Template strictly matching native ComfyUI KleinTokenizer
KLEIN_PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class FluxKleinEncoder(BaseEncoder):
    """
    Text conditioning encoder for Flux.2 Klein based on Qwen3 architecture.
    Strictly aligns with ComfyUI comfy/text_encoders/flux.py:
      - Uses KLEIN_PROMPT_TEMPLATE
      - Taps layer indices [9, 18, 27]
      - Concatenates 3 layers to 12288 dimensions
      - Accelerated direct Unified Memory (MPS) loading
    """

    def __init__(self, model_path: str, max_length: int = 512):
        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        self.device = config.DEVICE
        self.dtype = config.DTYPE
        self.max_length = max_length
        self.target_layers = [9, 18, 27]
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[FluxKlein] Directory not found: {self.model_path}")

        print(f"[FluxKlein] Loading tokenizer from '{self.model_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id != 151643:
            self.tokenizer.pad_token_id = 151643

        t0 = time.time()
        print(f"[FluxKlein] Staging model weights with C++ empty-tensor loader ({self.dtype})...")
        try:
            # Solid production loader: low_cpu_mem_usage=True uses meta-tensor initialization.
            # Bypasses device_map accelerate dispatch hooks to prevent Python 3.13 Metal segfaults.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[FluxKlein] Staging fallback ({e}); retrying standard loader...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                local_files_only=True,
                trust_remote_code=True,
            )

        print(f"[FluxKlein] Transferring weights to {self.device}...")
        self.model = self.model.to(device=self.device, non_blocking=True)
        self.model.eval()

        if self.device == "mps":
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        print(f"[FluxKlein] Model successfully loaded and resident on {self.device} in {elapsed:.2f}s.")

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

        # Format prompt with native thinking template
        formatted_prompt = KLEIN_PROMPT_TEMPLATE.format(raw_text)

        inputs = self.tokenizer(
            formatted_prompt,
            padding="max_length",
            max_length=max(self.max_length, kwargs.get("max_length", 512)),
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            all_hidden_states = outputs.hidden_states

            # Extract layer outputs for indices [9, 18, 27]
            tapped_states = [all_hidden_states[idx] for idx in self.target_layers]

            # Replicate native ComfyUI tensor reshaping logic:
            # torch.stack -> movedim(1, 2) -> reshape [batch, seq_len, 12288]
            stacked = torch.stack(tapped_states, dim=1)
            stacked = stacked.movedim(1, 2)
            conditioning = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)

            # Standard 768-dim pooled dummy representation
            pooled_output = torch.zeros(
                (conditioning.shape[0], 768),
                dtype=self.dtype,
                device=self.device
            )

            conditioning = conditioning.to(dtype=torch.bfloat16).cpu()
            pooled_output = pooled_output.to(dtype=torch.bfloat16).cpu()

        if self.device == "mps":
            torch.mps.empty_cache()

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


class FluxKlein4BEncoder(FluxKleinEncoder):
    def __init__(self, model_path: str = None):
        super().__init__(model_path=model_path or config.FLUX_KLEIN_4B_PATH)


class FluxKlein9BEncoder(FluxKleinEncoder):
    def __init__(self, model_path: str = None):
        super().__init__(model_path=model_path or config.FLUX_KLEIN_9B_PATH)
