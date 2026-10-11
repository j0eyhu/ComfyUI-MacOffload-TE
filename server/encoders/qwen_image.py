# encoders/qwen_image.py
import os
import traceback
from typing import Any, Dict, List
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModelForImageTextToText

import config
from .base import BaseEncoder

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
T2I_TEMPLATE = SYSTEM_PROMPT + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
IM_START_TOKEN_ID = 151644


class QwenImage21Encoder(BaseEncoder):
    """
    Multimodal Text and Vision conditioning encoder for Qwen-Image-2.1 (based on Qwen3-VL-8B).
    Strictly replicates ComfyUI comfy.text_encoders.qwen_image21 behavior:
      - Uses T2I_TEMPLATE with System Prompt
      - Extracts last hidden states WITHOUT final RMSNorm
      - Drops system turn tokens before the second <|im_start|>
      - Slices vision tokens and injects `image_slots`
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = config.QWEN_MODEL_PATH
        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        self.device = config.DEVICE
        self.dtype = config.DTYPE
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[QwenImage21] Directory not found: {self.model_path}")

        print(f"[QwenImage21] Loading weights from '{self.model_path}' onto {self.device} ({self.dtype})...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map=self.device,
            local_files_only=True
        )
        self.model.eval()
        print("[QwenImage21] Model successfully loaded and resident in Unified Memory.")

    def _extract_conditioning(
        self,
        text_prompt: str,
        pil_images: List[Any],
        keep_vision: bool
    ) -> List[List[Any]]:
        raw_text = text_prompt.strip() if text_prompt else ""
        if not raw_text:
            raw_text = " "

        # 1. Format prompt matching QwenImage21Tokenizer
        if len(pil_images) > 0:
            refs = " ".join(f"<image{i+1}>{VISION_BLOCK}" for i in range(len(pil_images)))
            full_prompt = T2I_TEMPLATE.replace("{}", refs + raw_text, 1)
            inputs = self.processor(
                text=[full_prompt],
                images=pil_images,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            full_prompt = T2I_TEMPLATE.format(raw_text)
            inputs = self.tokenizer(
                [full_prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

        input_ids = inputs["input_ids"][0].tolist()

        with torch.no_grad():
            # 2. Extract last layer outputs without final RMSNorm
            backbone = getattr(self.model, "model", self.model)
            language_model = getattr(backbone, "language_model", backbone)

            outputs = backbone(**inputs, output_hidden_states=True)

            # Crucial: ComfyUI uses the raw output of the last transformer block (hidden_states[-1] before model.norm)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                # In standard transformers, hidden_states[-1] is taken before the final norm layer
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]

            hidden_states = hidden_states.to(dtype=torch.bfloat16).cpu()

        # 3. Drop system turn (everything before the second <|im_start|>)
        seq_len = hidden_states.shape[1]
        keep = torch.ones(seq_len, dtype=torch.bool)

        im_starts = [i for i, t in enumerate(input_ids) if t == IM_START_TOKEN_ID]
        if len(im_starts) > 1:
            keep[:im_starts[1]] = False

        # 4. Handle vision slots
        extra = {}
        slots = []
        if len(pil_images) > 0 and not keep_vision:
            # Drop vision tokens matching native ComfyUI logic
            vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            in_vision = False
            v_start = 0
            for idx, token in enumerate(input_ids):
                if token == vision_start_id:
                    in_vision = True
                    v_start = idx
                elif token == vision_end_id and in_vision:
                    in_vision = False
                    keep[v_start:idx + 1] = False
                    slots.append(int(keep[:v_start].sum()))

            if len(slots) > 0:
                extra["image_slots"] = slots

        # 5. Slice hidden states
        out = hidden_states[:, keep]

        # 6. Align attention mask
        if "attention_mask" in inputs:
            attn_mask = inputs["attention_mask"].cpu()[:, keep]
            if attn_mask.sum() != torch.numel(attn_mask):
                extra["attention_mask"] = attn_mask

        return [[out, extra]]

    def encode(
        self,
        prompt: str,
        negative_prompt: str = "",
        images_vl: List[torch.Tensor] = None,
        keep_vision: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        pil_images = []
        if images_vl:
            for img_tensor in images_vl:
                tensor_chw = img_tensor.detach().squeeze(0).permute(2, 0, 1).cpu()
                pil_images.append(to_pil_image(tensor_chw))

        pos_cond = self._extract_conditioning(prompt, pil_images, keep_vision=keep_vision)
        neg_cond = self._extract_conditioning(negative_prompt, pil_images, keep_vision=keep_vision)

        if self.device == "mps":
            torch.mps.empty_cache()

        return {
            "positive": pos_cond,
            "negative": neg_cond
        }
