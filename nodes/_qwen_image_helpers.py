import math
from typing import Any, Dict, List, Optional, Tuple
import torch
from ._network_client import send_safetensors_request

try:
    import comfy.utils
    import comfy.model_management
    import node_helpers
except ImportError:
    # Fail-soft fallback for testing outside of a running ComfyUI process
    comfy = None
    node_helpers = None


def process_qwen_image_encode(
    prompt: str,
    negative_prompt: str,
    model_name: str = "Qwen-Image-2.1-Text-Encoder-Heretic",
    vae: Any = None,
    resolution: int = 1024,
    server_url: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Core processing logic for remote Qwen Image 2.1 multimodal conditioning.
    Preprocesses reference images, prepares safetensors payload, communicates with
    remote Mac server, attaches reference latents, and generates WanVAE latents.
    """
    ref_latents: List[Any] = []
    images_vl: List[torch.Tensor] = []
    latent_w = latent_h = resolution or 1024

    # 1. Process multi-image reference inputs matching native implementation
    image_keys = sorted(
        [k for k in kwargs.keys() if k.startswith("image_")],
        key=lambda n: int(n.rsplit("_", 1)[-1]),
    )

    for name in image_keys:
        image = kwargs.get(name)
        if image is None:
            continue

        # Apply identical resizing logic for both text encoder and VAE
        samples = image[:1].movedim(-1, 1)
        if resolution > 0:
            ratio = samples.shape[3] / samples.shape[2]
            width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
            height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
        else:
            width = round(samples.shape[3] / 32) * 32
            height = round(samples.shape[2] / 32) * 32

        width, height = max(32, width), max(32, height)
        if (width, height) == (samples.shape[3], samples.shape[2]):
            s = image[:1]
        else:
            if comfy is not None:
                s = comfy.utils.common_upscale(
                    samples, width, height, "lanczos", "disabled"
                ).movedim(1, -1)
            else:
                s = image[:1]

        if not images_vl:
            latent_w, latent_h = width, height

        rgb = s[:, :, :, :3]
        if s.shape[-1] > 3:
            rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])

        images_vl.append(rgb.cpu())

        if vae is not None:
            ref_latents.append(vae.encode(s))

    keep_vision = len(ref_latents) == 0

    # 2. Build serialized pure data safetensors payload
    tensors_payload: Dict[str, torch.Tensor] = {}
    for idx, img in enumerate(images_vl):
        tensors_payload[f"image_{idx}"] = img.contiguous()

    meta_payload = {
        "model_type": "qwen_image",
        "prompt": prompt if prompt is not None else "",
        "negative_prompt": negative_prompt if negative_prompt is not None else "",
        "keep_vision": keep_vision,
        "image_count": len(images_vl),
    }

    # 3. Post to Mac server endpoint via safetensors
    route = f"/v1/encode/{model_name}"
    res_tensors, res_meta = send_safetensors_request(
        route=route,
        tensors=tensors_payload,
        meta=meta_payload,
        server_url=server_url,
    )

    if "positive_embeds" not in res_tensors or "negative_embeds" not in res_tensors:
        raise ValueError(
            f"[MacOffload] Malformed safetensors response from '{route}': missing positive/negative embeddings."
        )

    pos_extra: Dict[str, Any] = {}
    neg_extra: Dict[str, Any] = {}

    pos_slots = res_meta.get("pos_slots", [])
    if pos_slots:
        pos_extra["image_slots"] = pos_slots

    neg_slots = res_meta.get("neg_slots", [])
    if neg_slots:
        neg_extra["image_slots"] = neg_slots

    if "positive_mask" in res_tensors:
        pos_extra["attention_mask"] = res_tensors["positive_mask"]

    if "negative_mask" in res_tensors:
        neg_extra["attention_mask"] = res_tensors["negative_mask"]

    positive = [[res_tensors["positive_embeds"], pos_extra]]
    negative = [[res_tensors["negative_embeds"], neg_extra]]

    # 4. Inject reference_latents into conditioning dictionaries
    if len(ref_latents) > 0 and node_helpers is not None:
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": ref_latents}, append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"reference_latents": ref_latents}, append=True
        )

    # 5. Construct WanVAE compatible 64-channel Latent tensor
    intermediate_device = (
        comfy.model_management.intermediate_device()
        if (comfy is not None and hasattr(comfy, "model_management"))
        else "cpu"
    )
    latent = torch.zeros(
        [1, 64, latent_h // 16, latent_w // 16],
        device=intermediate_device,
    )

    # 6. Return conditioning tuple (positive, negative, {"samples": latent})
    return (positive, negative, {"samples": latent})
