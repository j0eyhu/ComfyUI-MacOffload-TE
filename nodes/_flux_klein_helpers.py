from typing import Tuple, Any, Optional
import torch
from ._network_client import send_safetensors_request


def process_flux_klein_encode(
    model_name: str,
    text: str,
    server_url: Optional[str] = None,
) -> Tuple[Any]:
    """
    Core processing logic for remote Flux Klein text encoding.
    Prepares pure JSON metadata, invokes safetensors network client,
    and returns ComfyUI conditioning tuple.
    """
    model_name_lower = model_name.lower() if model_name else ""
    is_flux3 = (
        "flux3" in model_name_lower
        or "flux-3" in model_name_lower
        or ("qwen3" in model_name_lower and "4b" in model_name_lower)
    )
    meta = {
        "model_type": "flux3_klein" if is_flux3 else "flux_klein",
        "prompt": text if text is not None else "",
    }

    route = f"/v1/encode/{model_name}"
    res_tensors, res_meta = send_safetensors_request(
        route=route,
        tensors={},
        meta=meta,
        server_url=server_url,
    )

    if "conditioning" not in res_tensors:
        raise ValueError(
            f"[MacOffload] Malformed safetensors response from '{route}': missing 'conditioning' tensor."
        )

    conditioning = res_tensors["conditioning"]
    pooled_output = res_tensors.get("pooled_output")

    if pooled_output is None:
        pooled_output = torch.zeros(
            (conditioning.shape[0], 768),
            dtype=conditioning.dtype,
        )

    conditioning_tuple = [
        [
            conditioning,
            {
                "pooled_output": pooled_output,
            },
        ]
    ]

    return (conditioning_tuple,)
