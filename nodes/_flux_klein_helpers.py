from typing import Tuple, Any
from ._network_client import send_binary_encode_request


def process_flux_klein_encode(server_url: str, model_name: str, text: str) -> Tuple[Any]:
    """
    Core processing logic for remote Flux Klein text encoding.
    Prepares payload, invokes network client, and returns ComfyUI conditioning tuple.
    """
    payload = {
        "prompt": text if text is not None else "",
    }

    route = f"/v1/encode/{model_name}"
    result_data = send_binary_encode_request(server_url, route, payload)

    if "conditioning" not in result_data:
        raise ValueError(
            f"[MacOffload] Malformed response from '{route}': missing 'conditioning' key."
        )

    conditioning = result_data["conditioning"]
    return (conditioning,)
