import json
import struct
from typing import Any, Dict, Optional, Tuple
import requests
import torch
import safetensors.torch
from ._config import get_server_url


def serialize_safetensors_payload(
    tensors: Optional[Dict[str, torch.Tensor]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Serializes pure tensors and JSON metadata into a safe, non-executable
    safetensors binary buffer.
    """
    tensors = tensors or {}
    meta = meta or {}

    # Standardize tensors: must be contiguous tensors on CPU
    clean_tensors = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            clean_tensors[k] = v.detach().cpu().contiguous()

    # safetensors header metadata requires string values; wrap metadata dictionary in JSON string
    wrapped_meta = {"meta": json.dumps(meta)}
    return safetensors.torch.save(clean_tensors, metadata=wrapped_meta)


def deserialize_safetensors_payload(
    buf: bytes,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Safely deserializes a safetensors binary buffer into pure numerical tensors
    and JSON metadata. Incapable of executing arbitrary code.
    """
    if len(buf) < 8:
        raise ValueError("Buffer too small to be a valid safetensors stream.")

    # Read the 8-byte uint64 header size
    header_size = struct.unpack("<Q", buf[:8])[0]
    if len(buf) < 8 + header_size:
        raise ValueError("Malformed safetensors buffer: truncated header.")

    header_raw = json.loads(buf[8 : 8 + header_size].decode("utf-8"))
    raw_meta = header_raw.get("__metadata__", {})
    meta = json.loads(raw_meta.get("meta", "{}"))

    tensors = safetensors.torch.load(buf)
    return tensors, meta


def send_safetensors_request(
    route: str,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
    meta: Optional[Dict[str, Any]] = None,
    timeout: Tuple[int, int] = (10, 300),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Sends pure numerical tensors and metadata to the remote macOS server via HTTP POST
    using the safetensors format. The destination server address is resolved strictly
    from local disk configuration or the MACOFFLOAD_SERVER_URL environment variable.

    Args:
        route: API route path, e.g. "/v1/encode/flux2-klein-4b-uncensored-text-encoder"
        tensors: Optional dictionary of named torch.Tensor inputs (e.g. vision references)
        meta: Optional dictionary of JSON-serializable parameters (prompts, options)
        timeout: Tuple of (connect_timeout, read_timeout) in seconds

    Returns:
        Tuple of (tensors_dict, metadata_dict) safely decoded from response.
    """
    base_url = get_server_url().strip().rstrip("/")
    clean_route = "/" + route.strip().lstrip("/")
    endpoint = f"{base_url}{clean_route}"

    binary_data = serialize_safetensors_payload(tensors=tensors, meta=meta)

    try:
        response = requests.post(
            endpoint,
            data=binary_data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as err:
        raise RuntimeError(
            f"[MacOffload] Failed to connect to server at '{endpoint}'.\n"
            f"Diagnostics:\n"
            f"  1. Is the Mac server running? (python server.py)\n"
            f"  2. Is the IP and port '{base_url}' correct and reachable across your LAN?\n"
            f"  3. Check if macOS firewall or router is blocking port {endpoint.split(':')[-1].split('/')[0]}.\n"
            f"Details: {err}"
        ) from err
    except requests.exceptions.Timeout as err:
        raise RuntimeError(
            f"[MacOffload] Request to '{endpoint}' timed out after {timeout[1]}s.\n"
            f"Diagnostics:\n"
            f"  1. If this is the first execution, the model might still be loading into Unified Memory.\n"
            f"  2. Check the Mac server terminal for model loading progress.\n"
            f"Details: {err}"
        ) from err
    except requests.exceptions.HTTPError as err:
        err_detail = response.text
        try:
            json_err = response.json()
            if "detail" in json_err:
                err_detail = json_err["detail"]
        except Exception:
            pass
        raise RuntimeError(
            f"[MacOffload] Mac server returned HTTP {response.status_code} for '{endpoint}':\n"
            f"  Detail: {err_detail}"
        ) from err
    except requests.exceptions.RequestException as err:
        raise RuntimeError(
            f"[MacOffload] Request to '{endpoint}' failed: {err}"
        ) from err

    try:
        res_tensors, res_meta = deserialize_safetensors_payload(response.content)
    except Exception as err:
        raise RuntimeError(
            f"[MacOffload] Failed to deserialize safetensors response from '{endpoint}': {err}"
        ) from err

    return res_tensors, res_meta
