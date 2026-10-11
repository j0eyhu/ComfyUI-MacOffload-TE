import io
from typing import Any, Dict, Tuple
import requests
import torch


def send_binary_encode_request(
    server_url: str,
    route: str,
    payload: Dict[str, Any],
    timeout: Tuple[int, int] = (10, 300),
) -> Dict[str, Any]:
    """
    Serializes a dictionary payload into a PyTorch binary stream and sends it
    to the remote macOS server via HTTP POST.

    Args:
        server_url: Base server URL, e.g. "http://192.168.1.100:8288"
        route: API route path, e.g. "/v1/encode/flux_klein_4b"
        payload: Dictionary containing prompts and optional vision tensors
        timeout: Tuple of (connect_timeout, read_timeout) in seconds

    Returns:
        Deserialized dictionary containing conditioning and latent tensors.
    """
    clean_base = server_url.strip().rstrip("/")
    clean_route = "/" + route.strip().lstrip("/")
    endpoint = f"{clean_base}{clean_route}"

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    binary_data = buffer.getvalue()

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
            f"  2. Is the IP and port '{server_url}' correct and reachable across your LAN?\n"
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
        result_data = torch.load(io.BytesIO(response.content), weights_only=False)
    except Exception as err:
        raise RuntimeError(
            f"[MacOffload] Failed to deserialize tensor response from '{endpoint}': {err}"
        ) from err

    return result_data
