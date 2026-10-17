"""
Internal ComfyUI backend API routes for MacOffload.
Proxies model listing from the remote Mac server using the local disk configuration.
"""
import asyncio
from ._config import get_server_url

try:
    from aiohttp import web
    from server import PromptServer
    import requests
except Exception:
    PromptServer = None


def register_macoffload_routes():
    if PromptServer is None or not hasattr(PromptServer, "instance"):
        return
    if getattr(PromptServer.instance, "_macoffload_routes_registered", False):
        return
    PromptServer.instance._macoffload_routes_registered = True

    routes = PromptServer.instance.routes

    @routes.get("/macoffload/models")
    async def get_models_proxy(request):
        """
        Queries the remote macOS server using only the saved server address on disk.
        Returns strictly the list of model names.
        """
        saved_server_url = get_server_url()
        target_url = f"{saved_server_url}/v1/models"

        try:
            # Query remote Mac server in worker thread to prevent event loop blocking
            def _fetch():
                resp = requests.get(target_url, timeout=5)
                resp.raise_for_status()
                return resp.json()

            raw_data = await asyncio.to_thread(_fetch)
            models = raw_data.get("models", [])
            if not isinstance(models, list):
                models = []
            model_types = raw_data.get("model_types", {})
            if not isinstance(model_types, dict):
                model_types = {}

            return web.json_response({
                "models": models,
                "model_types": model_types,
            })
        except requests.exceptions.RequestException as err:
            err_msg = str(err)
            if "127.0.0.1" in target_url or "localhost" in target_url:
                err_msg += (
                    " (Target is set to 127.0.0.1/localhost. If te-server is running "
                    "on another Mac, please configure your Mac's LAN address in macoffload_config.json or MACOFFLOAD_SERVER_URL)"
                )
            return web.json_response(
                {
                    "error": f"Failed to connect to Mac server at '{target_url}': {err_msg}",
                    "models": [],
                },
                status=502,
            )
