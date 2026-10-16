"""
Internal ComfyUI backend API routes for MacOffload.
Proxies model listing using the saved disk configuration and manages local server URL settings.
"""
import asyncio
from ._config import get_server_url, set_server_url

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

    @routes.get("/macoffload/config")
    async def get_config_handler(request):
        """
        Returns the saved Mac server URL from local disk.
        """
        return web.json_response({"server_url": get_server_url()})

    @routes.post("/macoffload/config")
    async def set_config_handler(request):
        """
        Updates the Mac server URL on local disk.
        """
        try:
            data = await request.json()
            new_url = str(data.get("server_url", "")).strip()
            if not new_url:
                return web.json_response(
                    {"error": "server_url field is required"}, status=400
                )
            saved_url = set_server_url(new_url)
            return web.json_response({"status": "ok", "server_url": saved_url})
        except Exception as err:
            return web.json_response({"error": str(err)}, status=500)

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
            # Filter strictly to model names only to avoid exposing server system paths
            models = raw_data.get("models", [])
            if not isinstance(models, list):
                models = []

            return web.json_response({"models": models})
        except requests.exceptions.RequestException as err:
            err_msg = str(err)
            if "127.0.0.1" in target_url or "localhost" in target_url:
                err_msg += (
                    " (Target is set to 127.0.0.1/localhost. If te-server is running "
                    "on another Mac, please configure your Mac's LAN address in Settings)"
                )
            return web.json_response(
                {
                    "error": f"Failed to connect to Mac server at '{target_url}': {err_msg}",
                    "models": [],
                },
                status=502,
            )
