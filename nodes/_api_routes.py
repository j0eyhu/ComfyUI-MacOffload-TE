"""
Internal ComfyUI backend API routes for MacOffload.
Proxies cross-origin queries to avoid browser CORS and LAN Private Network restrictions.
"""
import asyncio
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
        server_url = str(request.rel_url.query.get("server_url", "")).strip().rstrip("/")
        if not server_url:
            return web.json_response({"error": "server_url query param is required"}, status=400)

        target_url = f"{server_url}/v1/models"
        try:
            # Query remote Mac server in worker thread to prevent event loop blocking
            def _fetch():
                resp = requests.get(target_url, timeout=5)
                resp.raise_for_status()
                return resp.json()

            data = await asyncio.to_thread(_fetch)
            return web.json_response(data)
        except requests.exceptions.RequestException as err:
            err_msg = str(err)
            if "127.0.0.1" in target_url or "localhost" in target_url:
                err_msg += " (Target is 127.0.0.1/localhost. If te-server is running on Mac, please enter your Mac's IP/hostname)"
            return web.json_response(
                {"error": f"Failed to connect to Mac server at '{target_url}': {err_msg}"},
                status=502,
            )
