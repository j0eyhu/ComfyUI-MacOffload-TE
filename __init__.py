"""
ComfyUI-MacOffload-TE
Offloads heavy Text Encoders (Flux Klein, Qwen Image 2.1) to remote macOS (Apple Silicon MPS).
"""
import importlib
import traceback

NODE_MODULES = [
    "node_flux_klein",
    "node_qwen_image",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
_failed_modules = []

for _module_name in NODE_MODULES:
    try:
        _module = importlib.import_module(f".nodes.{_module_name}", __name__)
        NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(
            getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {})
        )
    except Exception:
        _failed_modules.append(_module_name)
        traceback.print_exc()

try:
    from .nodes._api_routes import register_macoffload_routes
    register_macoffload_routes()
except Exception:
    pass

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

if _failed_modules:
    print(f"[MacOffload] [WARNING] Failed to load modules: {', '.join(_failed_modules)}")
else:
    print(f"[MacOffload] Initialized {len(NODE_CLASS_MAPPINGS)} remote text encoder nodes.")
