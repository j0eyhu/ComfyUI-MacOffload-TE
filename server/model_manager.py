# model_manager.py
import asyncio
import gc
import json
import os
import time
from typing import Any, Dict, List, Optional, Type
import torch

import config
from encoders.base import BaseEncoder
from encoders.qwen_image import QwenImage21Encoder
from encoders.flux_klein import FluxKleinEncoder, FluxKlein4BEncoder, FluxKlein9BEncoder

# Static fallback registry of supported encoder classes
ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {
    config.QWEN_FOLDER_NAME: QwenImage21Encoder,
    config.FLUX_KLEIN_4B_FOLDER_NAME: FluxKlein4BEncoder,
    config.FLUX_KLEIN_9B_FOLDER_NAME: FluxKlein9BEncoder,
}


def detect_model_type(folder_path: str) -> str:
    """
    Infers whether a model directory corresponds to Flux Klein or Qwen Image
    by inspecting config.json or heuristic folder names.
    """
    config_file = os.path.join(folder_path, "config.json")
    if os.path.isfile(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            archs = [str(a).lower() for a in data.get("architectures", [])]
            mtype = str(data.get("model_type", "")).lower()

            if any("qwen2_5_vl" in a or "qwen2vl" in a or "imagetext" in a for a in archs) or "vl" in mtype:
                return "qwen_image"
            if any("causallm" in a or "qwen" in a or "flux" in a for a in archs) or "qwen" in mtype:
                return "flux_klein"
        except Exception:
            pass

    # Fallback heuristic based on folder name
    base_lower = os.path.basename(folder_path).lower()
    if "qwen" in base_lower and ("image" in base_lower or "vl" in base_lower or "heretic" in base_lower):
        return "qwen_image"
    if "flux" in base_lower or "klein" in base_lower:
        return "flux_klein"

    return "flux_klein"


class ManagedModel:
    """
    Wrapper for resident encoder instances with access timestamp tracking.
    """

    def __init__(self, name: str, encoder: BaseEncoder):
        self.name = name
        self.encoder = encoder
        self.last_used: float = time.time()

    def touch(self) -> None:
        self.last_used = time.time()


class DynamicModelManager:
    """
    Pool manager handling on-demand model lifecycle, dynamic folder discovery,
    LRU eviction, and watchdog cleanup.
    """

    def __init__(self, max_models: int = 2, idle_timeout: int = 0):
        self.max_models = max(1, max_models)
        self.idle_timeout = idle_timeout
        self.pool: Dict[str, ManagedModel] = {}
        self.discovered_models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None
        self.scan_models()

    def scan_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Scans config.MODELS_DIR and discovers available encoder model folders.
        Also registers configured defaults and shorthand aliases.
        """
        discovered: Dict[str, Dict[str, Any]] = {}

        # 1. Scan subdirectories in MODELS_DIR
        models_dir = config.MODELS_DIR
        if os.path.isdir(models_dir):
            for entry in sorted(os.listdir(models_dir)):
                full_path = os.path.join(models_dir, entry)
                if os.path.isdir(full_path):
                    # Check if directory contains model files
                    has_weights = any(
                        f.endswith((".safetensors", ".bin", ".json"))
                        for f in os.listdir(full_path)
                    ) if os.path.exists(full_path) else False
                    if has_weights:
                        m_type = detect_model_type(full_path)
                        discovered[entry] = {
                            "name": entry,
                            "path": full_path,
                            "type": m_type,
                        }

        # 2. Register configured model paths if present (by folder name only)
        for cfg_path, m_type in [
            (config.FLUX_KLEIN_4B_PATH, "flux_klein"),
            (config.FLUX_KLEIN_9B_PATH, "flux_klein"),
            (config.QWEN_MODEL_PATH, "qwen_image"),
        ]:
            if cfg_path and os.path.exists(cfg_path):
                folder_name = os.path.basename(cfg_path)
                if folder_name not in discovered:
                    discovered[folder_name] = {
                        "name": folder_name,
                        "path": cfg_path,
                        "type": m_type,
                    }

        # 3. Fallback defaults by folder name if models dir is empty
        for default_name, cfg_path, m_type in [
            (config.FLUX_KLEIN_4B_FOLDER_NAME, config.FLUX_KLEIN_4B_PATH, "flux_klein"),
            (config.FLUX_KLEIN_9B_FOLDER_NAME, config.FLUX_KLEIN_9B_PATH, "flux_klein"),
            (config.QWEN_FOLDER_NAME, config.QWEN_MODEL_PATH, "qwen_image"),
        ]:
            if default_name not in discovered:
                discovered[default_name] = {
                    "name": default_name,
                    "path": cfg_path,
                    "type": m_type,
                }

        self.discovered_models = discovered
        return discovered

    def get_model_list(self) -> List[str]:
        """Returns clean list of model keys (preferring folder names)."""
        self.scan_models()
        # Sort so real folder names come first, aliases last
        items = list(self.discovered_models.keys())
        return items

    async def get_encoder(self, model_name: str) -> BaseEncoder:
        """
        Retrieves a loaded encoder from the pool or loads it dynamically.
        Supports both exact folder names and legacy shorthand aliases.
        """
        async with self.lock:
            # Refresh if not found
            if model_name not in self.discovered_models:
                self.scan_models()

            if model_name not in self.discovered_models and model_name not in ENCODER_REGISTRY:
                available = list(self.discovered_models.keys())
                raise ValueError(
                    f"Unsupported model_name: '{model_name}'. Available options on server: {available}"
                )

            # 1. Cache hit: update last used timestamp and return
            if model_name in self.pool:
                self.pool[model_name].touch()
                return self.pool[model_name].encoder

            # 2. Capacity check: evict least recently used (LRU) model
            if len(self.pool) >= self.max_models:
                lru_name = min(self.pool.keys(), key=lambda k: self.pool[k].last_used)
                print(
                    f"[ModelManager] Capacity limit ({self.max_models}) reached. "
                    f"Evicting LRU model '{lru_name}'..."
                )
                self._evict_model_unlocked(lru_name)

            # 3. Cache miss: instantiate and load dynamically
            info = self.discovered_models.get(model_name)
            if info:
                m_type = info["type"]
                m_path = info["path"]
                print(f"[ModelManager] Dynamically loading '{model_name}' (type={m_type}, path='{m_path}') on {config.DEVICE}...")
                if m_type == "flux_klein":
                    encoder = FluxKleinEncoder(model_path=m_path)
                else:
                    encoder = QwenImage21Encoder(model_path=m_path)
            else:
                print(f"[ModelManager] Dynamically loading fallback '{model_name}' on {config.DEVICE}...")
                encoder_cls = ENCODER_REGISTRY[model_name]
                encoder = encoder_cls()

            await asyncio.to_thread(encoder.load)
            self.pool[model_name] = ManagedModel(model_name, encoder)
            return encoder

    def _evict_model_unlocked(self, model_name: str) -> None:
        """
        Internal eviction helper (must be called with self.lock held).
        """
        if model_name in self.pool:
            item = self.pool.pop(model_name)
            del item.encoder
            del item
            gc.collect()
            if config.DEVICE == "mps":
                torch.mps.empty_cache()
            print(f"[ModelManager] Model '{model_name}' successfully evicted and memory reclaimed.")

    async def evict_model(self, model_name: str) -> None:
        """
        External eviction interface with lock protection.
        """
        async with self.lock:
            self._evict_model_unlocked(model_name)

    async def unload_all(self) -> None:
        """
        Unloads all resident models and clears Unified Memory.
        """
        async with self.lock:
            keys = list(self.pool.keys())
            for key in keys:
                self._evict_model_unlocked(key)
            self.pool.clear()
            gc.collect()
            if config.DEVICE == "mps":
                torch.mps.empty_cache()
            print("[ModelManager] All models unloaded from pool.")

    async def _watchdog_loop(self) -> None:
        """
        Background loop to periodically evict models exceeding idle_timeout.
        """
        if self.idle_timeout <= 0:
            return

        while True:
            try:
                await asyncio.sleep(30)
                async with self.lock:
                    now = time.time()
                    expired = [
                        name
                        for name, item in self.pool.items()
                        if now - item.last_used > self.idle_timeout
                    ]
                    for name in expired:
                        print(
                            f"[ModelManager] Watchdog: Model '{name}' idle for "
                            f">{self.idle_timeout}s. Evicting..."
                        )
                        self._evict_model_unlocked(name)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ModelManager] Watchdog loop warning: {e}")

    def start_watchdog(self) -> None:
        """
        Starts the watchdog background task if timeout is configured.
        """
        if self._watchdog_task is None and self.idle_timeout > 0:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    def stop_watchdog(self) -> None:
        """
        Cancels the watchdog background task.
        """
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            self._watchdog_task = None
