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
from encoders.flux_klein import FluxKleinEncoder
from encoders.flux3_klein import Flux3KleinEncoder

# Mapping of detected model type strings to their respective encoder implementations
ENCODER_CLASSES: Dict[str, Type[BaseEncoder]] = {
    "flux_klein": FluxKleinEncoder,
    "flux3_klein": Flux3KleinEncoder,
    "qwen_image": QwenImage21Encoder,
}


def detect_model_type(folder_path: str) -> str:
    """
    Infers whether a model directory corresponds to FLUX.2 Klein, FLUX.3 Klein, or Qwen Image
    by inspecting config.json or heuristic folder names.

    Model Distinctions:
      - FLUX.2 Klein: DiT variants (4B / 9B) with Qwen3 CausalLM text encoders.
      - FLUX.3 Klein: Single 5.6B DiT with Qwen3-VL-4B text encoder (hidden_size=2560).
      - Qwen Image: Multimodal Qwen3-VL text/image encoder (hidden_size=4096).
    """
    base_lower = os.path.basename(folder_path).lower()

    # 1. Explicit FLUX.3 / Qwen3-VL-4B folder indicators take precedence
    if "flux3" in base_lower or "flux-3" in base_lower:
        return "flux3_klein"
    if "qwen3" in base_lower and "4b" in base_lower and ("vl" in base_lower or "heretic" in base_lower):
        return "flux3_klein"

    # 2. Inspect config.json
    config_file = os.path.join(folder_path, "config.json")
    if os.path.isfile(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            archs = [str(a).lower() for a in data.get("architectures", [])]
            mtype = str(data.get("model_type", "")).lower()

            # Qwen3-VL: Distinguish FLUX.3 Klein (4B text encoder, hidden_size 2560)
            # from Qwen-Image (8B text encoder, hidden_size 4096)
            if any("qwen3_vl" in a or "qwen3vl" in a for a in archs) or mtype in ("qwen3_vl", "qwen3vl"):
                hidden_size = data.get("hidden_size") or data.get("text_config", {}).get("hidden_size")
                if hidden_size == 2560:
                    return "flux3_klein"
                elif hidden_size in (4096, 8192):
                    return "qwen_image"
                else:
                    return "flux3_klein" if ("4b" in base_lower or "flux" in base_lower or "klein" in base_lower) else "qwen_image"

            # Other VL models (Qwen2-VL, Qwen2.5-VL, ImageText) belong to qwen_image
            if any("qwen2_5_vl" in a or "qwen2vl" in a or "imagetext" in a for a in archs) or "vl" in mtype:
                return "qwen_image"

            # CausalLM / Qwen text models -> FLUX.2 Klein
            if any("causallm" in a or "qwen" in a or "flux" in a for a in archs) or "qwen" in mtype:
                return "flux_klein"
        except Exception:
            pass

    # 3. Fallback heuristic based on folder name
    if "image" in base_lower:
        return "qwen_image"
    if ("qwen3" in base_lower and "4b" in base_lower) or "flux3" in base_lower or "flux-3" in base_lower:
        return "flux3_klein"
    if "qwen" in base_lower and ("vl" in base_lower or "heretic" in base_lower):
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

    def __init__(
        self,
        max_models: Optional[int] = None,
        idle_timeout: Optional[int] = None,
        settling_delay: Optional[float] = None,
    ):
        self.max_models = max(1, max_models if max_models is not None else config.MAX_MODELS)
        self.idle_timeout = max(0, idle_timeout if idle_timeout is not None else config.IDLE_TIMEOUT)
        self.settling_delay = max(0.0, settling_delay if settling_delay is not None else config.SETTLING_DELAY)
        self.pool: Dict[str, ManagedModel] = {}
        self.discovered_models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None
        self.scan_models()

    def scan_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Scans config.MODELS_DIR and discovers available encoder model folders.
        No hardcoded paths or dummy entries needed.
        """
        discovered: Dict[str, Dict[str, Any]] = {}
        models_dir = config.MODELS_DIR

        if not os.path.isdir(models_dir):
            print(f"[ModelManager] Warning: MODELS_DIR '{models_dir}' does not exist or is not a directory.")
            self.discovered_models = discovered
            return discovered

        for entry in sorted(os.listdir(models_dir)):
            full_path = os.path.join(models_dir, entry)
            if os.path.isdir(full_path):
                # Check if directory contains model weights or configuration
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

        print(f"[ModelManager] Scanned '{models_dir}': found {len(discovered)} model(s) {list(discovered.keys())}")
        self.discovered_models = discovered
        return discovered

    def get_model_list(self) -> List[str]:
        """Returns clean list of discovered model folder names."""
        self.scan_models()
        return list(self.discovered_models.keys())

    async def get_encoder(self, model_name: str) -> BaseEncoder:
        """
        Retrieves a loaded encoder from the pool or loads it dynamically.
        Resolves model folder from dynamically discovered models in MODELS_DIR.
        """
        model_name = os.path.basename(model_name.strip())
        async with self.lock:
            # Refresh if not found in cache
            if model_name not in self.discovered_models:
                self.scan_models()

            if model_name not in self.discovered_models:
                available = list(self.discovered_models.keys())
                raise ValueError(
                    f"Model folder '{model_name}' not found on server. "
                    f"Available options: {available}"
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
                # Memory Settling Window: Allow macOS Mach kernel VM / Metal daemon
                # to complete asynchronous page unmapping before high-throughput SSD reading begins.
                if self.settling_delay > 0:
                    await asyncio.sleep(self.settling_delay)

            # 3. Cache miss: instantiate and load dynamically
            info = self.discovered_models[model_name]
            m_type = info["type"]
            m_path = info["path"]
            encoder_cls = ENCODER_CLASSES.get(m_type)
            if not encoder_cls:
                raise ValueError(f"No encoder implementation registered for model type '{m_type}'.")

            print(f"[ModelManager] Dynamically loading '{model_name}' (type={m_type}, class={encoder_cls.__name__}) on {config.DEVICE}...")
            encoder = encoder_cls(model_path=m_path)

            await asyncio.to_thread(encoder.load)
            self.pool[model_name] = ManagedModel(model_name, encoder)
            return encoder

    def _evict_model_unlocked(self, model_name: str) -> None:
        """
        Internal eviction helper with deterministic multi-tier memory purging.
        """
        if model_name in self.pool:
            item = self.pool.pop(model_name)

            # Step 1: Ensure hardware queues finish executing pending operations before deallocation
            if config.DEVICE == "mps" and hasattr(torch, "mps"):
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
            elif config.DEVICE == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

            # Step 2: Explicitly dismantle encoder attributes while device is idle
            try:
                if hasattr(item.encoder, "unload"):
                    item.encoder.unload()
            except Exception as e:
                print(f"[ModelManager] Warning during '{model_name}' unload: {e}")

            del item.encoder
            del item

            # Step 3: Multi-generation Python GC
            gc.collect()
            gc.collect()

            # Step 4: Empty PyTorch allocator caches
            if config.DEVICE == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif config.DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[ModelManager] Model '{model_name}' successfully evicted and memory reclaimed.")

    async def evict_model(self, model_name: str) -> None:
        """
        External eviction interface with lock protection.
        """
        model_name = os.path.basename(model_name.strip())
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
                await asyncio.sleep(config.WATCHDOG_INTERVAL)
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
