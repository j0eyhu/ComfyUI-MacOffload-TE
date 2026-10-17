# config.py
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Prevent PyTorch MPS allocator from hanging/throttling on unified memory during model swaps
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch

# Base Server Network Binding
HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
PORT = int(os.getenv("BRIDGE_PORT", 8288))

# Dynamic Memory & Pool Management
MAX_MODELS = int(os.getenv("MAX_MODELS", "1"))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "0"))
SETTLING_DELAY = float(os.getenv("SETTLING_DELAY", "1.0"))
WATCHDOG_INTERVAL = int(os.getenv("WATCHDOG_INTERVAL", "30"))

# Hardware Acceleration Setup
def detect_device() -> str:
    env_dev = os.getenv("BRIDGE_DEVICE")
    if env_dev:
        return env_dev
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = detect_device()
DTYPE = torch.bfloat16 if DEVICE in ("mps", "cuda") else torch.float32

# Root Directory for Encoders (All models reside in or are symlinked into MODELS_DIR)
# Always expand ~ on both environment variable input and default
MODELS_DIR = os.path.abspath(os.path.expanduser(os.getenv("MODELS_DIR", "~/models")))


def update_runtime_config(
    host: str = None,
    port: int = None,
    max_models: int = None,
    idle_timeout: int = None,
    device: str = None,
    models_dir: str = None,
    settling_delay: float = None,
    watchdog_interval: int = None,
) -> None:
    """Updates runtime configuration with CLI overrides."""
    global HOST, PORT, MAX_MODELS, IDLE_TIMEOUT, DEVICE, DTYPE, MODELS_DIR
    global SETTLING_DELAY, WATCHDOG_INTERVAL

    if host is not None:
        HOST = host
    if port is not None:
        PORT = port
    if max_models is not None:
        MAX_MODELS = max(1, max_models)
    if idle_timeout is not None:
        IDLE_TIMEOUT = max(0, idle_timeout)
    if settling_delay is not None:
        SETTLING_DELAY = max(0.0, settling_delay)
    if watchdog_interval is not None:
        WATCHDOG_INTERVAL = max(1, watchdog_interval)
    if device is not None:
        DEVICE = device
        DTYPE = torch.bfloat16 if DEVICE in ("mps", "cuda") else torch.float32
    if models_dir is not None:
        MODELS_DIR = os.path.abspath(os.path.expanduser(models_dir))
