# config.py
import os
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base Server Network Binding
HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
PORT = int(os.getenv("BRIDGE_PORT", 8288))

# Dynamic Memory & Pool Management
MAX_MODELS = int(os.getenv("MAX_MODELS", "2"))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "0"))

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

# Root Directory for Encoders
DEFAULT_MODELS_DIR = os.getenv("MODELS_DIR", os.path.expanduser("~/models"))
MODELS_DIR = os.path.abspath(DEFAULT_MODELS_DIR)

# Model Directory Names
QWEN_FOLDER_NAME = "Qwen-Image-2.1-Text-Encoder-Heretic"
FLUX_KLEIN_4B_FOLDER_NAME = "flux2-klein-4b-uncensored-text-encoder"
FLUX_KLEIN_9B_FOLDER_NAME = "flux2-klein-9b-uncensored-text-encoder"

# Individual Model Paths (allow individual env overrides)
QWEN_MODEL_PATH = os.path.abspath(
    os.path.expanduser(
        os.getenv("QWEN_MODEL_PATH", os.path.join(MODELS_DIR, QWEN_FOLDER_NAME))
    )
)
FLUX_KLEIN_4B_PATH = os.path.abspath(
    os.path.expanduser(
        os.getenv("FLUX_KLEIN_4B_PATH", os.path.join(MODELS_DIR, FLUX_KLEIN_4B_FOLDER_NAME))
    )
)
FLUX_KLEIN_9B_PATH = os.path.abspath(
    os.path.expanduser(
        os.getenv("FLUX_KLEIN_9B_PATH", os.path.join(MODELS_DIR, FLUX_KLEIN_9B_FOLDER_NAME))
    )
)


def update_runtime_config(
    host: str = None,
    port: int = None,
    max_models: int = None,
    idle_timeout: int = None,
    device: str = None,
    models_dir: str = None,
) -> None:
    """Updates runtime configuration with CLI overrides."""
    global HOST, PORT, MAX_MODELS, IDLE_TIMEOUT, DEVICE, DTYPE, MODELS_DIR
    global QWEN_MODEL_PATH, FLUX_KLEIN_4B_PATH, FLUX_KLEIN_9B_PATH

    if host is not None:
        HOST = host
    if port is not None:
        PORT = port
    if max_models is not None:
        MAX_MODELS = max(1, max_models)
    if idle_timeout is not None:
        IDLE_TIMEOUT = max(0, idle_timeout)
    if device is not None:
        DEVICE = device
        DTYPE = torch.bfloat16 if DEVICE in ("mps", "cuda") else torch.float32

    if models_dir is not None:
        MODELS_DIR = os.path.abspath(os.path.expanduser(models_dir))
        if "QWEN_MODEL_PATH" not in os.environ:
            QWEN_MODEL_PATH = os.path.join(MODELS_DIR, QWEN_FOLDER_NAME)
        if "FLUX_KLEIN_4B_PATH" not in os.environ:
            FLUX_KLEIN_4B_PATH = os.path.join(MODELS_DIR, FLUX_KLEIN_4B_FOLDER_NAME)
        if "FLUX_KLEIN_9B_PATH" not in os.environ:
            FLUX_KLEIN_9B_PATH = os.path.join(MODELS_DIR, FLUX_KLEIN_9B_FOLDER_NAME)
