import json
import os
from pathlib import Path
from typing import Dict, Any

DEFAULT_SERVER_URL = "http://127.0.0.1:8288"
CONFIG_FILE_NAME = "macoffload_config.json"


def get_config_file_path() -> Path:
    """
    Returns the path to the local configuration file stored inside the custom node directory.
    """
    node_dir = Path(__file__).resolve().parent.parent
    return node_dir / CONFIG_FILE_NAME


def load_config() -> Dict[str, Any]:
    """
    Reads the local configuration file from disk.
    If the file does not exist, returns defaults.
    """
    config_path = get_config_file_path()
    config = {
        "server_url": DEFAULT_SERVER_URL,
    }

    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    config.update(data)
        except Exception as e:
            print(f"[MacOffload] [Warning] Failed to read {config_path}: {e}")

    # Environment variable overrides file setting if explicitly set
    env_url = os.environ.get("MACOFFLOAD_SERVER_URL")
    if env_url and env_url.strip():
        config["server_url"] = env_url.strip()

    # Ensure URL is clean
    if "server_url" in config and isinstance(config["server_url"], str):
        config["server_url"] = config["server_url"].strip().rstrip("/")
    else:
        config["server_url"] = DEFAULT_SERVER_URL

    return config


def get_server_url() -> str:
    """
    Convenience helper to retrieve the sanitized server URL from disk / env.
    Address is read-only at runtime; changes are made strictly by editing
    macoffload_config.json or setting the MACOFFLOAD_SERVER_URL environment variable.
    """
    return load_config().get("server_url", DEFAULT_SERVER_URL)
