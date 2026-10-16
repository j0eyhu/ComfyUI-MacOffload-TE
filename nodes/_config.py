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


def save_config(config: Dict[str, Any]) -> None:
    """
    Persists configuration dictionary to local disk.
    """
    config_path = get_config_file_path()
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[MacOffload] [Error] Failed to save {config_path}: {e}")
        raise


def get_server_url() -> str:
    """
    Convenience helper to retrieve the sanitized server URL from disk / env.
    """
    return load_config().get("server_url", DEFAULT_SERVER_URL)


def set_server_url(server_url: str) -> str:
    """
    Updates the server URL setting on disk and returns the sanitized URL.
    """
    clean_url = (server_url or "").strip().rstrip("/")
    if not clean_url:
        clean_url = DEFAULT_SERVER_URL

    if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        clean_url = f"http://{clean_url}"

    config = load_config()
    config["server_url"] = clean_url
    save_config(config)
    return clean_url
