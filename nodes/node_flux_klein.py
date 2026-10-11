from ._flux_klein_helpers import process_flux_klein_encode


class RemoteCLIPTextEncode:
    """
    Remote CLIP Text Encoder node for Flux Klein models.
    Delegates tokenization and embedding forward pass to macOS (Apple Silicon MPS)
    via binary tensor streams, saving local GPU VRAM.
    """

    DESCRIPTION = (
        "Offloads Flux.2 Klein (4B / 9B) text encoding to a remote macOS server running "
        "on Apple Silicon Unified Memory (MPS), saving local GPU VRAM on 8GB cards."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8288",
                        "multiline": False,
                        "tooltip": "Base URL of the remote macOS server (e.g. http://192.168.1.100:8288).",
                    },
                ),
                "model_name": (
                    [
                        "flux2-klein-4b-uncensored-text-encoder",
                        "flux2-klein-9b-uncensored-text-encoder",
                    ],
                    {
                        "default": "flux2-klein-4b-uncensored-text-encoder",
                        "tooltip": "Model folder name on remote Mac server. Click 'Fetch Models' to query models from server.",
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Text prompt to encode into conditioning embeddings.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "⚡ MacOffload/Conditioning"

    def encode(self, server_url, model_name, text):
        return process_flux_klein_encode(server_url, model_name, text)


NODE_CLASS_MAPPINGS = {
    "MACOFFLOAD_RemoteCLIPTextEncode": RemoteCLIPTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MACOFFLOAD_RemoteCLIPTextEncode": "Remote CLIP Text Encode (Flux Klein) ⚡",
}
