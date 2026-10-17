from ._flux_klein_helpers import process_flux_klein_encode


class RemoteCLIPTextEncode:
    """
    Remote CLIP Text Encoder node for Flux Klein (Flux.2 & Flux.3) models.
    Delegates tokenization and embedding forward pass to macOS (Apple Silicon MPS)
    via safe safetensors streams, saving local GPU VRAM.
    """

    DESCRIPTION = (
        "Offloads Flux.2 Klein (4B / 9B DiT) and Flux.3 Klein (5.6B DiT, Qwen3-VL-4B TE) text encoding "
        "to a remote macOS server running on Apple Silicon Unified Memory (MPS), saving local GPU VRAM on 8GB cards."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    [
                        "flux2-klein-4b-uncensored-text-encoder",
                        "flux2-klein-9b-uncensored-text-encoder",
                        "Qwen3-VL-4b-Heretic",
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

    def encode(self, model_name: str, text: str, **kwargs):
        # kwargs gracefully swallows legacy server_url widget from older saved workflows
        return process_flux_klein_encode(model_name=model_name, text=text)


NODE_CLASS_MAPPINGS = {
    "MACOFFLOAD_RemoteCLIPTextEncode": RemoteCLIPTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MACOFFLOAD_RemoteCLIPTextEncode": "Remote CLIP Text Encode (Flux Klein) ⚡",
}
