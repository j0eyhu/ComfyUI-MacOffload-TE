from ._qwen_image_helpers import process_qwen_image_encode


class RemoteTextEncodeQwenImage21:
    """
    Remote Text & Vision Encoder node for Qwen-Image-2.1.
    Delegates heavy vision-language model inference to macOS (Apple Silicon MPS)
    via safe safetensors streams, avoiding VRAM swapping on local Windows GPU.
    """

    DESCRIPTION = (
        "Encodes text prompts and multi-image references for Qwen-Image-2.1 models using a remote "
        "macOS server running on Apple Silicon Unified Memory. Outputs positive/negative "
        "conditioning and a WanVAE-compatible 64-channel latent tensor."
    )

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_name": (
                    [
                        "Qwen-Image-2.1-Text-Encoder-Heretic",
                    ],
                    {
                        "default": "Qwen-Image-2.1-Text-Encoder-Heretic",
                        "tooltip": "Model folder name on remote Mac server. Click 'Fetch Models' to query models from server.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Positive prompt describing the desired image generation.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Negative prompt describing unwanted attributes.",
                    },
                ),
                "resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Reference images are resized to about resolution x resolution pixels, "
                        "at multiples of 32, preserving aspect ratio. 0 keeps each reference at its own size.",
                    },
                ),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {
                        "tooltip": "Optional WanVAE model used to encode reference images into reference latents."
                    },
                ),
            },
        }

        for i in range(1, 17):
            inputs["optional"][f"image_{i}"] = (
                "IMAGE",
                {"tooltip": f"Optional reference image #{i} for multimodal image-to-image conditioning."},
            )

        return inputs

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "⚡ MacOffload/Conditioning"

    def execute(
        self,
        prompt: str,
        negative_prompt: str,
        model_name: str = "Qwen-Image-2.1-Text-Encoder-Heretic",
        vae=None,
        resolution: int = 1024,
        **kwargs,
    ):
        return process_qwen_image_encode(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            vae=vae,
            resolution=resolution,
            **kwargs,
        )


NODE_CLASS_MAPPINGS = {
    "MACOFFLOAD_RemoteTextEncodeQwenImage21": RemoteTextEncodeQwenImage21,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MACOFFLOAD_RemoteTextEncodeQwenImage21": "Remote Text Encode Qwen Image 2.1 ⚡",
}
