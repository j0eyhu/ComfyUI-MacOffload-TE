# encoders/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch

class BaseEncoder(ABC):
    """
    Abstract base class for all text and multimodal conditioning encoders.
    """

    @abstractmethod
    def load(self) -> None:
        """Load weights into unified memory and set up pipeline."""
        pass

    @abstractmethod
    def encode(
        self,
        prompt: str,
        negative_prompt: str = "",
        images_vl: List[torch.Tensor] = None,
        keep_vision: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Extract conditioning features and return dictionary matching ComfyUI format.
        """
        pass
