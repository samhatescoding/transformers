# models/base.py

from abc import ABC, abstractmethod
from PIL import Image


class BaseModel(ABC):
    """
    Abstract base class for multimodal models.
    """

    name: str
    tokens: int

    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> str:
        """
        Takes an image and a prompt.
        Returns model output as string.
        """
        pass
