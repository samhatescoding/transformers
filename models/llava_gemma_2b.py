from __future__ import annotations

from .llava15_7b import Llava


class SmallLlava(Llava):
    default_model_id = "Intel/llava-gemma-2b"

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 256,
        stream: bool = True,
    ):
        # The previous default checkpoint was a tiny internal testing artifact.
        # It is useful for smoke tests, but it is not a meaningful vision-language
        # model and produces degenerate benchmark outputs. Use a real compact
        # LLaVA checkpoint by default instead.
        super().__init__(
            model_id=model_id or self.default_model_id,
            max_new_tokens=max_new_tokens,
            stream=stream,
        )
