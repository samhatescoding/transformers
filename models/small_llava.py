from __future__ import annotations

from .llava import Llava


class SmallLlava(Llava):
    def __init__(
        self,
        model_id: str = "Intel/llava-gemma-2b",
        max_new_tokens: int = 256,
        stream: bool = True,
        load_in_4bit: bool = False,
    ):
        # The previous default checkpoint was a tiny internal testing artifact.
        # It is useful for smoke tests, but it is not a meaningful vision-language
        # model and produces degenerate benchmark outputs. Use a real compact
        # LLaVA checkpoint by default instead.
        super().__init__(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            stream=stream,
            load_in_4bit=load_in_4bit,
        )
        self.name = "small-llava"
