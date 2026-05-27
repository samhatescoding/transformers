from ._openai_vision import GPT5VisionModel


class GPT54Nano(GPT5VisionModel):
    default_model_id = "gpt-5.4-nano"
    reasoning_effort = "none"
