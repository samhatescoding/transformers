from ._openai_vision import GPT5VisionModel


class GPT55(GPT5VisionModel):
    default_model_id = "gpt-5.5"
    reasoning_effort = "none"
