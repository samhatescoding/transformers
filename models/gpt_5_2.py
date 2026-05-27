from ._openai_vision import GPT5VisionModel


class GPT52(GPT5VisionModel):
    default_model_id = "gpt-5.2"
    reasoning_effort = "none"
