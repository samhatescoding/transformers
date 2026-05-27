from ._openai_vision import GPT5VisionModel


class GPT54(GPT5VisionModel):
    default_model_id = "gpt-5.4"
    reasoning_effort = "none"
