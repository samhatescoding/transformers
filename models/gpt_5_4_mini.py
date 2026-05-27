from ._openai_vision import GPT5VisionModel


class GPT54Mini(GPT5VisionModel):
    default_model_id = "gpt-5.4-mini"
    reasoning_effort = "none"
