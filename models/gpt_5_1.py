from ._openai_vision import GPT5VisionModel


class GPT51(GPT5VisionModel):
    default_model_id = "gpt-5.1"
    reasoning_effort = "none"
