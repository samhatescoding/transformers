from ._openai_vision import GPT5VisionModel


class GPT5(GPT5VisionModel):
    default_model_id = "gpt-5"
    reasoning_effort = "minimal"
    min_output_tokens = 32
