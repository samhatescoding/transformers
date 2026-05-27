from ._openai_vision import OpenAIResponsesVisionModel


class GPT4(OpenAIResponsesVisionModel):
    default_model_id = "gpt-4o"
