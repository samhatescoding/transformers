from ._openai_compatible_vision import OpenAICompatibleVisionModel


class Qwen37PlusPreview(OpenAICompatibleVisionModel):
    default_model_id = "qwen3.7-plus-preview"
    api_key_env = "DASHSCOPE_API_KEY"
    base_url_env = "DASHSCOPE_BASE_URL"
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
