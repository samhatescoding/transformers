from ._hf_model import AutoImageTextToTextModelBase


class Qwen35_4B(AutoImageTextToTextModelBase):
    display_name = "Qwen3.5"
    default_model_id = "Qwen/Qwen3.5-4B"
