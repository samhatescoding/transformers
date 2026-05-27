from ._hf_model import AutoImageTextToTextModelBase


class Qwen3VL8B(AutoImageTextToTextModelBase):
    display_name = "Qwen3-VL"
    default_model_id = "Qwen/Qwen3-VL-8B-Instruct"
