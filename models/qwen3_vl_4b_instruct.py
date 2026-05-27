from ._hf_model import AutoImageTextToTextModelBase


class Qwen3VL4B(AutoImageTextToTextModelBase):
    display_name = "Qwen3-VL"
    default_model_id = "Qwen/Qwen3-VL-4B-Instruct"
