from ._hf_model import AutoImageTextToTextModelBase


class Gemma3_27B(AutoImageTextToTextModelBase):
    display_name = "Gemma 3"
    default_model_id = "google/gemma-3-27b-it"
