from ._hf_model import AutoImageTextToTextModelBase


class Gemma4_31B(AutoImageTextToTextModelBase):
    display_name = "Gemma 4"
    default_model_id = "google/gemma-4-31B-it"
