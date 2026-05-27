from ._hf_model import AutoImageTextToTextModelBase


class Gemma4E2B(AutoImageTextToTextModelBase):
    display_name = "Gemma 4"
    default_model_id = "google/gemma-4-E2B-it"
