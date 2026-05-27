from ._hf_model import AutoImageTextToTextModelBase


class LlavaOnevision15_4BInstruct(AutoImageTextToTextModelBase):
    display_name = "LLaVA-OneVision 1.5"
    default_model_id = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(*args, **kwargs)
