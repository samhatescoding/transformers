from ._hf_model import AutoImageTextToTextModelBase


class RemoteCodeVisionLanguageModel(AutoImageTextToTextModelBase):
    """Shared base for multimodal checkpoints with custom Transformers code."""

    default_trust_remote_code = True
