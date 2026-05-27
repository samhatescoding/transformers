from ._llava_next import LlavaNextModelBase


class Falcon(LlavaNextModelBase):
    display_name = "Falcon VLM"
    default_model_id = "tiiuae/falcon-11B-vlm"

    @staticmethod
    def _name_from_model_id(model_id: str) -> str:
        return str(model_id).rsplit("/", 1)[-1].lower().replace("_", "-")
