from importlib import import_module

__all__ = [
    "BaseModel",
    "Llava",
    "LlavaOnevision",
    "SmallLlava",
    "Qwen25VL",
    "GPT4",
    "Falcon",
    "Gemma",
    "Baichuan2",
    "DBRX",
    "BLOOM",
    "Orion14B",
    "OLMoE",
]

_LAZY_IMPORTS = {
    "BaseModel": "_base_model",
    "Llava": "llava",
    "LlavaOnevision": "llava_onevision",
    "SmallLlava": "small_llava",
    "Qwen25VL": "qwen25_vl",
    "GPT4": "gpt4",
    "Falcon": "falcon",
    "Gemma": "gemma",
    "Baichuan2": "baichuan2",
    "DBRX": "dbrx",
    "BLOOM": "bloom",
    "Orion14B": "orion14b",
    "OLMoE": "olmoe",
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
