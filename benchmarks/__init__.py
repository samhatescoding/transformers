from importlib import import_module

__all__ = [
    "BaseBenchmark",
    "MSCOCOBenchmark",
    "Flickr30kBenchmark",
]

_LAZY_IMPORTS = {
    "BaseBenchmark": "base",
    "MSCOCOBenchmark": "mscoco",
    "Flickr30kBenchmark": "flickr30k",
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
