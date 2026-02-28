from importlib import import_module

__all__ = [
    "BaseDataset",
    "MSCOCO",
    "Flickr30k",
    "ImageNet1k",
    "UCF101",
    "FlyingThings3D",
    "Cityscapes",
    "INaturalist",
]

_LAZY_IMPORTS = {
    "BaseDataset": "base",
    "MSCOCO": "mscoco",
    "Flickr30k": "flickr30k",
    "ImageNet1k": "imagenet1k",
    "UCF101": "ucf101",
    "FlyingThings3D": "flyingthings3d",
    "Cityscapes": "cityscapes",
    "INaturalist": "inaturalist",
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
