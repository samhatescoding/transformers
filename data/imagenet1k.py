# data/imagenet1k.py

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
from PIL import Image
from datasets import load_dataset
from nltk.corpus import wordnet as wn

from .base import BaseDataset


class ImageNet1k(BaseDataset):
    """
    ImageNet-1k (ILSVRC 2012) via Hugging Face datasets.

    Notes:
    - This dataset is often gated on HF: you may need `huggingface-cli login`
      and to accept the dataset access request on the dataset page.
    - Label list is 1000 classes; using them all in a prompt will be huge.
      Consider returning synset ids or doing top-k candidate prompting.
    """

    def __init__(
        self,
        split: str = "validation",
        streaming: bool = True,
        dataset_id: str = "ILSVRC/imagenet-1k",
        use_auth_token: Optional[bool] = None,
    ):
        self.name = "imagenet-1k"
        self.split = split
        self.streaming = streaming

        # `use_auth_token`:
        # - If None: datasets will use your cached HF credentials if present
        # - If True: forces auth
        self.ds = load_dataset(
            dataset_id,
            split=split,
            streaming=streaming,
            use_auth_token=use_auth_token,
        )

        # Build labels from dataset features (works best when not streaming OR when features are known)
        # Typically the feature is named "label".
        try:
            label_feature = self.ds.features["label"]
            # `names` is a list of strings for ClassLabel features
            self.class_labels = list(getattr(label_feature, "names", [])) or []
        except Exception:
            self.class_labels = []

        # Public labels list can be updated by get_labels(rows), matching other datasets.
        self.labels = list(self.class_labels)

        # Optional mapping: class index -> ImageNet synset id (e.g. n02099601)
        # Some dataset configs expose synset IDs as label names.
        self.idx2synset: List[str] = []
        for name in self.class_labels:
            if isinstance(name, str) and len(name) == 9 and name[0] in {"n", "v", "a", "r"} and name[1:].isdigit():
                self.idx2synset.append(name)
            else:
                self.idx2synset = []
                break

    # Helpers
    
    @staticmethod
    def _synset_from_imagenet_synset_id(synset_id: str):
        """
        ImageNet synset ids look like 'n02099601'.
        WordNet expects e.g. wn.synset_from_pos_and_offset('n', 2099601).
        """
        pos = synset_id[0]                 # 'n'
        offset = int(synset_id[1:])        # 2099601
        return wn.synset_from_pos_and_offset(pos, offset)

    @staticmethod
    def _ancestor_synsets(syn, max_depth: int = 20) -> List[Any]:
        """
        BFS up hypernym tree; returns unique ancestors (not including the synset itself).
        """
        out = []
        seen = set()
        frontier = [(syn, 0)]
        while frontier:
            node, d = frontier.pop(0)
            if d >= max_depth:
                continue
            for h in node.hypernyms():
                if h.name() in seen:
                    continue
                seen.add(h.name())
                out.append(h)
                frontier.append((h, d + 1))
        return out
    
    # Dataset interface

    def __repr__(self) -> str:
        return f"<Dataset {self.name} | split={self.split} | streaming={self.streaming}>"

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.ds)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(self.ds):
            if i >= n:
                break
            samples.append(row)
        return samples

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        img = row.get("image")
        if img is None:
            raise ValueError("Row has no 'image' field.")

        if isinstance(img, Image.Image):
            return img.convert("RGB")

        # Some HF image rows can be dict-like
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"] is not None:
                from io import BytesIO
                return Image.open(BytesIO(img["bytes"])).convert("RGB")
            if "path" in img and img["path"]:
                return Image.open(img["path"]).convert("RGB")

        raise TypeError(f"Unsupported image type in row['image']: {type(img)}")

    def get_labels_img(self, row) -> List[str]:
        """
        Returns: [leaf_label] + [parent_label_1, parent_label_2, ...]
        Parent labels come from WordNet hypernyms.
        """
        idx = int(row["label"])
        if idx < 0 or idx >= len(self.class_labels):
            raise IndexError(f"Label index {idx} out of range for ImageNet class labels.")
        leaf_label = self.class_labels[idx]  # readable label for this class index

        # If synset ids are unavailable, return only the leaf class label.
        if not self.idx2synset or idx >= len(self.idx2synset):
            return [leaf_label]

        synset_id = self.idx2synset[idx]   # e.g. "n02099601"
        syn = self._synset_from_imagenet_synset_id(synset_id)

        parents = []
        for a in self._ancestor_synsets(syn, max_depth=20):
            # pick a readable name (first lemma)
            lemmas = a.lemma_names()
            if not lemmas:
                continue
            name = lemmas[0].replace("_", " ")
            parents.append(name)

        # Option: dedupe while keeping order
        seen = set()
        all_labels = []
        for x in [leaf_label] + parents:
            nx = self.normalize_text(x)
            if nx in seen:
                continue
            seen.add(nx)
            all_labels.append(x)

        return all_labels

    def get_labels(self, rows) -> List[str]:
        merged: List[str] = []
        seen = set()
        for row in rows:
            for label in self.get_labels_img(row):
                nlabel = self.normalize_text(label)
                if nlabel in seen:
                    continue
                seen.add(nlabel)
                merged.append(label)

        self.labels = merged
        return merged
