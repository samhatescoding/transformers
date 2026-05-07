# datasets/base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Set
from PIL import Image


class BaseDataset(ABC):
    """
    Minimal base class to standardize dataset access.
    """

    name: str
    labels: List[str]

    @abstractmethod
    def __iter__(self) -> Iterable[Dict[str, Any]]:
        ...
        
    @abstractmethod
    def get_labels(self, rows) -> List[str]:
        ...

    @abstractmethod
    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        ...

    def normalize_text(self, text: str) -> str:
        t = text.strip().lower()
        t = t.replace("\n", " ").strip()
        for ch in [".", ",", ";", ":", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}"]:
            t = t.replace(ch, "")
        t = " ".join(t.split())
        return t

    def label_map_normalized(self) -> Dict[str, str]:
        """
        Returns a mapping: normalized_label -> canonical_label
        """
        return {self.normalize_text(l): l for l in self.labels}

    def is_valid_label(self, label: str) -> bool:
        return self.normalize_text(label) in self.label_map_normalized()
    
    # -------------------------
    # Noun extraction utilities
    # -------------------------
    def extract_nouns(self, captions: List[str]) -> Set[str]:
        import re
        import nltk
        
        # Ensure required resources exist (downloads only if missing)
        def ensure(resource_path: str, download_name: str):
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(download_name, quiet=True)

        ensure("tokenizers/punkt", "punkt")
        # NLTK now uses *_eng for the English perceptron tagger in newer versions
        try:
            ensure("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
        except Exception:
            # Fallback for older NLTK
            ensure("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")

        ensure("corpora/wordnet", "wordnet")

        from nltk import word_tokenize, pos_tag
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        nouns: Set[str] = set()

        for cap in captions:
            cap = cap.strip()
            if not cap:
                continue

            tokens = word_tokenize(cap)
            tagged = pos_tag(tokens)

            for tok, tag in tagged:
                if tag.startswith("NN"):
                    w = tok.lower()
                    w = re.sub(r"[^a-z]", "", w)
                    if not w:
                        continue
                    w = lemmatizer.lemmatize(w, pos="n")
                    if len(w) >= 2:
                        nouns.add(w)

        return nouns
