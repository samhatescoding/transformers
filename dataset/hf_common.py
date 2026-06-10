from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List, Sequence
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw
from datasets import load_dataset

from ._base_dataset import BaseDataset


class HFBaseDataset(BaseDataset):
    def __init__(
        self,
        *,
        name: str,
        dataset_id: str,
        split: str,
        streaming: bool = True,
        config_name: str | None = None,
        revision: str | None = None,
        use_auth_token: bool | None = None,
        **load_kwargs: Any,
    ) -> None:
        self.name = name
        self.dataset_id = dataset_id
        self.split = split
        self.streaming = streaming
        self.labels: List[str] = []

        kwargs = dict(load_kwargs)
        if revision is not None:
            kwargs["revision"] = revision
        if use_auth_token is not None:
            kwargs["use_auth_token"] = use_auth_token

        if config_name is not None:
            self.ds = load_dataset(dataset_id, config_name, split=split, streaming=streaming, **kwargs)
        else:
            self.ds = load_dataset(dataset_id, split=split, streaming=streaming, **kwargs)

    def __repr__(self) -> str:
        return f"<Dataset {self.name} | repo={self.dataset_id} | split={self.split} | streaming={self.streaming}>"

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for row in self.ds:
            yield self._standardize_row(row)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for index, row in enumerate(self.ds):
            if index >= n:
                break
            samples.append(self._standardize_row(row))
        return samples

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return dict(row)

    def _get_first_present(self, row: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return None

    def _get_text(self, row: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
        value = self._get_first_present(row, keys)
        if value is None:
            return default
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text:
                    return text
            return default
        text = str(value).strip()
        return text or default

    def _get_text_list(self, row: Dict[str, Any], keys: Sequence[str]) -> List[str]:
        value = self._get_first_present(row, keys)
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, dict):
            values: List[str] = []
            for nested_key in ("text", "texts", "answer", "answers", "caption", "captions", "label", "labels"):
                nested = value.get(nested_key)
                if isinstance(nested, list):
                    values.extend(str(item).strip() for item in nested if str(item).strip())
                elif nested is not None and str(nested).strip():
                    values.append(str(nested).strip())
            return values
        text = str(value).strip()
        return [text] if text else []

    def _coerce_image(self, value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            request = Request(value, headers={"User-Agent": "transformers-benchmark-input-browser/1.0"})
            with urlopen(request, timeout=60) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        if hasattr(value, "convert"):
            try:
                return value.convert("RGB")
            except Exception:
                pass
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"])).convert("RGB")
            if value.get("path"):
                return Image.open(value["path"]).convert("RGB")
            for key in ("image", "frame"):
                nested = value.get(key)
                if nested is not None:
                    return self._coerce_image(nested)
        if isinstance(value, (list, tuple)) and value:
            return self._coerce_image(value[0])
        return Image.fromarray(value).convert("RGB")

    @staticmethod
    def _youtube_thumbnail_url(video_id: Any) -> str:
        normalized = str(video_id or "").strip()
        if not normalized:
            raise ValueError("Could not determine the source YouTube video ID.")
        return f"https://i.ytimg.com/vi/{normalized}/hqdefault.jpg"

    def _extract_image(self, row: Dict[str, Any], keys: Sequence[str]) -> Image.Image:
        value = self._get_first_present(row, keys)
        if value is None:
            raise ValueError(f"Could not find image data in row for keys: {list(keys)}")
        return self._coerce_image(value)

    def _extract_frames(self, row: Dict[str, Any], keys: Sequence[str], max_frames: int = 4) -> List[Image.Image]:
        value = self._get_first_present(row, keys)
        if value is None:
            image = self._get_first_present(row, ("image", "frame"))
            if image is None:
                return []
            return [self._coerce_image(image)]
        if isinstance(value, list):
            if max_frames == 1 and value:
                return [self._coerce_image(value[len(value) // 2])]
            return [self._coerce_image(item) for item in value[:max_frames]]
        if isinstance(value, dict):
            for nested_key in ("frames", "images", "video_frames"):
                nested = value.get(nested_key)
                if isinstance(nested, list):
                    if max_frames == 1 and nested:
                        return [self._coerce_image(nested[len(nested) // 2])]
                    return [self._coerce_image(item) for item in nested[:max_frames]]
        return [self._coerce_image(value)]

    def _extract_feature_labels(self, feature_keys: Sequence[str]) -> List[str]:
        try:
            features = getattr(self.ds, "features", {})
            for key in feature_keys:
                feature = features.get(key)
                names = list(getattr(feature, "names", []))
                if names:
                    return names
        except Exception:
            pass
        return []


class HFClassificationDataset(HFBaseDataset):
    def __init__(
        self,
        *,
        name: str,
        dataset_id: str,
        split: str,
        streaming: bool = True,
        image_keys: Sequence[str] = ("image",),
        label_keys: Sequence[str] = ("label",),
        fallback_labels: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dataset_id=dataset_id, split=split, streaming=streaming, **kwargs)
        self.image_keys = tuple(image_keys)
        self.label_keys = tuple(label_keys)
        self.labels = self._extract_feature_labels(self.label_keys)
        if not self.labels and fallback_labels is not None:
            self.labels = list(fallback_labels)

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return self._extract_image(row, self.image_keys)

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        raw = self._get_first_present(row, self.label_keys)
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, int) and 0 <= raw < len(self.labels):
            return [self.labels[raw]]
        text = str(raw).strip()
        return [text] if text else []

    def get_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        if self.labels:
            return list(self.labels)
        merged: List[str] = []
        seen = set()
        for row in rows:
            for label in self.get_labels_img(row):
                normalized = self.normalize_text(label)
                if normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(label)
        self.labels = merged
        return merged


class HFQADataset(HFBaseDataset):
    def __init__(
        self,
        *,
        name: str,
        dataset_id: str,
        split: str,
        streaming: bool = True,
        image_keys: Sequence[str] = ("image",),
        question_keys: Sequence[str] = ("question",),
        answer_keys: Sequence[str] = ("answers", "answer"),
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dataset_id=dataset_id, split=split, streaming=streaming, **kwargs)
        self.image_keys = tuple(image_keys)
        self.question_keys = tuple(question_keys)
        self.answer_keys = tuple(answer_keys)

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["question"] = self.get_question_from_row(row)
        out["answers"] = self.get_answers_from_row(row)
        return out

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return self._extract_image(row, self.image_keys)

    def get_labels(self, rows) -> List[str]:
        del rows
        return []

    def get_labels_img(self, row) -> List[str]:
        return self.get_answers_from_row(row)

    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        return self._get_text(row, self.question_keys)

    def get_answers_from_row(self, row: Dict[str, Any]) -> List[str]:
        return self._get_text_list(row, self.answer_keys)


class HFCaptionDataset(HFBaseDataset):
    def __init__(
        self,
        *,
        name: str,
        dataset_id: str,
        split: str,
        streaming: bool = True,
        image_keys: Sequence[str] = ("image",),
        caption_keys: Sequence[str] = ("captions", "caption"),
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dataset_id=dataset_id, split=split, streaming=streaming, **kwargs)
        self.image_keys = tuple(image_keys)
        self.caption_keys = tuple(caption_keys)

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["captions"] = self.get_captions_from_row(row)
        return out

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return self._extract_image(row, self.image_keys)

    def get_labels(self, rows) -> List[str]:
        del rows
        return []

    def get_labels_img(self, row) -> List[str]:
        return []

    def get_captions_from_row(self, row: Dict[str, Any]) -> List[str]:
        return self._get_text_list(row, self.caption_keys)


class HFVideoCaptionDataset(HFCaptionDataset):
    def __init__(
        self,
        *,
        frame_keys: Sequence[str] = ("frames", "video_frames", "images", "image"),
        max_frames: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(image_keys=frame_keys, **kwargs)
        self.frame_keys = tuple(frame_keys)
        self.max_frames = max_frames

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = super()._standardize_row(row)
        out["frames"] = self._extract_frames(row, self.frame_keys, max_frames=self.max_frames)
        return out

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        frames = row.get("frames")
        if not isinstance(frames, list) or not frames:
            frames = self._extract_frames(row, self.frame_keys, max_frames=self.max_frames)
        if not frames:
            raise ValueError("Could not extract a representative frame from the video row.")
        return self._coerce_image(frames[len(frames) // 2])


class HFVideoClassificationDataset(HFBaseDataset):
    def __init__(
        self,
        *,
        name: str,
        dataset_id: str,
        split: str,
        streaming: bool = True,
        frame_keys: Sequence[str] = ("frames", "video_frames", "images", "image"),
        label_keys: Sequence[str] = ("label_text", "label", "class"),
        fallback_labels: Sequence[str] | None = None,
        max_frames: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dataset_id=dataset_id, split=split, streaming=streaming, **kwargs)
        self.frame_keys = tuple(frame_keys)
        self.label_keys = tuple(label_keys)
        self.max_frames = max_frames
        self.labels = self._extract_feature_labels(self.label_keys)
        if not self.labels and fallback_labels is not None:
            self.labels = list(fallback_labels)

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["frames"] = self._extract_frames(row, self.frame_keys, max_frames=self.max_frames)
        labels = self.get_labels_img(row)
        out["label_text"] = labels[0] if labels else ""
        return out

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        frames = row.get("frames")
        if isinstance(frames, list) and frames:
            return self._coerce_image(frames[len(frames) // 2])
        extracted = self._extract_frames(row, self.frame_keys, max_frames=1)
        if not extracted:
            raise ValueError("Could not extract a representative frame from the row.")
        return extracted[0]

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        raw = self._get_first_present(row, self.label_keys)
        if raw is None:
            return []
        if isinstance(raw, int) and 0 <= raw < len(self.labels):
            return [self.labels[raw]]
        text = str(raw).strip()
        return [text] if text else []

    def get_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        if self.labels:
            return list(self.labels)
        merged: List[str] = []
        seen = set()
        for row in rows:
            for label in self.get_labels_img(row):
                normalized = self.normalize_text(label)
                if normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(label)
        self.labels = merged
        return merged


class HFMultipleChoiceSourceDataset(HFBaseDataset):
    def __init__(
        self,
        *,
        name: str,
        dataset_id: str,
        split: str,
        streaming: bool = True,
        mode: str = "image",
        image_keys: Sequence[str] = ("image",),
        source_image_keys: Sequence[str] = ("source_image", "image_before"),
        target_image_keys: Sequence[str] = ("target_image", "image_after"),
        frame_keys: Sequence[str] = ("frames", "video_frames", "images", "image"),
        question_keys: Sequence[str] = ("question", "prompt", "instruction"),
        answer_keys: Sequence[str] = ("answer", "caption", "text", "prompt"),
        max_frames: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dataset_id=dataset_id, split=split, streaming=streaming, **kwargs)
        self.mode = mode
        self.image_keys = tuple(image_keys)
        self.source_image_keys = tuple(source_image_keys)
        self.target_image_keys = tuple(target_image_keys)
        self.frame_keys = tuple(frame_keys)
        self.question_keys = tuple(question_keys)
        self.answer_keys = tuple(answer_keys)
        self.max_frames = max_frames

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        if self.mode == "pair":
            out["source_image"] = self._extract_image(row, self.source_image_keys)
            out["target_image"] = self._extract_image(row, self.target_image_keys)
        elif self.mode == "video":
            out["frames"] = self._extract_frames(row, self.frame_keys, max_frames=self.max_frames)
        else:
            out["image"] = self._extract_image(row, self.image_keys)
        out["question"] = self.get_question_from_row(row)
        out["answer"] = self.get_answer_from_row(row)
        return out

    def get_labels(self, rows) -> List[str]:
        del rows
        return []

    def get_labels_img(self, row) -> List[str]:
        answer = self.get_answer_from_row(row)
        return [answer] if answer else []

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        if row.get("source_image") is not None:
            return self._coerce_image(row["source_image"])
        if row.get("frames"):
            return self._coerce_image(row["frames"][0])
        if row.get("image") is not None:
            return self._coerce_image(row["image"])
        if self.mode == "pair":
            return self._extract_image(row, self.source_image_keys)
        if self.mode == "video":
            frames = self._extract_frames(row, self.frame_keys, max_frames=1)
            if frames:
                return frames[0]
        return self._extract_image(row, self.image_keys)

    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        return self._get_text(row, self.question_keys)

    def get_answer_from_row(self, row: Dict[str, Any]) -> str:
        return self._get_text(row, self.answer_keys)
