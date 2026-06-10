from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class OpenVid1M(HFMultipleChoiceSourceDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "nkp37/OpenVid-1M",
    ) -> None:
        super().__init__(
            name="openvid1m",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="image",
            image_keys=("image",),
            answer_keys=("prompt", "caption", "text"),
        )

    def _standardize_row(self, row):
        out = dict(row)
        video_name = str(row.get("video", ""))
        out["image"] = self._youtube_thumbnail_url(video_name[:11])
        out["question"] = self.get_question_from_row(row)
        out["answer"] = self.get_answer_from_row(row)
        out["caption"] = out["answer"]
        return out

    def get_captions_from_row(self, row):
        caption = str(row.get("caption", row.get("answer", ""))).strip()
        return [caption] if caption else []
