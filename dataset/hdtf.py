from __future__ import annotations

from .hf_common import HFVideoCaptionDataset


class HDTF(HFVideoCaptionDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "Darknsu/mead_hdtf_400_merge_video_audio_frames_only",
    ) -> None:
        super().__init__(
            name="hdtf",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            frame_keys=("png",),
            max_frames=1,
            caption_keys=("caption", "__key__"),
        )
