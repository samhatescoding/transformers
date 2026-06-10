from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmarks import BLIP3o60kBenchmark, ShareGPT4oImageBenchmark
from benchmarks.captioning.internvid import InternVidBenchmark
from benchmarks.detection._detection import DetectionBenchmark
from benchmarks.detection.lvis import LVISBenchmark
from dataset.blip3o_60k import BLIP3o60k
from dataset.cityscapes import Cityscapes
from dataset.docvqa import DocVQA
from dataset.gqa import GQA
from dataset.hdtf import HDTF
from dataset.imgedit import ImgEdit
from dataset.internvid import InternVid
from dataset.kinetics import Kinetics
from dataset.laion400m import LAION400M
from dataset.laion5b import LAION5B
from dataset.lsun import LSUN
from dataset.lvis import LVIS
from dataset.mscoco import MSCOCO
from dataset.openimages_v4 import OpenImagesV4
from dataset.openvid1m import OpenVid1M
from dataset.sharegpt4o_image import ShareGPT4oImage
from dataset.tad66k import TAD66K
from dataset.visual_cot import VisualCoT
from dataset.visual_genome import VisualGenome


class _Rows(list):
    features = {}


class DatasetIntegrationFixTests(unittest.TestCase):
    def test_gqa_selects_only_one_question_per_image(self):
        dataset = GQA.__new__(GQA)
        dataset.instruction_ds = [
            {"id": "q1", "imageId": "image-1", "question": "First?", "answer": "yes"},
            {"id": "q2", "imageId": "image-1", "question": "Repeated?", "answer": "no"},
            {"id": "q3", "imageId": "image-2", "question": "Second?", "answer": "blue"},
        ]
        dataset.image_ds = [
            {"id": "image-1", "image": Image.new("RGB", (8, 8), "red")},
            {"id": "image-2", "image": Image.new("RGB", (8, 8), "blue")},
        ]

        rows = dataset.get_samples(2)

        self.assertEqual([row["id"] for row in rows], ["q1", "q3"])
        self.assertEqual([row["image_id"] for row in rows], ["image-1", "image-2"])

    def test_docvqa_skips_incorrect_first_reference_row(self):
        dataset = DocVQA.__new__(DocVQA)
        dataset.ds = [
            {"questionId": "49153", "docId": 1, "question": "Incorrect row", "answers": ["0.28"]},
            {"questionId": "24580", "docId": 2, "question": "Valid row", "answers": ["answer"]},
            {"questionId": "24581", "docId": 2, "question": "Repeated document", "answers": ["answer"]},
            {"questionId": "57349", "docId": 3, "question": "Next row", "answers": ["answer 2"]},
        ]
        dataset.question_keys = ("question",)
        dataset.answer_keys = ("answers", "answer")

        rows = dataset.get_samples(2)

        self.assertEqual([row["questionId"] for row in rows], ["24580", "57349"])

    def test_blip3o_streams_webdataset_and_uses_prompt_reconstruction(self) -> None:
        image = Image.new("RGB", (8, 6), "purple")
        with patch("dataset.blip3o_60k.load_dataset", return_value=_Rows([{"jpg": image, "text": "purple castle"}])) as loader:
            dataset = BLIP3o60k()
            row = dataset.get_samples(1)[0]

        self.assertEqual(row["answer"], "purple castle")
        self.assertEqual(row["image"].size, (8, 6))
        self.assertEqual(BLIP3o60kBenchmark.task_type, "prompt_reconstruction")
        loader.assert_called_once_with(
            "webdataset",
            data_files="hf://datasets/BLIP3o/BLIP3o-60k/*.tar",
            split="train",
            streaming=True,
        )

    def test_sharegpt_text_to_image_uses_local_metadata_and_prompt_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            Image.new("RGB", (7, 5), "orange").save(image_path)
            (root / "text_to_image.json").write_text(
                json.dumps([{"prompt": "an orange sphere", "image": "image.png"}]),
                encoding="utf-8",
            )

            dataset = ShareGPT4oImage(data_dir=root)
            row = dataset.get_samples(1)[0]

        self.assertEqual(row["answer"], "an orange sphere")
        self.assertEqual(ShareGPT4oImageBenchmark.task_type, "prompt_reconstruction")

    def test_sharegpt_streams_processed_webdataset_when_local_root_is_absent(self) -> None:
        image = Image.new("RGB", (7, 5), "orange")
        with patch(
            "dataset.sharegpt4o_image.load_dataset",
            return_value=_Rows([{"jpg": image, "txt": "an orange sphere"}]),
        ) as loader:
            dataset = ShareGPT4oImage()
            row = dataset.get_samples(1)[0]

        self.assertEqual(row["answer"], "an orange sphere")
        self.assertEqual(row["image"].size, (7, 5))
        loader.assert_called_once_with(
            "hanlincs/sharegpt4oimage_processed",
            split="train",
            streaming=True,
            token=None,
        )

    def test_tad66k_reads_local_scores_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.jpg"
            Image.new("RGB", (6, 4), "blue").save(image_path)
            (root / "scores.csv").write_text(
                "filename,score,split\nsample.jpg,6.6,train\n",
                encoding="utf-8",
            )

            dataset = TAD66K(data_dir=root)
            row = dataset.get_samples(1)[0]

        self.assertEqual(row["rating"], 7)
        self.assertEqual(dataset.get_labels_img(row), ["7"])

    def test_tad66k_reads_remote_score_rows_without_downloading_images(self) -> None:
        labels = b"image,score\nsample.jpg,6.6\n"
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "labels.zip"
            import zipfile

            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("labels/merge/train.csv", labels)
            with patch("huggingface_hub.hf_hub_download", return_value=str(archive_path)):
                dataset = TAD66K()
                row = dataset.get_samples(1)[0]

        self.assertEqual(row["image_name"], "sample.jpg")
        self.assertEqual(row["rating"], 7)

    def test_cityscapes_uses_real_semantic_classes(self) -> None:
        dataset = Cityscapes.__new__(Cityscapes)
        dataset.labels = list(Cityscapes.LABELS)
        row = {
            "image": np.zeros((3, 4, 5), dtype=np.float32),
            "segmentation_19": np.array([[0, 0, 13], [13, 11, 11]]),
        }

        self.assertEqual(dataset.get_image_from_row(row).size, (5, 4))
        self.assertEqual(dataset.get_labels_img(row), ["road", "person", "car"])

    def test_lvis_parses_detection_annotations(self) -> None:
        dataset = LVIS.__new__(LVIS)
        dataset.category_id_to_label = {117: "rare object"}
        dataset.labels = ["rare object"]
        row = {"objects": {"bboxes": [[[10, 20, 30, 40]]], "classes": [117]}}

        self.assertEqual(
            dataset.get_annotations_for_row(row),
            [{"bbox": [10.0, 20.0, 30.0, 40.0], "label": "rare object"}],
        )

    def test_flickr30k_entities_scales_boxes_to_the_decoded_image(self) -> None:
        from dataset.flickr30k_entities import Flickr30kEntities

        dataset = Flickr30kEntities.__new__(Flickr30kEntities)
        row = {
            "image": Image.new("RGB", (1000, 750), "white"),
            "caption": "(A person) beside (a bicycle) .",
            "json_data": (
                "{'boxes': array([array([100, 50, 200, 150]), "
                "array([250, 100, 450, 300])], dtype=object), "
                "'caption': '(A person) beside (a bicycle) .', "
                "'gt_anno_ids': None, 'gt_ids': array([1, 2])}"
            ),
        }

        self.assertEqual(
            dataset.get_annotations_for_row(row),
            [
                {"label": "A person", "bbox": [200.0, 100.0, 200.0, 200.0]},
                {"label": "a bicycle", "bbox": [500.0, 200.0, 400.0, 400.0]},
            ],
        )

    def test_flickr30k_entities_does_not_parse_annotation_ids_as_boxes(self) -> None:
        from dataset.flickr30k_entities import Flickr30kEntities

        dataset = Flickr30kEntities.__new__(Flickr30kEntities)
        row = {
            "image": Image.new("RGB", (1200, 800), "white"),
            "caption": "(A woman) raises (her arm) .",
            "json_data": (
                "{'boxes': None, 'caption': '(A woman) raises (her arm) .', "
                "'gt_anno_ids': array([647491, 647490, 647488, 647489]), "
                "'gt_ids': array([4, 3, 1, 2])}"
            ),
        }

        self.assertEqual(dataset.get_annotations_for_row(row), [])

    def test_type_b_adapters_preserve_their_native_box_conventions(self) -> None:
        coco = MSCOCO.__new__(MSCOCO)
        coco.name = "mscoco"
        coco.labels = ["sink"]
        coco.category_id_to_label = {81: "sink"}
        coco._annotations_by_image_id = {
            397133: [{"bbox": [217.62, 240.54, 38.99, 57.75], "category_id": 81}]
        }
        coco_row = {"image_id": 397133, "image": Image.new("RGB", (640, 427), "white")}

        lvis = LVIS.__new__(LVIS)
        lvis.name = "lvis"
        lvis.labels = ["rare object"]
        lvis.category_id_to_label = {117: "rare object"}
        lvis_row = {
            "image": Image.new("RGB", (640, 360), "white"),
            "objects": {"bboxes": [[476.39, 303.29, 18.24, 21.42]], "classes": [117]},
        }

        openimages = OpenImagesV4.__new__(OpenImagesV4)
        openimages.name = "openimages_v4"
        openimages.labels = ["Fixed-wing aircraft"]
        openimages_row = {
            "image": Image.new("RGB", (1024, 447), "white"),
            "objects": [
                {
                    "label": "Fixed-wing aircraft",
                    "xmin": 0.02267303131520748,
                    "ymin": 0.07103825360536575,
                    "xmax": 0.964200496673584,
                    "ymax": 0.8005464673042297,
                }
            ],
        }

        cases = (
            (
                coco,
                coco_row,
                [217.62, 240.54, 256.61, 298.29],
            ),
            (
                lvis,
                lvis_row,
                [476.39, 303.29, 494.63, 324.71],
            ),
            (
                openimages,
                openimages_row,
                [
                    0.02267303131520748 * 1024,
                    0.07103825360536575 * 447,
                    0.964200496673584 * 1024,
                    0.8005464673042297 * 447,
                ],
            ),
        )

        for dataset, row, expected in cases:
            with self.subTest(dataset=dataset.name):
                benchmark = DetectionBenchmark(dataset=dataset, name=dataset.name)
                boxes = benchmark.get_ground_truth_boxes_for_row(row)
                boxes = benchmark.postprocess_ground_truth_boxes(boxes, image=row["image"])
                self.assertEqual(len(boxes), 1)
                for actual, wanted in zip(boxes[0]["xyxy"], expected):
                    self.assertAlmostEqual(actual, wanted, places=5)

    def test_repaired_adapters_use_image_bearing_sources(self) -> None:
        cases = (
            (VisualGenome, "dipta007/bengali-visual-genome-1.0-prompt", "train"),
            (LVIS, "fw407/lvis", "train"),
            (HDTF, "Darknsu/mead_hdtf_400_merge_video_audio_frames_only", "train"),
            (InternVid, "OpenGVLab/InternVid-Full", "train"),
            (LAION400M, "tempertrash/laion_400m", "train"),
            (LAION5B, "nousr/laion5b-subset-and-cliph-embeddings", "train"),
            (ImgEdit, "diffusion-cot/imgedit-simpler", "train"),
            (OpenVid1M, "nkp37/OpenVid-1M", "train"),
            (Kinetics, "iejMac/CLIP-Kinetics700", "train"),
        )
        for dataset_cls, dataset_id, expected_split in cases:
            with self.subTest(dataset=dataset_cls.__name__):
                with patch("dataset.hf_common.load_dataset", return_value=_Rows()) as loader:
                    dataset_cls()
                loader.assert_called_once_with(
                    dataset_id,
                    split=expected_split,
                    streaming=True,
                )

        self.assertTrue(issubclass(LVISBenchmark, DetectionBenchmark))

    def test_video_metadata_adapters_create_lazy_thumbnail_images(self) -> None:
        with patch("dataset.hf_common.load_dataset", return_value=_Rows()):
            internvid = InternVid()
            openvid = OpenVid1M()
            kinetics = Kinetics()

        self.assertEqual(
            internvid._standardize_row({"YoutubeID": "video123456", "Caption": "caption"})["image"],
            "https://i.ytimg.com/vi/video123456/hqdefault.jpg",
        )
        self.assertEqual(
            openvid._standardize_row({"video": "---_iRTHryQ_13_0to241.mp4", "caption": "caption"})["image"],
            "https://i.ytimg.com/vi/---_iRTHryQ/hqdefault.jpg",
        )
        self.assertEqual(
            kinetics._standardize_row({"youtube_id": "---0dWlqevI", "label": "pottery"})["image"],
            "https://i.ytimg.com/vi/---0dWlqevI/hqdefault.jpg",
        )

    def test_internvid_samples_distinct_youtube_videos(self) -> None:
        dataset = InternVid.__new__(InternVid)
        dataset.name = "internvid"
        dataset.frame_keys = ("image",)
        dataset.max_frames = 1
        dataset.caption_keys = ("captions", "Caption")
        dataset.ds = [
            {"YoutubeID": "video-a", "Caption": "first caption"},
            {"YoutubeID": "video-a", "Caption": "second caption"},
            {"YoutubeID": "video-b", "Caption": "third caption"},
            {"YoutubeID": "video-c", "Caption": "fourth caption"},
        ]

        rows = dataset.get_samples(3)

        self.assertEqual([row["video_id"] for row in rows], ["video-a", "video-b", "video-c"])
        self.assertEqual(
            [row["captions"] for row in rows],
            [
                ["first caption", "second caption"],
                ["third caption"],
                ["fourth caption"],
            ],
        )
        self.assertEqual(len({row["image"] for row in rows}), 3)
        benchmark = InternVidBenchmark(dataset=dataset)
        self.assertEqual(
            benchmark.get_valid_labels_for_row(rows[0]),
            ["first caption", "second caption"],
        )

        events = []
        dataset.set_preview_progress_callback(events.append)
        dataset.get_samples(2)
        self.assertEqual(
            events,
            [
                "Grouped captions for 1 of 2 distinct InternVid videos.",
                "Grouped captions for 2 of 2 distinct InternVid videos.",
            ],
        )

    def test_internvid_skips_videos_with_unavailable_thumbnails(self) -> None:
        dataset = InternVid.__new__(InternVid)
        dataset.name = "internvid"
        dataset.frame_keys = ("image",)
        dataset.max_frames = 1
        dataset.caption_keys = ("captions", "Caption")
        dataset.ds = [
            {"YoutubeID": "video-a", "Caption": "first caption"},
            {"YoutubeID": "video-b", "Caption": "broken caption"},
            {"YoutubeID": "video-c", "Caption": "third caption"},
        ]

        def load_thumbnail(row):
            if row["video_id"] == "video-b":
                return None
            color = "red" if row["video_id"] == "video-a" else "blue"
            return Image.new("RGB", (16, 12), color)

        rows = dataset.get_samples(2)
        with patch.object(dataset, "_load_thumbnail", side_effect=load_thumbnail):
            replacement = dataset.get_next_available_sample()

        self.assertEqual([row["video_id"] for row in rows], ["video-a", "video-b"])
        self.assertEqual(replacement["video_id"], "video-c")
        self.assertIsInstance(replacement["image"], Image.Image)

    def test_http_image_values_decode_to_rgb(self) -> None:
        payload = BytesIO()
        Image.new("RGBA", (7, 5), "navy").save(payload, format="PNG")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return payload.getvalue()

        dataset = VisualGenome.__new__(VisualGenome)
        with patch("dataset.hf_common.urlopen", return_value=FakeResponse()):
            image = dataset._coerce_image("https://example.test/image.png")

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (7, 5))

    def test_visual_cot_uses_available_repository_and_single_data_file(self) -> None:
        with patch("dataset.hf_common.load_dataset", return_value=_Rows()) as loader:
            VisualCoT()

        loader.assert_called_once_with(
            "deepcs233/Visual-CoT",
            split="train",
            streaming=True,
            data_files="viscot_363k.json",
        )

    def test_visual_cot_standardizes_conversation_schema(self) -> None:
        with patch("dataset.hf_common.load_dataset", return_value=_Rows()):
            dataset = VisualCoT()
        row = dataset._standardize_row(
            {
                "image": ["images/sample.jpg"],
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nWhat is shown? Please provide the bounding "
                            "box coordinate of the region that can help you answer "
                            "the question better."
                        ),
                    },
                    {"from": "gpt", "value": "[0.1, 0.2, 0.3, 0.4]"},
                    {"from": "human", "value": "<image>"},
                    {"from": "gpt", "value": "A bicycle."},
                ],
            }
        )

        self.assertEqual(row["question"], "What is shown?")
        self.assertNotIn("bounding box", row["question"].casefold())
        self.assertEqual(row["answers"], ["A bicycle."])
        self.assertEqual(dataset.get_answers_from_row(row), ["A bicycle."])

    def test_visual_genome_draws_the_selected_region(self) -> None:
        dataset = VisualGenome.__new__(VisualGenome)
        dataset.image_keys = ("img_context",)
        image = Image.new("RGB", (20, 20), "white")

        highlighted = dataset.get_image_from_row(
            {"img_context": image, "X": 2, "Y": 3, "W": 8, "H": 7}
        )

        self.assertEqual(highlighted.getpixel((2, 3)), (255, 0, 0))
        self.assertEqual(highlighted.getpixel((19, 19)), (255, 255, 255))

    def test_visual_cot_can_reuse_a_source_dataset_image(self) -> None:
        source_image = Image.new("RGB", (9, 7), "teal")
        source_rows = _Rows(
            [{"filename": "1000092795.jpg", "image": source_image}]
        )
        with patch("dataset.hf_common.load_dataset", return_value=_Rows()):
            dataset = VisualCoT()
        with patch("dataset.visual_cot.load_dataset", return_value=source_rows) as loader:
            image = dataset._load_source_image(
                {"dataset": "flickr30k"},
                "cot/flickr30k/1000092795.jpg",
            )

        self.assertEqual(image.size, (9, 7))
        loader.assert_called_once_with(
            "nlphuji/flickr30k",
            split="test",
            streaming=True,
            revision="refs/convert/parquet",
        )

    def test_lsun_indexes_multiple_official_scene_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for category in ("bedroom", "bridge"):
                folder = root / "train" / category
                folder.mkdir(parents=True)
                Image.new("RGB", (5, 5), category == "bridge" and "gray" or "white").save(folder / "sample.jpg")

            dataset = LSUN(data_dir=root)
            rows = dataset.get_samples(10)

        self.assertEqual({row["label"] for row in rows}, {"bedroom", "bridge"})
        self.assertGreater(len(dataset.labels), 1)


if __name__ == "__main__":
    unittest.main()
