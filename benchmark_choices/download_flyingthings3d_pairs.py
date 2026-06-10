from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset.flyingthings3d import FlyingThings3D


ROOT = Path(__file__).resolve().parent / "type_a"
CHOICES_PATH = ROOT / "flyingthings3d.json"


def _make_stereo_preview(left: Image.Image, right: Image.Image) -> Image.Image:
    label_height = 28
    width = left.width + right.width
    height = max(left.height, right.height) + label_height
    preview = Image.new("RGB", (width, height), "white")
    preview.paste(left, (0, label_height))
    preview.paste(right, (left.width, label_height))
    draw = ImageDraw.Draw(preview)
    draw.text((8, 7), "Left", fill="black")
    draw.text((left.width + 8, 7), "Right", fill="black")
    return preview


def _make_overview(previews: list[Image.Image]) -> Image.Image:
    thumbnail_width = 480
    label_height = 24
    thumbnails = []
    for row_index, preview in enumerate(previews):
        thumbnail = preview.copy()
        thumbnail.thumbnail((thumbnail_width, 160))
        card = Image.new("RGB", (thumbnail_width, thumbnail.height + label_height), "white")
        card.paste(thumbnail, ((thumbnail_width - thumbnail.width) // 2, label_height))
        ImageDraw.Draw(card).text((8, 5), f"Pair {row_index:02d}", fill="black")
        thumbnails.append(card)

    columns = 2
    rows = (len(thumbnails) + columns - 1) // columns
    cell_height = max(thumbnail.height for thumbnail in thumbnails)
    overview = Image.new("RGB", (columns * thumbnail_width, rows * cell_height), "#dddddd")
    for index, thumbnail in enumerate(thumbnails):
        x = (index % columns) * thumbnail_width
        y = (index // columns) * cell_height
        overview.paste(thumbnail, (x, y))
    return overview


def main() -> None:
    payload = json.loads(CHOICES_PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    source_row_indices = [row["source_row_index"] for row in rows]
    dataset_rows = FlyingThings3D().get_samples_at_indices(source_row_indices)
    if len(dataset_rows) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} stereo pairs, received {len(dataset_rows)}.")

    previews = []
    for choice_row, dataset_row in zip(rows, dataset_rows):
        if choice_row["source_id"] and choice_row["source_id"] != dataset_row["id"]:
            raise RuntimeError(
                f"Dataset order changed at row {choice_row['row_index']}: "
                f"{dataset_row['id']!r} != {choice_row['source_id']!r}"
            )
        choice_row["source_id"] = dataset_row["id"]

        left = dataset_row["source_image"].convert("RGB")
        right = dataset_row["target_image"].convert("RGB")
        pair_directory = ROOT / Path(choice_row["left_image"]).parent
        pair_directory.mkdir(parents=True, exist_ok=True)
        left.save(pair_directory / "left.png")
        right.save(pair_directory / "right.png")
        preview = _make_stereo_preview(left, right)
        preview.save(pair_directory / "stereo.png")
        previews.append(preview)
        print(f"Saved pair {choice_row['row_index']:02d}: {choice_row['source_id']}")
    _make_overview(previews).save(ROOT / "flyingthings3d_images" / "overview.jpg", quality=90)
    CHOICES_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
