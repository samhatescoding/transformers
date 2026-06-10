"""Build OpenAI vision fine-tuning JSONL from a simple labeled-image manifest."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image


MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_MODES = {"RGB", "RGBA"}
DETAIL_VALUES = {"low", "high", "auto"}


def _image_data_url(image_path: Path) -> str:
    if not image_path.is_file():
        raise ValueError(f"image file does not exist: {image_path}")
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds 10 MB: {image_path}")

    with Image.open(image_path) as image:
        if image.format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported format {image.format!r} for {image_path}; "
                "use JPEG, PNG, or WEBP"
            )
        if image.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"unsupported image mode {image.mode!r} for {image_path}; "
                "use RGB or RGBA"
            )

    mime_type = mimetypes.guess_type(image_path.name)[0]
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(image_path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"cannot determine supported image MIME type: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _required_text(record: dict[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field!r} must be a non-empty string")
    return value.strip()


def _build_example(
    record: dict[str, Any],
    line_number: int,
    image_base_dir: Path,
    default_system: str | None,
) -> dict[str, Any]:
    prompt = _required_text(record, "prompt", line_number)
    answer = _required_text(record, "answer", line_number)

    has_path = "image_path" in record
    has_url = "image_url" in record
    if has_path == has_url:
        raise ValueError(
            f"line {line_number}: provide exactly one of 'image_path' or 'image_url'"
        )

    if has_path:
        relative_path = _required_text(record, "image_path", line_number)
        image_url = _image_data_url((image_base_dir / relative_path).resolve())
    else:
        image_url = _required_text(record, "image_url", line_number)
        if not (
            image_url.startswith("https://")
            or image_url.startswith("http://")
            or image_url.startswith("data:image/")
        ):
            raise ValueError(
                f"line {line_number}: image_url must be an HTTP(S) or image data URL"
            )

    detail = record.get("detail", "low")
    if detail not in DETAIL_VALUES:
        raise ValueError(
            f"line {line_number}: detail must be one of {sorted(DETAIL_VALUES)}"
        )

    messages: list[dict[str, Any]] = []
    system = record.get("system", default_system)
    if system is not None:
        if not isinstance(system, str) or not system.strip():
            raise ValueError(f"line {line_number}: system must be a non-empty string")
        messages.append({"role": "system", "content": system.strip()})

    messages.extend(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": detail},
                    },
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    )
    return {"messages": messages}


def build_jsonl(
    manifest_path: Path,
    output_path: Path,
    image_base_dir: Path,
    default_system: str | None,
) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: each record must be an object")
            example = _build_example(
                record, line_number, image_base_dir, default_system
            )
            destination.write(json.dumps(example, ensure_ascii=True) + "\n")
            count += 1
    if count == 0:
        raise ValueError("manifest contains no examples")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--image-base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory for relative image_path entries (default: cwd).",
    )
    parser.add_argument("--default-system")
    args = parser.parse_args()

    count = build_jsonl(
        args.manifest,
        args.output,
        args.image_base_dir.resolve(),
        args.default_system,
    )
    print(f"Wrote {count} training examples to {args.output}")


if __name__ == "__main__":
    main()
