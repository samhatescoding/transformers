"""Upload JSONL data and explicitly submit a GPT-4o vision fine-tuning job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI, PermissionDeniedError


MODEL = "gpt-4o-2024-08-06"


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"File does not exist: {path}")
    if path.suffix.lower() != ".jsonl":
        raise SystemExit(f"Expected a .jsonl file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-file", required=True, type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--suffix")
    parser.add_argument(
        "--confirm-submit",
        action="store_true",
        help="Acknowledge that this creates a billable OpenAI fine-tuning job.",
    )
    args = parser.parse_args()

    _require_file(args.training_file)
    if args.validation_file is not None:
        _require_file(args.validation_file)

    if not args.confirm_submit:
        raise SystemExit(
            "No job submitted. Review the data and add --confirm-submit to "
            "upload files and create a billable fine-tuning job."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    client = OpenAI()
    with args.training_file.open("rb") as training_source:
        training_file = client.files.create(file=training_source, purpose="fine-tune")

    request: dict[str, str] = {
        "model": MODEL,
        "training_file": training_file.id,
    }
    validation_id = None
    if args.validation_file is not None:
        with args.validation_file.open("rb") as validation_source:
            validation_file = client.files.create(
                file=validation_source, purpose="fine-tune"
            )
        validation_id = validation_file.id
        request["validation_file"] = validation_id
    if args.suffix:
        request["suffix"] = args.suffix

    try:
        job = client.fine_tuning.jobs.create(**request)
    except PermissionDeniedError as exc:
        body = getattr(exc, "body", None) or {}
        error = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(error, dict) and error.get("code") == "training_not_available":
            print(
                json.dumps(
                    {
                        "status": "training_not_available",
                        "model": MODEL,
                        "training_file": training_file.id,
                        "validation_file": validation_id,
                        "message": (
                            "OpenAI rejected creation of a new fine-tuning job "
                            "for this organization. The uploaded files may remain "
                            "in project storage and can be deleted if unused."
                        ),
                    },
                    indent=2,
                )
            )
            raise SystemExit(1) from None
        raise
    print(
        json.dumps(
            {
                "job_id": job.id,
                "status": job.status,
                "model": MODEL,
                "training_file": training_file.id,
                "validation_file": validation_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
