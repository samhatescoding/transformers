"""Retrieve the state of an OpenAI fine-tuning job."""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    job = OpenAI().fine_tuning.jobs.retrieve(args.job_id)
    error = None
    if job.error:
        error = {
            "code": job.error.code,
            "message": job.error.message,
            "param": job.error.param,
        }
    print(
        json.dumps(
            {
                "job_id": job.id,
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model,
                "trained_tokens": job.trained_tokens,
                "error": error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
