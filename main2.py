# main.py
import os
import traceback

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from data import (
    BaseDataset,
    Cityscapes,
    Flickr30k,
    FlyingThings3D,
    ImageNet1k,
    INaturalist,
    MSCOCO,
    UCF101,
)
from models import (
    BLOOM,
    Baichuan2,
    DBRX,
    Falcon,
    Gemma,
    GPT4,
    Llava,
    OLMoE,
    Orion14B,
)


def make_prompt(labels: list[str]) -> str:
    # Keep prompt concise but strict.
    # NOTE: If labels get very large, this becomes a long prompt.
    label_set = ", ".join(labels)
    return (
        "USER: <image>\n"
        "Return exactly ONE label from this list (one item only, no extra words):\n"
        f"{label_set}\n"
        "ASSISTANT:"
    )


def main() -> int:
    N = 2
    LABEL_SAMPLE_SIZE = 64

    try:
        ds = MSCOCO(streaming=True)
        rows_for_labels = ds.get_samples(max(N, LABEL_SAMPLE_SIZE))
    except Exception as exc:
        print("\n[ERROR] Failed while loading MSCOCO data.")
        print(f"Reason: {exc.__class__.__name__}: {exc}")
        print("Hint: This often happens when network access to Hugging Face is blocked.")
        print("Full traceback:")
        print(traceback.format_exc())
        return 1

    rows = rows_for_labels[:N]

    # Build labels from dataset-provided taxonomy fields and/or text fallback.
    labels = ds.get_labels(rows_for_labels)
    print(f"Selected {N} images")
    print(f"Extracted {len(labels)} candidate labels from selected rows")
    print("First 30 labels:", labels[:30])
    try:
        model = DBRX(max_new_tokens=16)
        print("model loaded.")
    except Exception as exc:
        print("\n[ERROR] Failed while loading model.")
        print(f"Reason: {exc.__class__.__name__}: {exc}")
        print("Full traceback:")
        print(traceback.format_exc())
        return 1


    prompt = make_prompt(labels)

    for i, row in enumerate(rows):
        try:
            image = ds.get_image_from_row(row)

            # Extract candidate labels for this image
            nouns_this_image = ds.get_labels_img(row)

            # Normalize them
            nouns_norm = {ds.normalize_text(n) for n in nouns_this_image}

            print(f"\nImage {i+1}/{N}")
            pred = model.predict(image, prompt).strip()
            pred_norm = ds.normalize_text(pred)

            print("Prediction:", pred)

            if pred_norm in nouns_norm:
                print("Correct")
            else:
                print("Incorrect")
                print("Valid labels for this image:", sorted(nouns_norm))
        except Exception as exc:
            print(f"\n[ERROR] Failed during inference for image {i+1}/{N}.")
            print(f"Reason: {exc.__class__.__name__}: {exc}")
            print("Full traceback:")
            print(traceback.format_exc())
            return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print(f"\nProgram finished with exit code {exit_code}.")
