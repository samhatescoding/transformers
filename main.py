from data import MSCOCO
from models import LlavaModel


def main():
    dataset = MSCOCO(split="validation", streaming=True)
    model = LlavaModel(max_new_tokens=5)

    label_set = ", ".join(dataset.labels)

    samples = dataset.get_samples(3)

    for i, row in enumerate(samples):
        print(f"\nImage {i+1}")

        image = dataset.get_image_from_row(row)

        prompt = (
            "USER: <image>\n"
            "Return exactly ONE label from this list (one item only):\n"
            f"{label_set}\n"
            "ASSISTANT:"
        )

        prediction = model.predict(image, prompt)

        print("Raw prediction:", prediction)

        if dataset.is_valid_label(prediction):
            print("Valid COCO label")
        else:
            print("Not a valid COCO label")


if __name__ == "__main__":
    main()
