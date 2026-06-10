from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import ConceptualCaptions, OpenVid1M
from dataset.sharegpt4o_image import ShareGPT4oImageEdit


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOTS = {
    "conceptual_captions": ROOT / "type_g",
    "openvid1m": ROOT / "type_g",
    "sharegpt4o_image_edit": ROOT / "type_e",
}

CURATIONS = {
    "conceptual_captions": {
        "dataset": lambda: ConceptualCaptions(split="validation", streaming=True),
        "substitutions": [
            (("photography", "portraiture"), ("in pictures", "in photographs")),
            (("river", "lake"), ("snowy", "rainy")),
            (("sign", "billboard"), ("repaired", "painted")),
            (("player", "designer"), ("computer screen", "television screen")),
            (("green", "blue"), ("globe", "basketball")),
            (("stone cottage", "wooden cottage"), ("bedroom", "bathroom")),
            (("garbage truck", "delivery truck"), ("film", "photograph")),
            (("young woman", "older woman"), ("beach", "pier")),
            (("party", "picnic"), ("cherry blossoms", "autumn leaves")),
            (("classical music", "jazz music"), ("rising stars", "veteran performers")),
            (("man", "woman"), ("debris", "luggage")),
            (("nails", "shoes"), ("castle", "palace")),
            (("vegetable", "fruit"), ("simple", "spicy")),
            (("issue # 4b", "issue # 5b"), ("transformers", "robots")),
            (("little girl", "little boy"), ("bath", "shower")),
            (("police procedural", "medical drama"), ("street", "hospital")),
            (("new terminal", "old terminal"), ("setting sun", "rising sun")),
            (("job interview", "wedding"), ("makeup", "hairstyle")),
            (("father and son", "mother and daughter"), ("ice skates", "snowshoes")),
            (("dentist", "surgeon"), ("turbine", "laser")),
        ],
    },
    "openvid1m": {
        "dataset": lambda: OpenVid1M(split="train", streaming=True),
        "substitutions": [
            (("black sweater", "navy sweater"), ("serious", "thoughtful")),
            (("purple van", "blue van"), ("pink rims", "silver rims")),
            (("red baseball cap", "navy baseball cap"), ("11:10 PM", "10:10 PM")),
            (("pink shirt", "orange shirt"), ("spoon", "fork")),
            (("black cap with a red logo", "black cap with a blue logo"), ("partly cloudy", "clear")),
            (("gray baseball cap", "black baseball cap"), ("microbreweries", "coffee roasters")),
            (("blue bottle", "green bottle"), ("excavator", "bulldozer")),
            (("black muscle car", "dark blue muscle car"), ("snowy road", "wet road")),
            (("27th of January", "26th of January"), ("dark suit", "navy suit")),
            (("brown jacket", "black jacket"), ("cup of coffee", "cup of tea")),
            (("white shirt", "light blue shirt"), ("beige top", "pink top")),
            (("conical hat", "wide-brimmed hat"), ("fruits and vegetables", "flowers and herbs")),
            (("black dress", "navy dress"), ("white railing", "wooden railing")),
            (("circle formation", "two rows"), ("basketball", "soccer ball")),
            (("young boy", "young girl"), ('"JAZZ"', '"STARS"')),
            (("glasses", "sunglasses"), ("dumbbells", "a barbell")),
            (("two women", "two men"), ("cups and a book", "notebooks and a tablet")),
            (("blue shirt", "green shirt"), ("bottle of water", "cup of coffee")),
            (("purple convertible", "red convertible"), ("black interior", "tan interior")),
            (("gray t-shirt", "black t-shirt"), ("purple sports car", "blue sports car")),
        ],
    },
    "sharegpt4o_image_edit": {
        "dataset": lambda: ShareGPT4oImageEdit(split="train", streaming=True),
        "substitutions": [
            (("white background", "light gray background"), ("transparent", "solid white")),
            (("'200'", "'300'"), ("laptop", "tablet")),
            (("jet black tiles", "deep navy tiles"), ("pale rose tiles", "pale blue tiles")),
            (("blazing magenta", "electric cyan"), ("graphic black", "deep navy")),
            (("dark, rich navy", "charcoal black"), ("soft, internal glow", "bright neon glow")),
            (("black calligraphy", "dark red calligraphy"), ("textured silk", "aged parchment")),
            (("soft watercolor painting", "soft gouache painting"), ("vibrant red text circle", "vibrant blue text circle")),
            (("'Sweet'", "'Dream'"), ("backpack", "bag")),
            (("dark grey area", "light beige area"), ("vertical line and chevron designs", "dot and wave designs")),
            (("deep blues", "deep greens"), ("shimmering metallics", "matte gold accents")),
            (("blue tones", "green tones"), ("large blue chip lettering", "large red chip lettering")),
            (("mottled teal background", "mottled blue background"), ("orange rim", "red rim")),
            (("chrome", "brushed steel"), ("crisp definition", "soft definition")),
            (("electric blue", "electric purple"), ("light cyan", "light green")),
            (("light blue sea glass", "pale green sea glass"), ("rose gold inlays", "silver inlays")),
            (("collaboration", "innovation"), ("dynamism", "reliability")),
            (("Ford automobile", "Chevrolet automobile"), ("black and white", "sepia toned")),
            (("deep blue", "deep green"), ("neon pink", "neon orange")),
            (("lighter red tone", "lighter blue tone"), ("bold", "subtle")),
            (("gummy bears", "jelly beans"), ("multicolored sparkling confetti", "golden sparkling confetti")),
        ],
    },
}


def replace_once(text: str, source: str, replacement: str) -> str:
    pattern = re.compile(re.escape(source), flags=re.IGNORECASE)
    changed, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"Expected exactly one {source!r} in {text!r}.")
    return changed


def main() -> None:
    for dataset_name, config in CURATIONS.items():
        output_root = OUTPUT_ROOTS[dataset_name]
        output_root.mkdir(parents=True, exist_ok=True)
        dataset = config["dataset"]()
        rows = dataset.get_samples(20)
        substitutions = config["substitutions"]
        if len(rows) != 20 or len(substitutions) != 20:
            raise ValueError(
                f"{dataset_name} requires exactly 20 rows and substitution pairs."
            )

        output_rows = []
        for row_index, (row, pair) in enumerate(zip(rows, substitutions)):
            prompt = dataset.get_answer_from_row(row).strip()
            (source_a, replacement_a), (source_b, replacement_b) = pair
            changed_a = replace_once(prompt, source_a, replacement_a)
            changed_b = replace_once(prompt, source_b, replacement_b)
            changed_both = replace_once(changed_a, source_b, replacement_b)
            output_rows.append(
                {
                    "row_index": row_index,
                    "correct_prompt": prompt,
                    "distractors": [changed_a, changed_b, changed_both],
                    "attributes": [
                        {"source": source_a, "replacement": replacement_a},
                        {"source": source_b, "replacement": replacement_b},
                    ],
                }
            )

        path = output_root / f"{dataset_name}.json"
        path.write_text(
            json.dumps({"rows": output_rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(output_rows)} rows to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
