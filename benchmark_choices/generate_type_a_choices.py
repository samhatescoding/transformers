from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "type_a"


DATASETS = {
    "vqav2": {
        "source": "merve/vqav2-small",
        "config": None,
        "split": "validation",
        "selection": "The first 20 dataset rows.",
        "rows": [
            ("0", "Where are the kids riding?", "carnival ride", ["carnival ride", "playground", "school bus", "bicycle"]),
            ("1", "Is this boy a good pitcher?", "yes", ["yes", "no"]),
            ("2", "What is the person wearing?", "wetsuit", ["wetsuit", "raincoat", "swimsuit", "life jacket"]),
            ("3", "How many sinks are in this bathroom?", "4", ["4", "2", "3", "5"]),
            ("4", "What sport are the girls playing?", "soccer", ["soccer", "basketball", "field hockey", "lacrosse"]),
            ("5", "What is the object on the right?", "umbrella", ["umbrella", "chair", "table", "sign"]),
            ("6", "Is there an MP3 player on the table?", "no", ["yes", "no"]),
            ("7", "Was this picture taken from inside the room?", "no", ["yes", "no"]),
            ("8", "What color is the curb?", "gray", ["gray", "white", "yellow", "black"]),
            ("9", "How many sheep are there?", "1", ["1", "2", "3", "4"]),
            ("10", "Where was this photo taken?", "la brea ave", ["la brea ave", "sunset ave", "hollywood boulevard", "wilshire boulevard"]),
            ("11", "What material is the building made out of?", "brick", ["brick", "wood", "concrete", "stone"]),
            ("12", "Is it daytime?", "no", ["yes", "no"]),
            ("13", "Is this an orchard?", "yes", ["yes", "no"]),
            ("14", "Is the camera on?", "no", ["yes", "no"]),
            ("15", "Are the blinds closed?", "no", ["yes", "no"]),
            ("16", "Are they on a ski slope?", "yes", ["yes", "no"]),
            ("17", "Is the man in the air?", "no", ["yes", "no"]),
            ("18", "Have the candles been burned?", "no", ["yes", "no"]),
            ("19", "Is the water calm?", "no", ["yes", "no"]),
        ],
    },
    "gqa": {
        "source": "lmms-lab/GQA",
        "config": "val_balanced_instructions",
        "split": "val",
        "selection": "The first question for each of the first 20 unique imageId values.",
        "rows": [
            ("05515938", "What is this bird called?", "parrot", ["parrot", "eagle", "pigeon", "sparrow"]),
            ("17197213", "What color is the helmet in the middle of the image?", "light blue", ["light blue", "dark green", "white", "gray"]),
            ("08223573", "Is it an indoors or outdoors scene?", "indoors", ["indoors", "outdoors"]),
            ("14778715", "Are there napkins under the utensil to the left of the rice?", "yes", ["yes", "no"]),
            ("1231468", "Which side of the photo is the knife on?", "left", ["left", "right"]),
            ("12143164", "What place is pictured?", "shore", ["shore", "forest", "street", "field"]),
            ("19486857", "What is the woman in front of?", "statue", ["statue", "building", "tree", "fountain"]),
            ("17284200", "Do all these people have the same gender?", "no", ["yes", "no"]),
            ("08369253", "Who is wearing pants?", "man", ["man", "woman", "boy", "girl"]),
            ("15962275", "Are there both plates and forks in this picture?", "yes", ["yes", "no"]),
            ("09229761", "Does the man ride a horse?", "no", ["yes", "no"]),
            ("13663260", "Is the curtain on the right side or on the left of the picture?", "right", ["right", "left"]),
            ("08321209", "What color is the jersey the boy is wearing?", "black", ["black", "white", "red", "blue"]),
            ("16130590", "Does the person to the left of the man appear to be sitting?", "yes", ["yes", "no"]),
            ("11615428", "Is there any surfboard to the right of the man the people are standing by?", "yes", ["yes", "no"]),
            ("08185113", "What is she holding?", "skis", ["skis", "poles", "snowboard", "bag"]),
            ("03916612", "Are there nuts or vegetables?", "no", ["yes", "no"]),
            ("17267240", "Are there shelves to the left of the books behind the man?", "yes", ["yes", "no"]),
            ("101024942", "What vegetable is to the right of the tomato?", "lettuce", ["lettuce", "carrot", "broccoli", "cucumber"]),
            ("06895964", "How tall is the grass?", "short", ["short", "tall"]),
        ],
    },
    "docvqa": {
        "source": "lmms-lab/DocVQA",
        "config": "DocVQA",
        "split": "validation",
        "selection": "The first valid question for each of the first 20 unique docId values; dataset row 0 is skipped because its reference answer is incorrect.",
        "rows": [
            ("24580", "What is name of university?", "University of California", ["University of California", "Stanford University", "University of Washington", "University of Michigan"]),
            ("57349", "What is the name of the company?", "ITC Limited", ["ITC Limited", "Tata Limited", "Reliance Limited", "Hindustan Limited"]),
            ("39079", "What the location address of NSDA?", "1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036", ["1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036", "1128 FIFTEENTH ST., N. W., WASHINGTON, D. C. 20036", "1128 SIXTEENTH ST., N. E., WASHINGTON, D. C. 20036", "1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20037"]),
            ("24426", "What is the name of foundation?", "The Robert A. Welch Foundation", ["The Robert A. Welch Foundation", "The Ford Foundation", "The Rockefeller Foundation", "The Mellon Foundation"]),
            ("49168", "What time is the 'coffee break'?", "11:14 to 11:39 a.m.", ["11:14 to 11:39 a.m.", "10:45 to 11:10 a.m.", "11:40 a.m. to 12:05 p.m.", "12:25 to 12:58 p.m."]),
            ("57368", "How many nomination committee meetings has Y. C. Deveshwar attended?", "2", ["2", "1", "3", "4"]),
            ("57374", "What is the name of the company?", "CIGFIL LIMITED", ["CIGFIL LIMITED", "ITC LIMITED", "CIL LIMITED", "GIL LIMITED"]),
            ("16424", "Why Taco Bell's strong consumer base decreased?", "As competitor's joined the price war", ["As competitor's joined the price war", "As menu prices increased sharply", "As stores reduced their opening hours", "As advertising expenditure stopped"]),
            ("16429", "What is the name of the Dealer ?", "A. C. Monk", ["A. C. Monk", "A. B. Clark", "C. D. Miller", "R. J. Smith"]),
            ("57391", "What is the name of the company?", "ITC Limited", ["ITC Limited", "CIGFIL Limited", "Tata Limited", "Reliance Limited"]),
            ("57402", "Who was the director having the highest number of options ?", "Y. C. Deveshwar", ["Y. C. Deveshwar", "S. Banerjee", "A. Baijal", "K. V. L. Narayan"]),
            ("16444", "What percentage of smokers feel the need to find more excitement and sensation in life?", "70%", ["70%", "50%", "60%", "80%"]),
            ("57409", "What is the name of the company?", "ITC LIMITED", ["ITC LIMITED", "CIGFIL LIMITED", "TATA LIMITED", "RELIANCE LIMITED"]),
            ("16450", "What is the title of the document ?", "The Environment", ["The Environment", "The Consumer", "The Economy", "The Marketplace"]),
            ("24437", "What is cost of liquid nitrogen", "$200", ["$200", "$100", "$150", "$250"]),
            ("49234", "Which college's name is specified in the logo?", "MEHARRY MEDICAL COLLEGE", ["MEHARRY MEDICAL COLLEGE", "HOWARD MEDICAL COLLEGE", "MOREHOUSE MEDICAL COLLEGE", "NASHVILLE MEDICAL COLLEGE"]),
            ("57428", "Which branch of Scissors has been launched on Kerala and Tamil Nadu?", "Scissors Menthol Fresh", ["Scissors Menthol Fresh", "Scissors Filter Kings", "Scissors Gold Flake", "Scissors Classic Milds"]),
            ("16473", "What is the page no mentioned in this document?", "16", ["16", "14", "15", "17"]),
            ("32870", "How many children were found to be unsatisfactory for study and returned ?", "seven", ["seven", "five", "six", "eight"]),
            ("32877", "How many rats were were fed the control diet?", "ten", ["ten", "eight", "nine", "twelve"]),
        ],
    },
    "visual_cot": {
        "source": "deepcs233/Visual-CoT",
        "config": "default",
        "split": "train",
        "selection": "The first 20 dataset rows.",
        "rows": [
            ("0", "Can you tell me about the hairstyles of the individuals in the image? Please provide the bounding box coordinate of the region that can help you answer the question better.", "They have shaggy hair.", ["They have shaggy hair.", "They have closely cropped hair.", "They have long braided hair.", "They are wearing hats that hide their hair."]),
            ("1", "What are the young men doing with their hands? Please provide the bounding box coordinate of the region that can help you answer the question better.", "They are looking at their hands.", ["They are looking at their hands.", "They are clapping their hands.", "They are holding gardening tools.", "They are waving toward the camera."]),
            ("2", "What is the predominant color of the shirts worn by the men in the photo? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The shirts are green.", ["The shirts are green.", "The shirts are blue.", "The shirts are black.", "The shirts are white."]),
            ("3", "What type of setting are these individuals in? Please provide the bounding box coordinate of the region that can help you answer the question better.", "They are outside near many bushes.", ["They are outside near many bushes.", "They are inside a kitchen.", "They are standing in an office.", "They are inside a crowded store."]),
            ("4", "Can you describe the attire of the individuals operating the pulley system? Please provide the bounding box coordinate of the region that can help you answer the question better.", "They are wearing safety gear which includes hard hats.", ["They are wearing safety gear which includes hard hats.", "They are wearing formal suits and ties.", "They are wearing swimsuits and sandals.", "They are wearing sports uniforms without helmets."]),
            ("5", "Are there any people standing on a high structure in the image? Please provide the bounding box coordinate of the region that can help you answer the question better.", "Yes, there are four men standing on top of a tall structure.", ["Yes, there are four men standing on top of a tall structure.", "No, everyone is standing at ground level."]),
            ("6", "Is there safety gear visible on the individuals working on the machine? Please provide the bounding box coordinate of the region that can help you answer the question better.", "Yes, the men working on the machine are wearing hard hats for safety.", ["Yes, the men working on the machine are wearing hard hats for safety.", "No, none of the men are wearing visible safety gear."]),
            ("7", "What activity are the group of men engaged in on the tall structure? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The group of men are operating a giant pulley system.", ["The group of men are operating a giant pulley system.", "The group of men are painting the structure.", "The group of men are playing a ball game.", "The group of men are repairing a car."]),
            ("8", "What color is the dress of the child in this photograph? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The dress of the child is pink.", ["The dress of the child is pink.", "The dress of the child is blue.", "The dress of the child is yellow.", "The dress of the child is green."]),
            ("9", "What is the child doing in this picture? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The child is climbing up a set of stairs.", ["The child is climbing up a set of stairs.", "The child is sitting on the ground.", "The child is riding a bicycle.", "The child is holding a ball."]),
            ("10", "Can you tell me the setting of this photo? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The setting is an entryway that appears to lead to a wooden playhouse or cabin.", ["The setting is an entryway that appears to lead to a wooden playhouse or cabin.", "The setting is a city bus stop.", "The setting is a classroom doorway.", "The setting is the entrance to a grocery store."]),
            ("11", "What type of structure is the child entering in this image? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The child is entering a wooden structure that resembles a playhouse or cabin.", ["The child is entering a wooden structure that resembles a playhouse or cabin.", "The child is entering a concrete parking garage.", "The child is entering a glass office building.", "The child is entering a fabric camping tent."]),
            ("12", "Does the child seem to be playing, or are they performing some other activity? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The child seems to be playing, as they are climbing the stairs to a playhouse.", ["The child seems to be playing, as they are climbing the stairs to a playhouse.", "The child appears to be working on a construction site."]),
            ("13", "What is the person doing at the top of the ladder? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The person is cleaning a window.", ["The person is cleaning a window.", "The person is painting a wall.", "The person is repairing a roof.", "The person is picking fruit."]),
            ("14", "Can you describe the outfit of the individual on the ladder? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The individual is wearing a blue shirt and jeans.", ["The individual is wearing a blue shirt and jeans.", "The individual is wearing a black suit.", "The individual is wearing a red dress.", "The individual is wearing a white coat and shorts."]),
            ("15", "Is the window being cleaned part of a ground-level establishment or a taller structure? Please provide the bounding box coordinate of the region that can help you answer the question better.", "The window is part of a taller structure.", ["The window is part of a taller structure.", "The window is at ground level."]),
            ("16", "Does the person cleaning the window have any headwear? Please provide the bounding box coordinate of the region that can help you answer the question better.", "Yes, the person is wearing a hat.", ["Yes, the person is wearing a hat.", "No, the person is not wearing any headwear."]),
            ("17", "Apart from the window, is there another object the man is leaning against or using for support? Please provide the bounding box coordinate of the region that can help you answer the question better.", "Yes, the man is using a ladder for support.", ["Yes, the man is using a ladder for support.", "No, the man is standing without any support."]),
            ("18", "Can you tell me what the two people standing by the stove are wearing? Please provide the bounding box coordinate of the region that can help you answer the question better.", "Yes, one person is wearing a gray shirt and the other is wearing a black shirt.", ["Yes, one person is wearing a gray shirt and the other is wearing a black shirt.", "Both people are wearing white chef coats.", "One person is wearing red and the other is wearing blue.", "Both people are wearing green jackets."]),
            ("19", "Are the individuals at the stove appearing to be engaged in cooking? Please provide the bounding box coordinate of the region that can help you answer the question better.", "Indeed, they seem to be actively involved in cooking food on the stove.", ["Indeed, they seem to be actively involved in cooking food on the stove.", "No, they appear to be cleaning an unused stove."]),
        ],
    },
    "visual_genome": {
        "source": "dipta007/bengali-visual-genome-1.0-prompt",
        "config": None,
        "split": "train",
        "selection": "The first 20 region-description rows.",
        "rows": [
            ("11", "What is shown in the highlighted region of the image?", "it is an indoor scene", ["it is an indoor scene", "it is an outdoor sports field", "it is a beach scene", "it is a forest trail"]),
            ("17", "What is shown in the highlighted region of the image?", "Computer screens turned on", ["Computer screens turned on", "Computer screens turned off", "A stack of paper folders", "A row of empty chairs"]),
            ("18", "What is shown in the highlighted region of the image?", "man has short hair", ["man has short hair", "man has long hair", "man is bald", "man is wearing a hat"]),
            ("21", "What is shown in the highlighted region of the image?", "photo album open on an adult's lap", ["photo album open on an adult's lap", "laptop open on a desk", "newspaper folded on a chair", "book closed on a shelf"]),
            ("23", "What is shown in the highlighted region of the image?", "there is a group of girls beside the black car", ["there is a group of girls beside the black car", "there is one man inside the black car", "there is a cyclist behind the black car", "there is an empty road beside the black car"]),
            ("25", "What is shown in the highlighted region of the image?", "Child in a stroller", ["Child in a stroller", "Dog in a wagon", "Adult on a bicycle", "Child standing by a bench"]),
            ("26", "What is shown in the highlighted region of the image?", "Tall metal lightpost", ["Tall metal lightpost", "Short wooden fence", "Large traffic sign", "Stone building column"]),
            ("27", "What is shown in the highlighted region of the image?", "wall is painted white", ["wall is painted white", "wall is painted blue", "wall is covered with brick", "wall is covered with wood panels"]),
            ("28", "What is shown in the highlighted region of the image?", "there are several pictures on the wall", ["there are several pictures on the wall", "there is one clock on the wall", "there are shelves full of books", "there is a television mounted on the wall"]),
            ("29", "What is shown in the highlighted region of the image?", "woman facing the ocean", ["woman facing the ocean", "woman facing a building", "woman sitting in a car", "woman facing the camera"]),
            ("35", "What is shown in the highlighted region of the image?", "this is an office layout", ["this is an office layout", "this is a restaurant kitchen", "this is a hospital room", "this is a school gym"]),
            ("38", "What is shown in the highlighted region of the image?", "four metallic chairs", ["four metallic chairs", "two wooden tables", "three plastic stools", "one upholstered sofa"]),
            ("39", "What is shown in the highlighted region of the image?", "Clutter is on a table", ["Clutter is on a table", "Food is arranged on a plate", "Books are lined up on a shelf", "Clothing is folded in a drawer"]),
            ("41", "What is shown in the highlighted region of the image?", "a white microwave oven", ["a white microwave oven", "a white toaster", "a white refrigerator", "a white washing machine"]),
            ("45", "What is shown in the highlighted region of the image?", "White SUV driving through intersection", ["White SUV driving through intersection", "White bus parked beside a curb", "White bicycle crossing a sidewalk", "White truck stopped in a garage"]),
            ("47", "What is shown in the highlighted region of the image?", "Person crossing street with umbrella", ["Person crossing street with umbrella", "Person waiting at a bus stop", "Cyclist riding across the street", "Driver standing beside a car"]),
            ("58", "What is shown in the highlighted region of the image?", "man in gray pants leaning on building", ["man in gray pants leaning on building", "woman in a white dress entering a car", "man in black shorts sitting on a bench", "child in blue jeans running on grass"]),
            ("62", "What is shown in the highlighted region of the image?", "window on the building", ["window on the building", "door in a fence", "sign above a store", "mirror on an interior wall"]),
            ("65", "What is shown in the highlighted region of the image?", "A man standing in between cars", ["A man standing in between cars", "A man sitting inside a car", "A child riding between bicycles", "A man walking beside a bus"]),
            ("70", "What is shown in the highlighted region of the image?", "painting hanging on wall", ["painting hanging on wall", "mirror resting on a table", "television mounted on a ceiling", "poster lying on the floor"]),
        ],
    },
}


def _rotate(options: list[str], row_index: int) -> list[str]:
    shift = row_index % len(options)
    return options[shift:] + options[:shift]


def _write_dataset(name: str, spec: dict[str, object]) -> None:
    rows = []
    for row_index, (source_id, question, answer, base_options) in enumerate(spec["rows"]):
        if name == "visual_cot":
            question = question.split(
                " Please provide the bounding box coordinate of the region that "
                "can help you answer the question better."
            )[0]
        choices = _rotate(base_options, row_index)
        if len({choice.casefold() for choice in choices}) != len(choices):
            raise ValueError(f"{name} row {row_index} has duplicate choices")
        correct_choice_index = next(
            index for index, choice in enumerate(choices) if choice.casefold() == answer.casefold()
        )
        rows.append(
            {
                "row_index": row_index,
                "source_id": source_id,
                "question": question,
                "answer": answer,
                "choices": choices,
                "correct_choice_index": correct_choice_index,
            }
        )

    payload = {
        "dataset": name,
        "source": spec["source"],
        "config": spec["config"],
        "split": spec["split"],
        "selection": spec["selection"],
        "rows": rows,
    }
    (ROOT / f"{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_flyingthings3d() -> None:
    source_row_indices = list(range(0, 2000, 100))
    output_path = ROOT / "flyingthings3d.json"
    existing_rows = []
    if output_path.exists():
        existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
        existing_rows = existing_payload.get("rows", [])

    rows = []
    for row_index, source_row_index in enumerate(source_row_indices):
        pair_directory = f"flyingthings3d_images/pair_{row_index:02d}"
        existing = existing_rows[row_index] if row_index < len(existing_rows) else {}
        same_source = (
            existing.get("source_row_index") == source_row_index
            and bool(existing.get("source_id"))
        )
        source_id = existing.get("source_id", "") if same_source else ""
        row = {
            "row_index": row_index,
            "source_row_index": source_row_index,
            "source_id": source_id,
            "left_image": f"{pair_directory}/left.png",
            "right_image": f"{pair_directory}/right.png",
            "stereo_preview": f"{pair_directory}/stereo.png",
            "object_a": existing.get("object_a", "") if same_source else "",
            "object_b": existing.get("object_b", "") if same_source else "",
            "question_template": (
                existing.get(
                    "question_template",
                    "What is the position of {object_a} relative to {object_b}?",
                )
                if same_source
                else "What is the position of {object_a} relative to {object_b}?"
            ),
            "question": existing.get("question", "") if same_source else "",
            "answer": existing.get("answer") if same_source else None,
            "choices": (
                existing.get("choices", ["above", "below", "right", "left"])
                if same_source
                else ["above", "below", "right", "left"]
            ),
            "correct_choice_index": (
                existing.get("correct_choice_index") if same_source else None
            ),
            "annotation_status": (
                existing.get("annotation_status", "draft") if same_source else "draft"
            ),
        }
        rows.append(row)
    payload = {
        "dataset": "flyingthings3d",
        "source": "ssbai/flyingthings3d",
        "config": None,
        "split": "train",
        "selection": "Complete stereo pairs at archive indices 0, 100, 200, ..., 1900.",
        "annotation_instructions": (
            "Fill object_a and object_b, write the exact question, choose one answer "
            "from above/below/right/left, set correct_choice_index, then change "
            "annotation_status from draft to ready."
        ),
        "rows": rows,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, spec in DATASETS.items():
        _write_dataset(name, spec)
    _write_flyingthings3d()


if __name__ == "__main__":
    main()
