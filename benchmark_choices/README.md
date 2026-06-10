# Curated Benchmark Choices

This directory stores manually curated answer and prompt choices used by the
benchmark suite.

## Type A: answering questions

`type_a/` contains possible answers for the first 20 selected rows of
FlyingThings3D, Visual Genome, VQAv2, GQA, DocVQA, and Visual-CoT.

- Binary questions retain the natural two choices rather than adding invalid
  filler answers.
- Open questions have four type-matched choices, including the reference
  answer and three distractors.
- FlyingThings3D stores 20 actual left/right clean-pass stereo pairs under
  `type_a/flyingthings3d_images/`. Its JSON rows are manual annotation
  templates for questions such as "What is the position of the car relative
  to the chair?" with the fixed choices `above`, `below`, `right`, and `left`.
  Fill `object_a`, `object_b`, `question`, `answer`, and
  `correct_choice_index`, then set `annotation_status` to `ready`. Draft rows
  are ignored by the runtime.
  `python benchmark_choices/finalize_flyingthings3d_choices.py` copies a
  populated `question_template` into an empty `question`, validates each
  answer, calculates `correct_choice_index`, and marks all complete rows ready.
- Visual Genome uses an image with the selected region outlined in red and
  asks which description matches that highlighted region.
- GQA keeps only the first question associated with each image ID, so its
  first 20 benchmark rows contain 20 different images.
- DocVQA skips its incorrect first reference row and keeps only the first
  question associated with each document ID, so its 20 rows use 20 different
  document images.

Run `python benchmark_choices/generate_type_a_choices.py` to regenerate the
six JSON files and validate that every known answer occurs exactly once among
its choices. Existing FlyingThings3D draft or ready annotations are preserved.
Run `python benchmark_choices/download_flyingthings3d_pairs.py` to download
the 20 exact stereo pairs referenced by its JSON file.

## Types E and G: prompt reconstruction

The attribute catalog produces balanced four-choice prompt sets for the first
20 benchmark rows of each prompt-based dataset.

- `balanced_attributes.json` contains manually curated substitutions for 20
  attribute categories.
- `datasets.json` maps dataset names to a curation bank.

For each correct prompt, the generator finds two distinct attributes and
creates a 2x2 factorial set:

1. both original attributes;
2. attribute A changed;
3. attribute B changed;
4. both attributes changed.

Thus each original and replacement value appears in exactly two options.
Rows among the first 20 fail explicitly when two applicable substitutions
cannot be found, rather than falling back to an unbalanced choice set.

The row-specific files in `type_e/` and `type_g/` take precedence over the
generic attribute bank. `type_e/` contains HQ-Edit, ImgEdit, MagicBrush, and
ShareGPT-4o-Image Edit. `type_g/` contains BLIP3o-60k, Conceptual Captions,
DiffusionDB, OpenVid-1M, and ShareGPT4o-Image. Each row records:

- `correct_prompt`: both original attributes;
- `attributes`: the two source-to-replacement substitutions;
- `distractors[0]`: only attribute A changed;
- `distractors[1]`: only attribute B changed;
- `distractors[2]`: both attributes changed.

The folders contain 20 row-specific prompt sets for every Type E and Type G
benchmark.

Run `python benchmark_choices/generate_manual_distractors.py` to reselect the
attributes from the ordered curation rules and regenerate the distractors.
The three later additions can be refreshed from their exact source metadata
with `python benchmark_choices/generate_missing_manual_distractors.py`.
