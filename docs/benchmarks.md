# Benchmark Plan

This file lays out a benchmark plan for the 36 datasets described in `Datasets.pdf`.

The goal is to keep the plan practical for this repo:

- Reuse the benchmark style that already exists where possible.
- Prefer simple, easy-to-explain metrics.
- Use proxy tasks for strongly generative datasets when that makes the benchmark easier to run and compare.

Current repo status:

- Already implemented in some form: ImageNet-1k, MS COCO, Flickr30k, UCF101, FlyingThings3D, Cityscapes, iNaturalist.
- Reusable building block already present: a generic classification benchmark.
- Not yet implemented: the remaining datasets below.

Quick metric glossary:

- Accuracy: the fraction of examples the model gets right.
- Top-1 accuracy: the model only gets credit if its single best answer is correct.
- IoU, short for Intersection over Union: how much a predicted box or mask overlaps the true one. `1.0` is perfect overlap, `0.0` means no overlap.
- Precision: of the things the model predicted, how many were actually right.
- Recall: of the things that were truly there, how many the model found.
- F1: one number that balances precision and recall.
- AP or Average Precision: a standard detection score that rewards finding the right objects with good localization across confidence thresholds.
- BLEU: a text-overlap score for captions. Higher means the generated caption is closer to reference captions.
- OCR, short for optical character recognition: reading text that appears inside an image.
- Optical flow: the motion of pixels from one frame to the next in a video or frame pair.
- Endpoint error: the distance between the predicted motion vector and the true one in optical flow.
- AUROC: a ranking score for binary decisions such as real vs fake or normal vs anomalous. `1.0` is perfect and `0.5` is random guessing.
- Recall@k: whether the correct answer appears in the model's top `k` choices.

Benchmark type legend:

- `Classification`: choose the correct class label for an image or clip.
- `Detection`: predict bounding boxes for objects or phrases. Only box-based detection tasks count here.
- `Captioning`: generate, select, or match captions or prompts against visual content.
- `Visual Q&A`: answer a question grounded in an image or document.
- `Other`: anything that does not fit the four benchmark patterns above.

## 1. ImageNet-1k

Status: already covered by `benchmarks/imagenet1k.py`.

Plan:
- Measure single-label image classification.
- Use top-1 accuracy as the main score.
- Keep the existing "choose one label from a candidate list" setup.
- Report confusion between visually similar classes, because ImageNet has many fine-grained labels.
- Low-risk follow-up: add top-5 accuracy later if we want a more standard ImageNet report.

## 2. UCF101

Status: already covered by `benchmarks/ucf101.py`.

Plan:
- Measure action recognition from a small set of frames sampled from each clip.
- Use top-1 accuracy.
- Keep the current contact-sheet approach first, because it fits the repo's image-in, text-out interface.
- Record per-class accuracy so we can see which actions are consistently confused.
- Later upgrade path: add true video input if the model API supports it.

## 3. MS COCO

Status: already covered by `benchmarks/mscoco.py`.

### 3.1 Image Captioning (Captioning)

- Add a separate captioning benchmark because MS COCO also has human-written captions.
- Use BLEU first, since that already matches the repo's Flickr30k captioning approach.
- Keep captions short and compare against the available reference captions for each image.
- This is cheaper than detection because scoring a caption is simpler than matching and scoring many predicted boxes.

### 3.2 Object Detection (Detection)

- Measure object detection.
- Keep IoU-based matching and report precision, recall, F1, and mean IoU.
- Explain IoU in the benchmark output because it is central to understanding box quality.
- Keep prompts constrained to labels that are known to be present, which reduces prompt noise.
- Later upgrade path: add standard COCO AP if we want closer alignment with the official benchmark.
- This is more expensive because it requires parsing many outputs and scoring localization quality.

## 4. Flickr30k Entities

Status: partially covered by `benchmarks/flickr30k.py`, but only for captioning right now.

### 4.1 Image Captioning (Captioning)

- Keep caption generation as the first benchmark because it is already close to done.
- Use BLEU against the five reference captions.
- In the markdown and future UI, explain BLEU as "how similar the caption is to the human-written captions."

### 4.2 Phrase Grounding (Detection)

- Add phrase grounding, where the model must point to the region described by a phrase.
- Use IoU on bounding boxes.
- Start with one phrase at a time instead of scoring a whole caption worth of phrases in a single response.
- This is the part that best uses the "Entities" annotations.

## 5. FlyingThings3D

Status: already covered by `benchmarks/flyingthings3d.py`.

### 5.1 Optical Flow (Other)

- Measure optical flow on sampled points.
- Keep endpoint error and point accuracy as the main scores.
- Explain optical flow directly in the benchmark docs as "where each pixel moved between frame A and frame B."
- Keep point sampling instead of full dense flow first, because it is much easier to score and inspect.
- This stays cheaper than 5.2 by using only a sparse set of sampled points.

### 5.2 Disparity or Depth (Other)

- Add a second benchmark around disparity or depth if the loaded split exposes it cleanly.
- Explain disparity simply as "the difference in where the same point appears in the left and right views," which is used to infer depth.
- Use either a denser point grid or a wider set of stereo correspondences than 5.1.
- This would let FlyingThings3D test 3D understanding in addition to motion estimation.
- This phase is intentionally heavier than 5.1.

## 6. LSUN

Status: planned.

Plan:
- Measure scene classification, such as bedroom, classroom, or church.
- Use top-1 accuracy.
- Reuse the generic classification benchmark.
- Keep the label set limited to the LSUN categories present in the chosen split.
- This should be one of the easiest new benchmarks to add after the existing classification tasks.

## 7. Cityscapes

Status: already covered by `benchmarks/cityscapes.py`.

### 7.1 Semantic Segmentation (Other)

- Measure semantic segmentation using point queries.
- Keep point accuracy as the main score.
- Explain semantic segmentation as "assigning a class label to every pixel in the image."
- Keep pointwise evaluation first because full-mask generation is harder to prompt and parse reliably.

### 7.2 Instance Segmentation (Other)

- Add an instance segmentation benchmark later for object-level masks, especially for cars, people, and bicycles.
- Use mask IoU or box IoU if mask output is not practical yet.
- Explain instance segmentation as "separating individual objects of the same class, not just labeling the pixels by class."
- This is a later step because it is harder to prompt and parse reliably.

## 8. Visual Genome

Status: planned.

### 8.1 Relationship Prediction (Other)

- Start with relationship prediction, for example `person riding horse`.
- Use triplet accuracy for subject-relation-object predictions.
- Keep a small fixed number of queried relationships per image so the prompt stays readable.

### 8.2 Object Grounding (Detection)

- Add object grounding with boxes, scored by IoU.
- Ask the model to localize one named object or phrase at a time.
- This keeps the task concrete and easier to inspect manually.

### 8.3 Visual Question Answering (Visual Q&A)

- Visual Genome also supports question answering, so add a QA benchmark only after 8.1 and 8.2 are stable.
- Use exact-match accuracy.
- This dataset is broad, so splitting it into separate subtasks is better than trying to score everything at once.

## 9. VQA v2.0

Status: planned.

Plan:
- Measure visual question answering: answer a question about an image.
- Use exact-match accuracy first.
- If we want closer alignment with the official benchmark later, add the standard VQA soft score that gives partial credit when several annotators agree with the answer.
- Keep answers short and normalized, because punctuation and capitalization should not decide the score.
- This is a strong candidate for an early addition because it fits the current prompt-response pattern well.

## 10. Fashion-MNIST

Status: planned.

Plan:
- Measure clothing-category classification.
- Use top-1 accuracy.
- Reuse the generic classification benchmark.
- Keep it as a lightweight smoke test because the images are small and the label set is simple.
- This benchmark is useful less for realism and more for quick regression testing.

## 11. Kinetics Human Action Video Dataset

Status: planned.

Plan:
- Measure action classification from sampled video frames.
- Use top-1 accuracy.
- Mirror the UCF101 benchmark design first so both video datasets use the same interface.
- Report per-class accuracy because Kinetics has many actions that differ in subtle ways.
- Later upgrade path: use longer clips or native video support if available.

## 12. iNaturalist

Status: already covered by `benchmarks/inaturalist.py`.

Plan:
- Measure fine-grained species classification.
- Use top-1 accuracy.
- Keep the benchmark label-driven, but note in documentation that this task is harder than ImageNet because many classes look extremely similar.
- Report both overall accuracy and genus-level or higher-level fallback accuracy later if the dataset exposes that taxonomy cleanly.
- This is a good benchmark for whether the model can handle subtle visual differences.

## 13. Places

Status: planned.

Plan:
- Measure scene recognition, such as kitchen, coast, or corridor.
- Use top-1 accuracy.
- Reuse the generic classification benchmark.
- Keep the candidate label list reasonably sized per run to avoid turning the prompt into a giant label dump.
- This complements ImageNet because it tests scene understanding instead of object understanding.

## 14. Conceptual Captions

Status: planned.

### 14.1 Caption Selection (Captioning)

- Add a simpler retrieval-style version where the model chooses the correct caption from several options.
- Use multiple-choice accuracy.
- This is cheaper and more robust to score than free caption generation.

### 14.2 Caption Generation (Captioning)

- Measure image caption generation.
- Use BLEU first, because the repo already uses a BLEU-style caption score for Flickr30k.
- Keep captions concise and compare against the provided references.
- This is more expensive because it requires open-ended generation and noisier text scoring.

## 15. Open Images V4

Status: planned.

### 15.1 Image-Level Classification (Classification)

- Open Images also supports image-level labels, so add a separate classification benchmark.
- Use top-1 or multi-label F1 depending on the chosen annotation slice.
- This is useful when box annotations are unavailable or too slow to benchmark at scale.
- This is the cheapest Open Images entry because it avoids localization.

### 15.2 Object Detection (Detection)

- Measure object detection.
- Use AP or, if we want a simpler first pass, IoU-based precision, recall, and F1 like the current COCO benchmark.
- Explain AP in plain English as "how well the detector balances finding real objects without inventing too many fake ones."
- This is more expensive than 15.1 because it requires localization and box matching.

### 15.3 Relationship Detection (Other)

- Add relationship detection only after boxes are stable.
- Use triplet accuracy plus box IoU for the related objects.
- This keeps the benchmark aligned with the richer annotations without making it the first implementation target.

## 16. GQA

Status: planned.

Plan:
- Measure grounded visual question answering.
- Use exact-match accuracy on answers.
- Keep question answering as the first task instead of full scene-graph prediction.
- Report answer length and normalization rules because tiny wording differences can otherwise look like failures.
- Later upgrade path: add explanation or evidence fields, but do not score them initially.

## 17. MVTec AD

Status: planned.

### 17.1 Anomaly Classification (Classification)

- Measure anomaly detection, meaning whether the model can tell if an industrial item is normal or defective.
- Use image-level AUROC for "normal vs anomalous" first.
- Explain anomaly detection directly in the benchmark file because it is not a common term outside inspection tasks.

### 17.2 Anomaly Localization (Other)

- If the model can localize the defect, add mask IoU or pixel F1 against the defect mask.
- This should be reported separately from classification because a model might notice a defect without locating it precisely.
- This benchmark should support both classification-only mode and localization mode.

## 18. LVIS

Status: planned.

Plan:
- Measure long-tail object detection.
- Use AP as the main score.
- Report rare, common, and frequent category performance separately because LVIS is designed around the long-tail problem.
- Explain the long-tail issue as "some classes appear often, while many classes are rare."
- Build this only after Open Images or a stronger COCO-style detector benchmark exists.

## 19. DocVQA

Status: planned.

### 19.1 Answer Exactness (Visual Q&A)

- Measure document visual question answering.
- Use exact match.
- Normalize whitespace, casing, and punctuation aggressively before scoring.
- This is the cheaper first score because it is a single normalization and string comparison.

### 19.2 OCR-Tolerant Answering (Visual Q&A)

- Add ANLS as a second score.
- Explain ANLS as a string-similarity score that gives partial credit when the answer is almost right, which is helpful for OCR mistakes.
- This benchmark will test both reading text in images and understanding document layout.
- This is slightly more expensive than 19.1 because it adds similarity computation.

## 20. DFDC

Status: planned.

Plan:
- Measure deepfake detection: decide whether a face video is real or fake.
- Use AUROC first and plain accuracy second.
- Explain deepfakes plainly in the benchmark docs so the task is understandable to non-specialists.
- Start with clip-level classification from sampled frames, because full video processing may not fit the current interface.
- Later upgrade path: add temporal consistency features if the model can process video natively.

## 21. TextCaps

Status: planned.

### 21.1 Caption Quality (Captioning)

- Measure captioning for images that contain important visible text.
- Use BLEU for overall caption similarity.
- Compare predicted captions against references.
- This is the cheaper first score because it reuses standard caption-evaluation machinery.

### 21.2 Text Preservation (Captioning)

- Add a text-token recall measure.
- Explain OCR in the benchmark docs because this dataset depends on reading words inside the image, not just describing objects.
- Check whether important text strings visible in the image were preserved in the caption.
- This is a good benchmark to add after basic captioning is stable.
- This is more expensive because it needs token-level text extraction and matching logic on top of caption scoring.

## 22. LAION-400M

Status: planned.

### 22.1 Image-to-Caption Matching (Captioning)

- Use a retrieval-style proxy benchmark rather than a full generative benchmark.
- Show an image and ask the model to pick the correct caption from several candidates.
- Use multiple-choice accuracy or Recall@k.
- Keep the candidate pool small in this first phase.

### 22.2 Caption-to-Image Matching (Captioning)

- Add the reverse direction as a separate test.
- Show a caption and ask the model to choose the matching image from several candidates.
- Use a larger candidate pool or harder negatives than 22.1 so this phase is intentionally more expensive.
- This gives a better picture of multimodal alignment than only testing one direction.
- Keep the sampled subset clean because LAION metadata is noisy.

## 23. FairFace

Status: planned, but should be handled carefully.

Plan:
- Measure attribute classification only if this evaluation is truly wanted.
- Use per-attribute accuracy for age group, gender presentation, and race category as labeled by the dataset.
- Report results separately by attribute instead of collapsing everything into one score.
- Include a note that this is a sensitive benchmark and that labels reflect dataset conventions rather than objective truths.
- Keep this benchmark opt-in rather than enabled by default.

## 24. HDTF

Status: planned, but lower priority for the current repo shape.

### 24.1 Speech or Transcript Matching (Other)

- Measure audio-visual talking-head understanding rather than full video generation at first.
- A practical proxy task is transcript or spoken-phrase matching from the video.
- Use retrieval accuracy or multiple-choice accuracy.

### 24.2 Lip-Sync Judgment (Other)

- Add a benchmark for lip-sync quality if the model can jointly process video and audio.
- Explain lip-sync as "whether the mouth motion matches the spoken audio."
- This likely needs audio input support, so it is not an early benchmark candidate.

## 25. LAION-5B

Status: planned.

### 25.1 Image-to-Caption Matching (Captioning)

- Mirror the LAION-400M benchmark design with a larger or newer metadata source.
- Use image-to-caption matching accuracy or Recall@k.
- Keep the candidate pool small in this first phase.

### 25.2 Caption-to-Image Matching (Captioning)

- Add the reverse-direction retrieval task here as well.
- Use a larger candidate pool or harder negatives than 25.1 so this phase is intentionally more expensive.
- Keep the benchmark on a curated sample, not the raw full dataset.
- Report failure cases caused by noisy web captions because that is part of what this dataset measures.
- If both LAION benchmarks exist, keep the prompt format identical so they are comparable.

## 26. DiffusionDB

Status: planned.

Plan:
- Treat this as a prompt-recovery benchmark.
- Show the generated image and ask the model to choose the original prompt from several candidates.
- Use multiple-choice accuracy.
- This is easier and more reproducible than scoring image generation directly.
- Keep distractor prompts semantically similar so the benchmark measures real prompt understanding instead of keyword matching.

## 27. TAD66K

Status: planned.

### 27.1 Score Prediction (Other)

- Measure aesthetic judgment within a theme.
- Use score prediction correlation.
- Explain correlation in simple language as "do higher model scores line up with higher human scores."
- This is the cheaper first phase because each image only needs one predicted score.

### 27.2 Pairwise Preference (Other)

- Add a pairwise preference version where the model chooses which of two images should score higher.
- Use pairwise accuracy.
- Theme-specific reporting matters here because beauty standards differ across themes like plants, people, or landscapes.
- This benchmark is best added after a clean pairwise preference framework exists.
- This is more expensive because each example contains at least two images and harder comparison logic.

## 28. MagicBrush

Status: planned.

### 28.1 Instruction Recovery (Other)

- Use an edit-understanding proxy benchmark.
- Show the source image and edited target image, then ask the model to recover the editing instruction from a small candidate list.
- Use multiple-choice accuracy.
- This avoids needing an image-editing model just to benchmark the dataset.

### 28.2 Editing Model Evaluation (Other)

- For actual editing models, score instruction-following and image coherence separately.
- Keep those as separate scores because an edit can follow the instruction but still produce a broken-looking image, or vice versa.
- This should come later than instruction recovery.

## 29. Pick-a-Pic

Status: planned.

Plan:
- Measure pairwise preference prediction.
- Show the prompt and two candidate images, then ask which image humans preferred.
- Use accuracy, with ties handled explicitly.
- Explain preference benchmarking as "predicting which output people liked better."
- This is one of the cleanest ways to benchmark generative-image judgment without generating new images.

## 30. InternVid

Status: planned.

### 30.1 Video-to-Caption Matching (Captioning)

- Measure video-caption alignment.
- Show frames from a clip and ask the model to choose the best caption from several options.
- Use multiple-choice accuracy or Recall@k.
- Keep the candidate pool small in this first phase.

### 30.2 Caption-to-Video Matching (Captioning)

- Add the reverse direction as a separate benchmark.
- Use a larger candidate pool or stronger hard negatives than 30.1 so this phase is intentionally more expensive.
- Keep captions descriptive and distractors close in meaning so the task stays challenging.
- This benchmark can share most of its machinery with UCF101 and Kinetics.

## 31. Open Vid-1M

Status: planned.

### 31.1 Video-to-Prompt Matching (Other)

- Measure prompt-video alignment using a retrieval-style benchmark.
- Show a clip and ask the model to choose the matching prompt.
- Use multiple-choice accuracy or Recall@k.
- Keep the candidate pool small in this first phase.

### 31.2 Video-to-Caption Matching (Captioning)

- Add a caption-alignment version as a second task where available.
- Use either larger candidate pools or denser distractor selection than 31.1 so this phase is intentionally more expensive.
- Because the dataset is prompt-rich, this is likely a better fit than pure action recognition.
- Keep the benchmark subset high quality and short enough to inspect manually.

## 32. Visual CoT

Status: planned.

### 32.1 Final Answer Accuracy (Visual Q&A)

- Measure multimodal reasoning with grounded evidence.
- Use answer accuracy as the primary score.
- This is the cheaper first phase because it only checks the final answer.

### 32.2 Grounding Quality (Detection)

- If bounding boxes are part of the target, add IoU as a secondary grounding score.
- Explain CoT, short for chain of thought, simply as "step-by-step reasoning," but do not require the benchmark to score the reasoning text itself at first.
- This benchmark should stay strict about final answers and only treat explanations as optional metadata.
- This is more expensive because it requires structured grounding output and localization scoring.

## 33. HQ-Edit

Status: planned.

### 33.1 Instruction Recovery (Other)

- Use an edit-instruction recovery benchmark first.
- Show before-and-after images and ask the model to choose or write the instruction that best explains the edit.
- Use exact match or multiple-choice accuracy.

### 33.2 Editing Model Evaluation (Other)

- If we later benchmark actual image editors, split scoring into alignment with the instruction and visual coherence of the result.
- Explain coherence as "does the edited image still look like a plausible image rather than a broken one."

## 34. BLIP3o-60k

Status: planned.

### 34.1 Instruction Recovery (Other)

- Treat this as an instruction-following image-edit dataset.
- Start with instruction recovery from image pairs.
- Use multiple-choice accuracy.
- Keep a small, carefully designed distractor set so the benchmark checks true edit understanding.

### 34.2 Editing Model Evaluation (Other)

- Later upgrade path: benchmark actual editing models with a separate generation pipeline.
- Keep this separate from instruction recovery so we do not mix "understands the edit" with "can perform the edit."

## 35. ImgEdit

Status: planned.

### 35.1 Instruction Recovery (Other)

- Use the same first benchmark shape as MagicBrush and HQ-Edit.
- Show source and target images and ask the model to recover the edit instruction.
- Use multiple-choice accuracy.

### 35.2 Editing Model Evaluation (Other)

- Because this dataset includes many edit types, report accuracy by edit family such as add, remove, background, or style change.
- This will make failures easier to understand than a single blended score.
- Keep this as a separate track from instruction recovery.

## 36. ShareGPT-4o-Image

Status: planned.

### 36.1 Text-to-Image Prompt Understanding (Other)

- Show an output image and ask the model to choose the best prompt.
- Use multiple-choice accuracy.
- This is the cheaper first phase because each example uses one image and one prompt-selection task.

### 36.2 Image-Edit Instruction Understanding (Other)

- Show source and target images and ask for the best instruction.
- Use multiple-choice accuracy.
- This should be one of the last benchmarks added, because it is easiest to build after the prompt-recovery and edit-recovery benchmark patterns are already in place.
- This is more expensive because each example uses an image pair and requires reasoning about the change between them.

Suggested build order:

- First wave: LSUN, VQA v2.0, Fashion-MNIST, Places, Conceptual Captions.
- Second wave: Open Images V4, GQA, TextCaps, Kinetics, DocVQA.
- Third wave: MVTec AD, LVIS, DiffusionDB, Pick-a-Pic, InternVid, Open Vid-1M.
- Fourth wave: Visual Genome, LAION-400M, LAION-5B, TAD66K, MagicBrush, HQ-Edit, BLIP3o-60k, ImgEdit, ShareGPT-4o-Image.
- Special-case or opt-in wave: FairFace, HDTF, DFDC.
