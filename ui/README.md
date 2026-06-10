# Benchmark input browser

From the repository root, run:

```powershell
python -m ui.input_browser
```

The script can also be launched by path from any working directory:

```powershell
python C:\Users\Samuel\Documents\GitHub\transformers\ui\input_browser.py
```

The launcher adds the repository root to Python's import path before loading
benchmark modules.

The browser groups the concrete benchmarks under the eight benchmark types.
Selecting a benchmark loads its first 20 rows from its configured Hugging Face
source. The detail screen displays the exact image and prompt produced by the
same preparation methods used immediately before:

```python
model.predict(image, prompt)
```

For the curated type-A datasets, the runtime also reads answer choices from
`benchmark_choices/type_a/`. Loading errors for unavailable, gated, or
credentialed datasets are shown in the detail screen.

The dark loading screen reports each construction stage: importing the
benchmark, initializing its HF adapter, preparing rows, selecting the sample,
constructing the image, selecting prompt labels or choices, rendering the
prompt, and finalizing the RGB preview. Revisiting a benchmark reports when
its prepared rows are reused from memory.

MS COCO detection requires the official instance annotations because its HF
rows do not include detection boxes. On the first run, step 3 may download the
roughly 253 MB COCO annotations archive. It is cached under
`.tmp/coco_annotations/` in the repository, regardless of the directory from
which the UI was launched.

## Tests

The regular UI test is intentionally fast and offline: it verifies that all
benchmark tiles dispatch correctly and that a returned image and prompt are
rendered.

The live integration test loads the real first input for every benchmark and
can take a long time or download substantial data:

```powershell
$env:RUN_INPUT_BROWSER_LIVE_TESTS="1"
python -m unittest tests.tests_datasets.test_input_browser_live -v
```

It uses Hugging Face's normal authentication, including `HF_TOKEN` and saved
login credentials. To diagnose one benchmark, set a case-insensitive name
filter:

```powershell
$env:INPUT_BROWSER_LIVE_BENCHMARK="MS COCO"
python -m unittest tests.tests_datasets.test_input_browser_live -v
```
