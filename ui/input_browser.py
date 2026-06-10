from __future__ import annotations

import importlib
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageTk


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass(frozen=True)
class BenchmarkType:
    code: str
    title: str
    description: str


@dataclass(frozen=True)
class BenchmarkSpec:
    type_code: str
    name: str
    module: str
    class_name: str


@dataclass
class PreparedBenchmark:
    benchmark: Any
    rows: list[dict[str, Any]]
    labels: list[str]


@dataclass
class ModelInputPreview:
    benchmark_name: str
    dataset_name: str
    row_index: int
    row_count: int
    image: Image.Image
    prompt: str
    prompt_labels: list[str]
    source: str
    split: str
    correct_answers: list[str]
    show_correct_answer: bool = True
    displayed_box_count: int = 0


ProgressCallback = Callable[[int, int, str], None]


BENCHMARK_TYPES = (
    BenchmarkType("A", "Answering Questions", "Select an answer to a visual question."),
    BenchmarkType("B", "Bounding Box Detection", "Return labels and normalized object boxes."),
    BenchmarkType("C", "Captioning", "Generate a concise image or video caption."),
    BenchmarkType("E", "Editing Reconstruction", "Identify the instruction used to edit an image."),
    BenchmarkType("G", "Generating Reconstruction", "Identify the prompt used to generate an image."),
    BenchmarkType("L", "Labeling", "Choose the correct class label."),
    BenchmarkType("P", "Preference", "Choose the preferred image in a pair."),
    BenchmarkType("R", "Rating", "Assign an aesthetic score from 1 to 10."),
)


BENCHMARK_SPECS = (
    BenchmarkSpec("A", "FlyingThings3D", "benchmarks.multiple_choice.flyingthings3d", "FlyingThings3DBenchmark"),
    BenchmarkSpec("A", "Visual Genome", "benchmarks.visual_qa.visual_genome", "VisualGenomeBenchmark"),
    BenchmarkSpec("A", "VQA v2.0", "benchmarks.visual_qa.vqa_v2", "VQAv2Benchmark"),
    BenchmarkSpec("A", "GQA", "benchmarks.visual_qa.gqa", "GQABenchmark"),
    BenchmarkSpec("A", "DocVQA", "benchmarks.visual_qa.docvqa", "DocVQABenchmark"),
    BenchmarkSpec("A", "Visual CoT", "benchmarks.visual_qa.visual_cot", "VisualCoTBenchmark"),
    BenchmarkSpec("B", "Flickr30k Entities", "benchmarks.detection.flickr30k_entities", "Flickr30kEntitiesBenchmark"),
    BenchmarkSpec("B", "MS COCO", "benchmarks.detection.mscoco", "MSCOCOBenchmark"),
    BenchmarkSpec("B", "LVIS", "benchmarks.detection.lvis", "LVISBenchmark"),
    BenchmarkSpec("B", "OpenImages V4", "benchmarks.detection.openimages_v4_detection", "OpenImagesV4DetectionBenchmark"),
    BenchmarkSpec("B", "iNaturalist", "benchmarks.detection.inaturalist", "INaturalistDetectionBenchmark"),
    BenchmarkSpec("B", "Visual CoT", "benchmarks.detection.visual_cot", "VisualCoTDetectionBenchmark"),
    BenchmarkSpec("C", "Flickr30k", "benchmarks.captioning.flickr30k", "Flickr30kBenchmark"),
    BenchmarkSpec("C", "MS COCO Captions", "benchmarks.captioning.mscoco_caption", "MSCOCOCaptionBenchmark"),
    BenchmarkSpec("C", "TextCaps", "benchmarks.captioning.textcaps", "TextCapsBenchmark"),
    BenchmarkSpec("C", "Conceptual Captions", "benchmarks.captioning.conceptual_captions_caption", "ConceptualCaptionsCaptionBenchmark"),
    BenchmarkSpec("C", "HDTF", "benchmarks.captioning.hdtf", "HDTFBenchmark"),
    BenchmarkSpec("C", "InternVid", "benchmarks.captioning.internvid", "InternVidBenchmark"),
    BenchmarkSpec("C", "LAION-400M", "benchmarks.captioning.laion400m", "LAION400MBenchmark"),
    BenchmarkSpec("C", "LAION-5B", "benchmarks.captioning.laion5b", "LAION5BBenchmark"),
    BenchmarkSpec("C", "OpenVid-1M", "benchmarks.captioning.openvid1m", "OpenVid1MCaptionBenchmark"),
    BenchmarkSpec("E", "HQ-Edit", "benchmarks.multiple_choice.hq_edit", "HQEditBenchmark"),
    BenchmarkSpec("E", "ImgEdit", "benchmarks.multiple_choice.imgedit", "ImgEditBenchmark"),
    BenchmarkSpec("E", "MagicBrush", "benchmarks.multiple_choice.magicbrush", "MagicBrushBenchmark"),
    BenchmarkSpec("E", "ShareGPT-4o-Image", "benchmarks.multiple_choice.sharegpt4o_image_edit", "ShareGPT4oImageEditBenchmark"),
    BenchmarkSpec("G", "BLIP3o-60k", "benchmarks.multiple_choice.blip3o_60k", "BLIP3o60kBenchmark"),
    BenchmarkSpec("G", "Conceptual Captions", "benchmarks.multiple_choice.conceptual_captions", "ConceptualCaptionsBenchmark"),
    BenchmarkSpec("G", "DiffusionDB", "benchmarks.multiple_choice.diffusiondb", "DiffusionDBBenchmark"),
    BenchmarkSpec("G", "OpenVid-1M", "benchmarks.multiple_choice.openvid1m", "OpenVid1MBenchmark"),
    BenchmarkSpec("G", "ShareGPT4o-Image", "benchmarks.multiple_choice.sharegpt4o_image", "ShareGPT4oImageBenchmark"),
    BenchmarkSpec("L", "Cityscapes", "benchmarks.classification.cityscapes", "CityscapesBenchmark"),
    BenchmarkSpec("L", "FairFace", "benchmarks.classification.fairface", "FairFaceBenchmark"),
    BenchmarkSpec("L", "Fashion-MNIST", "benchmarks.classification.fashion_mnist", "FashionMNISTBenchmark"),
    BenchmarkSpec("L", "ImageNet-1K", "benchmarks.classification.imagenet1k", "ImageNet1kBenchmark"),
    BenchmarkSpec("L", "iNaturalist", "benchmarks.classification.inaturalist", "INaturalistBenchmark"),
    BenchmarkSpec("L", "LSUN", "benchmarks.classification.lsun", "LSUNBenchmark"),
    BenchmarkSpec("L", "MVTec AD", "benchmarks.classification.mvtec_ad", "MVTecADBenchmark"),
    BenchmarkSpec("L", "OpenImages V4", "benchmarks.classification.openimages_v4", "OpenImagesV4Benchmark"),
    BenchmarkSpec("L", "Places", "benchmarks.classification.places", "PlacesBenchmark"),
    BenchmarkSpec("L", "DFDC", "benchmarks.video_classification.dfdc", "DFDCBenchmark"),
    BenchmarkSpec("L", "Kinetics", "benchmarks.video_classification.kinetics", "KineticsBenchmark"),
    BenchmarkSpec("L", "UCF101", "benchmarks.video_classification.ucf101", "UCF101Benchmark"),
    BenchmarkSpec("P", "Pick-a-Pic", "benchmarks.multiple_choice.pick_a_pic", "PickAPicBenchmark"),
    BenchmarkSpec("R", "TAD66K", "benchmarks.classification.tad66k", "TAD66KBenchmark"),
)


class BenchmarkInputService:
    def __init__(self, sample_count: int = 20, label_sample_size: int = 64, streaming: bool = True):
        self.sample_count = sample_count
        self.label_sample_size = label_sample_size
        self.streaming = streaming
        self._cache: dict[BenchmarkSpec, PreparedBenchmark] = {}

    def specs_for_type(self, type_code: str) -> list[BenchmarkSpec]:
        return [spec for spec in BENCHMARK_SPECS if spec.type_code == type_code]

    def prepare(
        self,
        spec: BenchmarkSpec,
        progress: ProgressCallback | None = None,
    ) -> PreparedBenchmark:
        cached = self._cache.get(spec)
        if cached is not None:
            self._emit(progress, 1, "Benchmark module already loaded.")
            self._emit(progress, 2, "Benchmark and dataset adapter already initialized.")
            self._emit(progress, 3, f"Using {len(cached.rows)} prepared rows from the in-memory cache.")
            return cached

        self._emit(progress, 1, f"Importing {spec.module}.")
        module = importlib.import_module(spec.module)
        benchmark_cls = getattr(module, spec.class_name)

        self._emit(progress, 2, "Initializing the benchmark and its Hugging Face dataset adapter.")
        benchmark = benchmark_cls(streaming=self.streaming)

        preparation_message = getattr(
            benchmark,
            "preview_preparation_message",
            (
                f"Fetching and preparing up to {self.sample_count} rows. "
                "This may download metadata, images, or video frames."
            ),
        )
        self._emit(
            progress,
            3,
            str(preparation_message),
        )
        set_progress_callback = getattr(
            benchmark.dataset,
            "set_preview_progress_callback",
            None,
        )
        if callable(set_progress_callback):
            set_progress_callback(lambda message: self._emit(progress, 3, message))
        try:
            rows, labels = benchmark.prepare(
                n=self.sample_count,
                label_sample_size=max(self.label_sample_size, self.sample_count),
            )
        finally:
            if callable(set_progress_callback):
                set_progress_callback(None)
        if not rows:
            raise ValueError(f"{spec.name} returned no benchmark rows.")
        prepared = PreparedBenchmark(benchmark=benchmark, rows=rows, labels=labels)
        self._cache[spec] = prepared
        return prepared

    def preview(
        self,
        spec: BenchmarkSpec,
        row_index: int,
        progress: ProgressCallback | None = None,
    ) -> ModelInputPreview:
        prepared = self.prepare(spec, progress=progress)
        if row_index < 0 or row_index >= len(prepared.rows):
            raise IndexError(f"Row {row_index} is outside 0..{len(prepared.rows) - 1}.")

        self._emit(progress, 4, f"Selecting prepared row {row_index + 1} of {len(prepared.rows)}.")
        benchmark = prepared.benchmark
        row = prepared.rows[row_index]

        self._emit(progress, 5, "Constructing the exact image input used by the benchmark.")
        try:
            image = benchmark.get_image_for_row(row)
        except Exception:
            replacement_getter = getattr(benchmark.dataset, "get_next_available_sample", None)
            if not callable(replacement_getter):
                raise
            self._emit(progress, 5, "The source image is unavailable; selecting the next valid row.")
            row = replacement_getter()
            prepared.rows[row_index] = row
            image = benchmark.get_image_for_row(row)
        if not isinstance(image, Image.Image):
            image = benchmark._coerce_image(image)

        self._emit(
            progress,
            6,
            "Selecting the row-specific labels or answer choices included in the prompt.",
        )
        prompt_labels = benchmark.get_prompt_labels_for_row(row=row, labels=prepared.labels)

        self._emit(progress, 7, "Rendering the exact text prompt passed to model.predict.")
        prompt = benchmark.make_prompt(labels=prompt_labels, row=row, image=image)
        correct_answers = benchmark.get_valid_labels_for_row(row)

        self._emit(progress, 8, "Adding reference annotations to the display copy and finalizing the preview.")
        display_image = image.convert("RGB")
        displayed_box_count = 0
        if spec.type_code == "B":
            ground_truth_boxes = benchmark.get_ground_truth_boxes_for_row(row)
            ground_truth_boxes = benchmark.postprocess_ground_truth_boxes(
                ground_truth_boxes,
                image=image,
            )
            display_image = self._draw_detection_boxes(display_image, ground_truth_boxes)
            displayed_box_count = len(ground_truth_boxes)

        dataset = benchmark.dataset
        return ModelInputPreview(
            benchmark_name=str(benchmark.name),
            dataset_name=str(getattr(dataset, "name", "")),
            row_index=row_index,
            row_count=len(prepared.rows),
            image=display_image,
            prompt=prompt,
            prompt_labels=list(prompt_labels),
            source=str(getattr(dataset, "dataset_id", "local or custom source")),
            split=str(getattr(dataset, "split", "")),
            correct_answers=[str(answer) for answer in correct_answers],
            show_correct_answer=spec.type_code != "B",
            displayed_box_count=displayed_box_count,
        )

    @staticmethod
    def _draw_detection_boxes(
        image: Image.Image,
        boxes: list[dict[str, Any]],
    ) -> Image.Image:
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        image_width, image_height = annotated.size
        line_width = max(2, round(max(image_width, image_height) / 250))

        for box in boxes:
            xyxy = box.get("xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in xyxy]
            x0 = min(max(0.0, x0), float(image_width - 1))
            y0 = min(max(0.0, y0), float(image_height - 1))
            x1 = min(max(0.0, x1), float(image_width - 1))
            y1 = min(max(0.0, y1), float(image_height - 1))
            draw.rectangle([x0, y0, x1, y1], outline="#ff3b30", width=line_width)

            label = str(box.get("label", "")).strip()
            if label:
                text_bbox = draw.textbbox((x0, y0), label)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                label_y = max(0.0, y0 - text_height - 4)
                draw.rectangle(
                    [x0, label_y, x0 + text_width + 6, label_y + text_height + 4],
                    fill="#ff3b30",
                )
                draw.text((x0 + 3, label_y + 2), label, fill="white")

        return annotated

    @staticmethod
    def _emit(progress: ProgressCallback | None, step: int, message: str) -> None:
        if progress is not None:
            progress(step, 8, message)


class BenchmarkInputBrowser:
    BG = "#0d1117"
    CARD = "#161b22"
    CARD_HOVER = "#212a35"
    INPUT_BG = "#090c10"
    BORDER = "#30363d"
    ACCENT = "#58a6ff"
    ACCENT_ACTIVE = "#79c0ff"
    TEXT = "#f0f6fc"
    MUTED = "#8b949e"
    SUCCESS = "#3fb950"
    ERROR = "#f85149"

    def __init__(self, service: BenchmarkInputService | None = None):
        self.service = service or BenchmarkInputService()
        self.root = tk.Tk()
        self.root.title("Benchmark Model Input Browser")
        self.root.geometry("1280x840")
        self.root.minsize(960, 680)
        self.root.configure(bg=self.BG)
        self._configure_styles()
        self.current_spec: BenchmarkSpec | None = None
        self.current_index = 0
        self._photo: ImageTk.PhotoImage | None = None
        self._preview_image: Image.Image | None = None
        self._request_id = 0
        self._show_type_grid()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=self.INPUT_BG,
            background=self.ACCENT,
            bordercolor=self.BORDER,
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
        )
        style.configure(
            "Dark.Vertical.TScrollbar",
            troughcolor=self.BG,
            background=self.BORDER,
            arrowcolor=self.TEXT,
            bordercolor=self.BG,
        )

    def run(self) -> None:
        self.root.mainloop()

    def _clear(self) -> None:
        self._request_id += 1
        for child in self.root.winfo_children():
            child.destroy()

    def _header(self, title: str, back_command=None) -> tk.Frame:
        frame = tk.Frame(self.root, bg=self.BG)
        frame.pack(fill=tk.X, padx=24, pady=(20, 12))
        if back_command is not None:
            self._button(frame, "< Back", back_command, padx=14, pady=6).pack(side=tk.LEFT)
        tk.Label(
            frame,
            text=title,
            bg=self.BG,
            fg=self.TEXT,
            font=("Segoe UI", 20, "bold"),
        ).pack(side=tk.LEFT, padx=(16 if back_command else 0, 0))
        return frame

    def _show_type_grid(self) -> None:
        self._clear()
        self._header("The 8 Benchmarking Types")
        grid = tk.Frame(self.root, bg=self.BG)
        grid.pack(fill=tk.BOTH, expand=True, padx=24, pady=12)
        for column in range(4):
            grid.grid_columnconfigure(column, weight=1, uniform="type")
        for row in range(2):
            grid.grid_rowconfigure(row, weight=1, uniform="type")

        for index, benchmark_type in enumerate(BENCHMARK_TYPES):
            button = tk.Button(
                grid,
                text=f"{benchmark_type.code}\n{benchmark_type.title}\n\n{benchmark_type.description}",
                command=lambda item=benchmark_type: self._show_benchmark_grid(item),
                bg=self.CARD,
                fg=self.TEXT,
                activebackground=self.CARD_HOVER,
                activeforeground=self.TEXT,
                highlightbackground=self.BORDER,
                highlightcolor=self.ACCENT,
                highlightthickness=1,
                relief=tk.FLAT,
                bd=0,
                wraplength=230,
                font=("Segoe UI", 13, "bold"),
                cursor="hand2",
            )
            button.grid(row=index // 4, column=index % 4, sticky="nsew", padx=9, pady=9)

    def _show_benchmark_grid(self, benchmark_type: BenchmarkType) -> None:
        self._clear()
        self._header(f"{benchmark_type.code}: {benchmark_type.title}", self._show_type_grid)

        canvas = tk.Canvas(self.root, bg=self.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self.root,
            orient=tk.VERTICAL,
            command=canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )
        body = tk.Frame(canvas, bg=self.BG)
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(24, 0), pady=(0, 20))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 16), pady=(0, 20))

        specs = self.service.specs_for_type(benchmark_type.code)
        for column in range(4):
            body.grid_columnconfigure(column, weight=1, uniform="benchmark")
        for index, spec in enumerate(specs):
            button = tk.Button(
                body,
                text=spec.name,
                command=self._benchmark_click_command(spec),
                bg=self.CARD,
                fg=self.TEXT,
                activebackground=self.CARD_HOVER,
                activeforeground=self.TEXT,
                highlightbackground=self.BORDER,
                highlightcolor=self.ACCENT,
                highlightthickness=1,
                relief=tk.FLAT,
                padx=16,
                pady=30,
                font=("Segoe UI", 12, "bold"),
                cursor="hand2",
            )
            button.grid(row=index // 4, column=index % 4, sticky="nsew", padx=8, pady=8)

    def _benchmark_click_command(self, spec: BenchmarkSpec):
        return lambda: self._show_preview(spec, 0)

    def _show_preview(self, spec: BenchmarkSpec, row_index: int) -> None:
        self.current_spec = spec
        self.current_index = row_index
        self._preview_image = None
        self._photo = None
        self._clear()
        header = self._header(spec.name, lambda: self._show_benchmark_grid(self._type_for(spec.type_code)))

        self.prev_button = self._button(
            header,
            "Previous",
            lambda: self._navigate(-1),
            state=tk.DISABLED,
        )
        self.prev_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.next_button = self._button(
            header,
            "Next",
            lambda: self._navigate(1),
            state=tk.DISABLED,
        )
        self.next_button.pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Starting benchmark input construction...")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=self.BG,
            fg=self.ACCENT,
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.status_label.pack(fill=tk.X, padx=24, pady=(0, 8))

        self.loading_panel = tk.Frame(
            self.root,
            bg=self.CARD,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        self.loading_panel.pack(fill=tk.X, padx=24, pady=(0, 12))
        self.progress_bar = ttk.Progressbar(
            self.loading_panel,
            mode="determinate",
            maximum=8,
            value=0,
            style="Dark.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill=tk.X, padx=12, pady=(12, 6))
        self.loading_log = ScrolledText(
            self.loading_panel,
            wrap=tk.WORD,
            height=4,
            font=("Consolas", 9),
            bg=self.INPUT_BG,
            fg=self.MUTED,
            insertbackground=self.TEXT,
            selectbackground=self.ACCENT,
            selectforeground=self.INPUT_BG,
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.loading_log.pack(fill=tk.X, padx=12, pady=(0, 12))
        self._style_text_scrollbar(self.loading_log)
        self.loading_log.insert("1.0", "Waiting to start...\n")
        self.loading_log.configure(state=tk.DISABLED)

        content = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6, bg=self.BG)
        content.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 24))
        left = tk.Frame(content, bg=self.CARD)
        right = tk.Frame(content, bg=self.CARD)
        content.add(left, minsize=420, stretch="always")
        content.add(right, minsize=420, stretch="always")

        image_box = tk.Frame(
            left,
            bg=self.INPUT_BG,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        image_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        self.image_label = tk.Label(
            image_box,
            text="",
            bg=self.INPUT_BG,
            fg=self.TEXT,
            anchor=tk.CENTER,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Configure>", self._resize_preview)

        tk.Label(
            right,
            text="Exact prompt passed to model.predict(image, prompt)",
            bg=self.CARD,
            fg=self.TEXT,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(12, 4))

        self.answer_panel = tk.Frame(right, bg=self.CARD)
        self.answer_panel.pack(side=tk.BOTTOM, fill=tk.X)
        self.answer_panel.configure(height=230)
        self.answer_panel.pack_propagate(False)
        tk.Label(
            self.answer_panel,
            text="Correct answer",
            bg=self.CARD,
            fg=self.SUCCESS,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(0, 4))
        self.answer_text = ScrolledText(
            self.answer_panel,
            wrap=tk.WORD,
            height=6,
            font=("Segoe UI", 10),
            bg=self.INPUT_BG,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            selectbackground=self.ACCENT,
            selectforeground=self.INPUT_BG,
            relief=tk.FLAT,
            borderwidth=0,
            padx=8,
            pady=6,
        )
        self.answer_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        self._style_text_scrollbar(self.answer_text)
        self.answer_text.insert("1.0", "Loading...")
        self.answer_text.configure(state=tk.DISABLED)
        self.meta_var = tk.StringVar(value="")
        tk.Label(
            self.answer_panel,
            textvariable=self.meta_var,
            bg=self.CARD,
            fg=self.TEXT,
            justify=tk.LEFT,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill=tk.X, padx=12, pady=(0, 12))

        self.prompt_text = ScrolledText(
            right,
            wrap=tk.WORD,
            font=("Consolas", 11),
            height=12,
            bg=self.INPUT_BG,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            selectbackground=self.ACCENT,
            selectforeground=self.INPUT_BG,
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        self._style_text_scrollbar(self.prompt_text)
        self.prompt_text.configure(state=tk.DISABLED)

        self._load_async(spec, row_index)

    def _load_async(self, spec: BenchmarkSpec, row_index: int) -> None:
        self._request_id += 1
        request_id = self._request_id

        def worker() -> None:
            def progress(step: int, total: int, message: str) -> None:
                self.root.after(
                    0,
                    lambda: self._display_progress(request_id, step, total, message),
                )

            try:
                preview = self.service.preview(spec, row_index, progress=progress)
            except Exception as exc:
                self.root.after(0, lambda: self._display_error(request_id, exc))
                return
            self.root.after(0, lambda: self._display_preview(request_id, preview))

        threading.Thread(target=worker, daemon=True).start()

    def _display_progress(
        self,
        request_id: int,
        step: int,
        total: int,
        message: str,
    ) -> None:
        if request_id != self._request_id:
            return
        self.progress_bar.configure(maximum=total, value=step)
        self.status_label.configure(fg=self.ACCENT)
        self.status_var.set(f"Step {step} of {total}: {message}")
        self.loading_log.configure(state=tk.NORMAL)
        if step == 1:
            self.loading_log.delete("1.0", tk.END)
        self.loading_log.insert(tk.END, f"[{step}/{total}] {message}\n")
        self.loading_log.see(tk.END)
        self.loading_log.configure(state=tk.DISABLED)

    def _display_preview(self, request_id: int, preview: ModelInputPreview) -> None:
        if request_id != self._request_id:
            return
        self._preview_image = preview.image.copy()
        self._render_preview_image()
        self.loading_panel.pack_forget()

        self.prompt_text.configure(state=tk.NORMAL)
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", preview.prompt)
        self.prompt_text.configure(state=tk.DISABLED)
        if preview.show_correct_answer:
            answer_text = (
                "\n\n".join(preview.correct_answers)
                if preview.correct_answers
                else "No reference answer"
            )
            self.answer_text.configure(state=tk.NORMAL)
            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert("1.0", answer_text)
            self.answer_text.configure(state=tk.DISABLED)
        else:
            self.answer_panel.pack_forget()

        width, height = preview.image.size
        self.status_label.configure(fg=self.SUCCESS)
        self.status_var.set(f"Sample {preview.row_index + 1} of {preview.row_count}")
        self.progress_bar.configure(value=8)
        self.loading_log.configure(state=tk.NORMAL)
        self.loading_log.insert(tk.END, "[done] Exact model input is ready.\n")
        self.loading_log.see(tk.END)
        self.loading_log.configure(state=tk.DISABLED)
        self.meta_var.set(
            "\n".join(
                [
                    f"Benchmark: {preview.benchmark_name}",
                    f"Dataset: {preview.dataset_name}",
                    f"HF source: {preview.source}",
                    f"Split: {preview.split}",
                    f"Image sent to model: {width} x {height} RGB",
                    f"Prompt choices/labels: {len(preview.prompt_labels)}",
                    (
                        f"UI reference overlay: {preview.displayed_box_count} ground-truth box(es)"
                        if preview.displayed_box_count
                        else "UI reference overlay: none"
                    ),
                ]
            )
        )
        self.prev_button.configure(state=tk.NORMAL if preview.row_index > 0 else tk.DISABLED)
        self.next_button.configure(
            state=tk.NORMAL if preview.row_index + 1 < preview.row_count else tk.DISABLED
        )

    def _resize_preview(self, _event=None) -> None:
        if self._preview_image is not None:
            self._render_preview_image()

    def _render_preview_image(self) -> None:
        if self._preview_image is None:
            return
        width = max(1, self.image_label.winfo_width())
        height = max(1, self.image_label.winfo_height())
        if width <= 1 or height <= 1:
            width, height = 650, 650
        image = self._fit_image(self._preview_image, width, height)
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo, text="")

    @staticmethod
    def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
        fitted = image.copy()
        fitted.thumbnail(
            (max(1, int(width)), max(1, int(height))),
            Image.Resampling.LANCZOS,
        )
        return fitted

    def _display_error(self, request_id: int, exc: Exception) -> None:
        if request_id != self._request_id:
            return
        self.status_label.configure(fg=self.ERROR)
        self.status_var.set(f"Could not load this benchmark: {type(exc).__name__}: {exc}")
        self.progress_bar.configure(style="Dark.Horizontal.TProgressbar")
        self.loading_log.configure(state=tk.NORMAL)
        self.loading_log.insert(tk.END, f"[error] {type(exc).__name__}: {exc}\n")
        self.loading_log.see(tk.END)
        self.loading_log.configure(state=tk.DISABLED)
        self.prompt_text.configure(state=tk.NORMAL)
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert(
            "1.0",
            "The benchmark input could not be constructed. This usually means the "
            "Hugging Face dataset is unavailable, gated, or requires credentials.",
        )
        self.prompt_text.configure(state=tk.DISABLED)
        self.answer_text.configure(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", "Unavailable")
        self.answer_text.configure(state=tk.DISABLED)

    def _navigate(self, offset: int) -> None:
        if self.current_spec is None:
            return
        self._show_preview(self.current_spec, self.current_index + offset)

    def _button(self, parent, text: str, command, **kwargs) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.CARD,
            fg=self.TEXT,
            activebackground=self.CARD_HOVER,
            activeforeground=self.TEXT,
            disabledforeground=self.MUTED,
            relief=tk.FLAT,
            highlightbackground=self.BORDER,
            highlightcolor=self.ACCENT,
            highlightthickness=1,
            cursor="hand2",
            **kwargs,
        )

    def _style_text_scrollbar(self, text_widget: ScrolledText) -> None:
        text_widget.vbar.configure(
            bg=self.BORDER,
            activebackground=self.CARD_HOVER,
            troughcolor=self.INPUT_BG,
            highlightbackground=self.INPUT_BG,
            highlightcolor=self.INPUT_BG,
            borderwidth=0,
        )

    @staticmethod
    def _type_for(code: str) -> BenchmarkType:
        return next(item for item in BENCHMARK_TYPES if item.code == code)


def main() -> int:
    BenchmarkInputBrowser().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
