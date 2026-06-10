from __future__ import annotations

import json
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass(frozen=True)
class ResultRun:
    path: Path
    group: str
    model: str
    benchmark: str
    report: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    score_label: str
    score: float | None
    score_is_error: bool
    completed: int
    successes: int
    failures: int
    seconds_per_sample: float | None

    @property
    def display_score(self) -> str:
        if self.score is None:
            return "-"
        if self.score_label in {"Accuracy", "F1", "BLEU", "Success"}:
            return f"{self.score * 100:.1f}%"
        return f"{self.score:.3f}"


class ResultRepository:
    def __init__(self, results_dir: str | Path = DEFAULT_RESULTS_DIR):
        self.results_dir = Path(results_dir)

    def load(self) -> list[ResultRun]:
        if not self.results_dir.exists():
            return []
        runs: list[ResultRun] = []
        for path in sorted(self.results_dir.rglob("*.json")):
            if path.name == "run_summary.json" or "old_results" in path.parts:
                continue
            run = self.load_file(path)
            if run is not None:
                runs.append(run)
        return runs

    def load_file(self, path: str | Path) -> ResultRun | None:
        result_path = Path(path)
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("report"), dict):
            return None

        report = payload["report"]
        raw_samples = report.get("results", [])
        samples = tuple(item for item in raw_samples if isinstance(item, dict))
        stats = report.get("stats", {}) if isinstance(report.get("stats"), dict) else {}
        score_label, score, score_is_error = infer_score(samples, stats)
        completed = _as_int(stats.get("number_of_benchmark_samples_completed"), len(samples))
        successes = _as_int(stats.get("success_count"), _sample_success_count(samples))
        failures = _as_int(stats.get("failure_count"), max(0, completed - successes))
        seconds_per_sample = _as_float(stats.get("wall_clock_time_per_sample_seconds_mean"))
        try:
            group = result_path.parent.relative_to(self.results_dir).as_posix()
        except ValueError:
            group = result_path.parent.name

        return ResultRun(
            path=result_path,
            group=group or ".",
            model=str(payload.get("model") or "unknown"),
            benchmark=str(payload.get("benchmark") or report.get("benchmark") or "unknown"),
            report=report,
            samples=samples,
            score_label=score_label,
            score=score,
            score_is_error=score_is_error,
            completed=completed,
            successes=successes,
            failures=failures,
            seconds_per_sample=seconds_per_sample,
        )


def infer_score(
    samples: Iterable[dict[str, Any]],
    stats: dict[str, Any],
) -> tuple[str, float | None, bool]:
    sample_list = list(samples)
    for key, label, is_error in (
        ("f1", "F1", False),
        ("bleu", "BLEU", False),
        ("absolute_error", "MAE", True),
    ):
        values = [_as_float(sample.get(key)) for sample in sample_list]
        present = [value for value in values if value is not None]
        if present:
            return label, sum(present) / len(present), is_error

    correct = [sample.get("correct") for sample in sample_list]
    scored = [value for value in correct if isinstance(value, bool)]
    if scored:
        return "Accuracy", sum(scored) / len(scored), False

    mae = _as_float(stats.get("mean_absolute_error"))
    if mae is not None:
        return "MAE", mae, True
    completed = _as_int(stats.get("number_of_benchmark_samples_completed"), 0)
    successes = _as_int(stats.get("success_count"), 0)
    if completed:
        return "Success", successes / completed, False
    return "Score", None, False


def _sample_success_count(samples: Iterable[dict[str, Any]]) -> int:
    count = 0
    for sample in samples:
        stats = sample.get("stats")
        if not isinstance(stats, dict) or stats.get("success", True):
            count += 1
    return count


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class BenchmarkResultsBrowser:
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
    WARNING = "#d29922"
    ERROR = "#f85149"
    PURPLE = "#bc8cff"

    def __init__(self, results_dir: str | Path = DEFAULT_RESULTS_DIR):
        self.repository = ResultRepository(results_dir)
        self.runs = self.repository.load()
        self.filtered_runs: list[ResultRun] = []
        self.current_run: ResultRun | None = None
        self.current_sample_index = 0

        self.root = tk.Tk()
        self.root.title("Benchmark Results Browser")
        self.root.geometry("1440x900")
        self.root.minsize(1050, 700)
        self.root.configure(bg=self.BG)
        self._configure_styles()
        self._show_dashboard()

    def run(self) -> None:
        self.root.mainloop()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Results.Treeview",
            background=self.CARD,
            fieldbackground=self.CARD,
            foreground=self.TEXT,
            rowheight=34,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.map(
            "Results.Treeview",
            background=[("selected", self.CARD_HOVER)],
            foreground=[("selected", self.TEXT)],
        )
        style.configure(
            "Results.Treeview.Heading",
            background=self.INPUT_BG,
            foreground=self.MUTED,
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Results.Treeview.Heading", background=[("active", self.CARD_HOVER)])
        style.configure(
            "Dark.TCombobox",
            fieldbackground=self.INPUT_BG,
            background=self.CARD,
            foreground=self.TEXT,
            arrowcolor=self.TEXT,
            bordercolor=self.BORDER,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", self.INPUT_BG)],
            foreground=[("readonly", self.TEXT)],
        )
        style.configure(
            "Dark.Vertical.TScrollbar",
            troughcolor=self.BG,
            background=self.BORDER,
            arrowcolor=self.TEXT,
            bordercolor=self.BG,
        )

    def _clear(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()

    def _header(self, title: str, subtitle: str = "", back_command=None) -> tk.Frame:
        frame = tk.Frame(self.root, bg=self.BG)
        frame.pack(fill=tk.X, padx=24, pady=(18, 12))
        if back_command is not None:
            self._button(frame, "< Back", back_command, padx=14, pady=6).pack(
                side=tk.LEFT, padx=(0, 16)
            )
        title_box = tk.Frame(frame, bg=self.BG)
        title_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            title_box,
            text=title,
            bg=self.BG,
            fg=self.TEXT,
            font=("Segoe UI", 20, "bold"),
            anchor="w",
        ).pack(fill=tk.X)
        if subtitle:
            tk.Label(
                title_box,
                text=subtitle,
                bg=self.BG,
                fg=self.MUTED,
                font=("Segoe UI", 9),
                anchor="w",
            ).pack(fill=tk.X, pady=(2, 0))
        return frame

    def _show_dashboard(self) -> None:
        self._clear()
        self._header(
            "Benchmark Results",
            f"Loaded {len(self.runs)} result files from {self.repository.results_dir}",
        )
        self._build_filters()
        self.summary_frame = tk.Frame(self.root, bg=self.BG)
        self.summary_frame.pack(fill=tk.X, padx=24, pady=(2, 12))

        table_card = tk.Frame(
            self.root,
            bg=self.CARD,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        table_card.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 24))
        columns = ("model", "benchmark", "score", "samples", "success", "latency", "group")
        self.run_tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            style="Results.Treeview",
            selectmode="browse",
        )
        headings = {
            "model": "MODEL",
            "benchmark": "BENCHMARK",
            "score": "PRIMARY SCORE",
            "samples": "SAMPLES",
            "success": "RUN HEALTH",
            "latency": "SEC / SAMPLE",
            "group": "RESULT GROUP",
        }
        widths = {
            "model": 190,
            "benchmark": 210,
            "score": 130,
            "samples": 80,
            "success": 105,
            "latency": 110,
            "group": 210,
        }
        for column in columns:
            self.run_tree.heading(column, text=headings[column])
            self.run_tree.column(
                column,
                width=widths[column],
                minwidth=70,
                anchor=tk.W if column in {"model", "benchmark", "group"} else tk.CENTER,
            )
        scrollbar = ttk.Scrollbar(
            table_card,
            orient=tk.VERTICAL,
            command=self.run_tree.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self.run_tree.configure(yscrollcommand=scrollbar.set)
        self.run_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=10)
        self.run_tree.bind("<Double-1>", lambda _event: self._open_selected_run())
        self.run_tree.bind("<Return>", lambda _event: self._open_selected_run())

        footer = tk.Frame(self.root, bg=self.BG)
        footer.pack(fill=tk.X, padx=24, pady=(0, 18))
        self.table_status = tk.StringVar()
        tk.Label(
            footer,
            textvariable=self.table_status,
            bg=self.BG,
            fg=self.MUTED,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)
        self._button(
            footer,
            "Open selected result",
            self._open_selected_run,
            padx=18,
            pady=7,
        ).pack(side=tk.RIGHT)
        self._apply_filters()

    def _build_filters(self) -> None:
        panel = tk.Frame(
            self.root,
            bg=self.CARD,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        panel.pack(fill=tk.X, padx=24, pady=(0, 12))
        self.search_var = tk.StringVar()
        self.model_var = tk.StringVar(value="All models")
        self.benchmark_var = tk.StringVar(value="All benchmarks")
        self.group_var = tk.StringVar(value="All result groups")

        self._filter_label(panel, "SEARCH", 0)
        search = tk.Entry(
            panel,
            textvariable=self.search_var,
            bg=self.INPUT_BG,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            relief=tk.FLAT,
            highlightbackground=self.BORDER,
            highlightcolor=self.ACCENT,
            highlightthickness=1,
            font=("Segoe UI", 10),
        )
        search.grid(row=1, column=0, sticky="ew", padx=(12, 7), pady=(0, 12), ipady=6)
        search.bind("<KeyRelease>", lambda _event: self._apply_filters())

        filter_specs = (
            ("MODEL", self.model_var, sorted({run.model for run in self.runs}), 1, "All models"),
            (
                "BENCHMARK",
                self.benchmark_var,
                sorted({run.benchmark for run in self.runs}),
                2,
                "All benchmarks",
            ),
            (
                "RESULT GROUP",
                self.group_var,
                sorted({run.group for run in self.runs}),
                3,
                "All result groups",
            ),
        )
        for label, variable, values, column, all_label in filter_specs:
            self._filter_label(panel, label, column)
            combo = ttk.Combobox(
                panel,
                textvariable=variable,
                values=[all_label, *values],
                state="readonly",
                style="Dark.TCombobox",
                font=("Segoe UI", 10),
            )
            combo.grid(
                row=1,
                column=column,
                sticky="ew",
                padx=(7, 12 if column == 3 else 7),
                pady=(0, 12),
                ipady=4,
            )
            combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_filters())
        panel.grid_columnconfigure(0, weight=2)
        for column in range(1, 4):
            panel.grid_columnconfigure(column, weight=1)

    def _filter_label(self, parent: tk.Widget, text: str, column: int) -> None:
        tk.Label(
            parent,
            text=text,
            bg=self.CARD,
            fg=self.MUTED,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).grid(row=0, column=column, sticky="ew", padx=12, pady=(10, 4))

    def _apply_filters(self) -> None:
        query = self.search_var.get().strip().casefold()
        model = self.model_var.get()
        benchmark = self.benchmark_var.get()
        group = self.group_var.get()
        filtered = []
        for run in self.runs:
            haystack = f"{run.model} {run.benchmark} {run.group}".casefold()
            if query and query not in haystack:
                continue
            if model != "All models" and run.model != model:
                continue
            if benchmark != "All benchmarks" and run.benchmark != benchmark:
                continue
            if group != "All result groups" and run.group != group:
                continue
            filtered.append(run)
        self.filtered_runs = sorted(
            filtered,
            key=lambda run: (
                run.benchmark.casefold(),
                _score_sort_value(run),
                run.model.casefold(),
            ),
        )
        self._render_summary_cards()
        for item in self.run_tree.get_children():
            self.run_tree.delete(item)
        for index, run in enumerate(self.filtered_runs):
            health = f"{run.successes}/{run.completed}" if run.completed else "-"
            latency = f"{run.seconds_per_sample:.2f}" if run.seconds_per_sample is not None else "-"
            self.run_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    run.model,
                    run.benchmark,
                    f"{run.score_label}: {run.display_score}",
                    run.completed,
                    health,
                    latency,
                    run.group,
                ),
            )
        self.table_status.set(
            f"{len(self.filtered_runs)} visible runs. Double-click a row to inspect samples."
        )

    def _render_summary_cards(self) -> None:
        for child in self.summary_frame.winfo_children():
            child.destroy()
        completed = sum(run.completed for run in self.filtered_runs)
        failures = sum(run.failures for run in self.filtered_runs)
        models = len({run.model for run in self.filtered_runs})
        benchmarks = len({run.benchmark for run in self.filtered_runs})
        latencies = [
            run.seconds_per_sample
            for run in self.filtered_runs
            if run.seconds_per_sample is not None
        ]
        mean_latency = sum(latencies) / len(latencies) if latencies else None
        values = (
            ("VISIBLE RUNS", str(len(self.filtered_runs)), self.ACCENT),
            ("MODELS", str(models), self.PURPLE),
            ("BENCHMARKS", str(benchmarks), self.SUCCESS),
            ("SAMPLES", f"{completed:,}", self.TEXT),
            ("FAILURES", f"{failures:,}", self.ERROR if failures else self.SUCCESS),
            (
                "MEAN LATENCY",
                f"{mean_latency:.2f}s" if mean_latency is not None else "-",
                self.WARNING,
            ),
        )
        for column, (label, value, color) in enumerate(values):
            self.summary_frame.grid_columnconfigure(column, weight=1, uniform="summary")
            self._metric_card(self.summary_frame, label, value, color).grid(
                row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0)
            )

    def _metric_card(self, parent: tk.Widget, label: str, value: str, color: str) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=self.CARD,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        tk.Label(
            card,
            text=label,
            bg=self.CARD,
            fg=self.MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=12, pady=(9, 1))
        tk.Label(
            card,
            text=value,
            bg=self.CARD,
            fg=color,
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w", padx=12, pady=(0, 9))
        return card

    def _open_selected_run(self) -> None:
        selection = self.run_tree.selection()
        if not selection:
            return
        self.current_run = self.filtered_runs[int(selection[0])]
        self.current_sample_index = 0
        self._show_run_detail()

    def _show_run_detail(self) -> None:
        run = self.current_run
        if run is None:
            return
        self._clear()
        header = self._header(
            f"{run.model} / {run.benchmark}",
            f"{run.group}  |  {run.path.name}",
            self._show_dashboard,
        )
        self._button(
            header,
            "Open result folder",
            lambda: self._open_folder(run.path.parent),
            padx=14,
            pady=6,
        ).pack(side=tk.RIGHT)

        metrics = tk.Frame(self.root, bg=self.BG)
        metrics.pack(fill=tk.X, padx=24, pady=(0, 12))
        success_rate = run.successes / run.completed if run.completed else None
        cards = (
            (run.score_label.upper(), run.display_score, self.SUCCESS),
            ("SAMPLES", str(run.completed), self.TEXT),
            (
                "RUN HEALTH",
                f"{success_rate * 100:.1f}%" if success_rate is not None else "-",
                self.SUCCESS if run.failures == 0 else self.WARNING,
            ),
            (
                "SEC / SAMPLE",
                f"{run.seconds_per_sample:.2f}" if run.seconds_per_sample is not None else "-",
                self.ACCENT,
            ),
            ("FAILURES", str(run.failures), self.ERROR if run.failures else self.SUCCESS),
        )
        for column, card in enumerate(cards):
            metrics.grid_columnconfigure(column, weight=1, uniform="detail")
            self._metric_card(metrics, *card).grid(
                row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0)
            )

        content = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6, bg=self.BG)
        content.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 24))
        left = tk.Frame(
            content,
            bg=self.CARD,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        right = tk.Frame(
            content,
            bg=self.CARD,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        content.add(left, minsize=360, width=440)
        content.add(right, minsize=500, stretch="always")
        self._build_sample_list(left, run)
        self._build_sample_detail(right)
        self._render_sample()

    def _build_sample_list(self, parent: tk.Frame, run: ResultRun) -> None:
        tk.Label(
            parent,
            text="SAMPLES",
            bg=self.CARD,
            fg=self.MUTED,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(12, 6))
        columns = ("index", "status", "score", "latency")
        self.sample_tree = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            style="Results.Treeview",
            selectmode="browse",
        )
        for column, title, width in (
            ("index", "#", 46),
            ("status", "RESULT", 85),
            ("score", "SCORE", 82),
            ("latency", "SECONDS", 72),
        ):
            self.sample_tree.heading(column, text=title)
            self.sample_tree.column(column, width=width, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=self.sample_tree.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self.sample_tree.configure(yscrollcommand=scrollbar.set)
        self.sample_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 10))
        for position, sample in enumerate(run.samples):
            status = _sample_status(sample)
            score = _sample_score_text(sample)
            sample_stats = sample.get("stats", {}) if isinstance(sample.get("stats"), dict) else {}
            latency = _as_float(sample_stats.get("wall_clock_time_seconds"))
            self.sample_tree.insert(
                "",
                tk.END,
                iid=str(position),
                values=(
                    sample.get("index", position + 1),
                    status,
                    score,
                    f"{latency:.2f}" if latency is not None else "-",
                ),
            )
        self.sample_tree.bind("<<TreeviewSelect>>", self._select_sample)
        if run.samples:
            self.sample_tree.selection_set("0")
            self.sample_tree.focus("0")

    def _build_sample_detail(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg=self.CARD)
        top.pack(fill=tk.X, padx=12, pady=(12, 8))
        self.sample_title = tk.StringVar()
        self.sample_status = tk.StringVar()
        tk.Label(
            top,
            textvariable=self.sample_title,
            bg=self.CARD,
            fg=self.TEXT,
            font=("Segoe UI", 14, "bold"),
        ).pack(side=tk.LEFT)
        self.sample_status_label = tk.Label(
            top,
            textvariable=self.sample_status,
            bg=self.CARD,
            fg=self.SUCCESS,
            font=("Segoe UI", 10, "bold"),
        )
        self.sample_status_label.pack(side=tk.RIGHT)

        tk.Label(
            parent,
            text="MODEL OUTPUT",
            bg=self.CARD,
            fg=self.ACCENT,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12)
        self.prediction_text = self._text_panel(parent, height=7, font=("Consolas", 11))
        tk.Label(
            parent,
            text="REFERENCE / EVALUATION",
            bg=self.CARD,
            fg=self.SUCCESS,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(8, 0))
        self.reference_text = self._text_panel(parent, height=7, font=("Segoe UI", 10))
        tk.Label(
            parent,
            text="SAMPLE TELEMETRY",
            bg=self.CARD,
            fg=self.MUTED,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(8, 0))
        self.telemetry_text = self._text_panel(parent, height=8, font=("Consolas", 9))

    def _text_panel(self, parent: tk.Widget, height: int, font: tuple) -> ScrolledText:
        widget = ScrolledText(
            parent,
            wrap=tk.WORD,
            height=height,
            font=font,
            bg=self.INPUT_BG,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            selectbackground=self.ACCENT,
            selectforeground=self.INPUT_BG,
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=8,
        )
        widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 0))
        self._style_text_scrollbar(widget)
        widget.configure(state=tk.DISABLED)
        return widget

    def _select_sample(self, _event=None) -> None:
        selection = self.sample_tree.selection()
        if not selection:
            return
        self.current_sample_index = int(selection[0])
        self._render_sample()

    def _render_sample(self) -> None:
        run = self.current_run
        if run is None or not run.samples:
            return
        sample = run.samples[self.current_sample_index]
        index = sample.get("index", self.current_sample_index + 1)
        status = _sample_status(sample)
        self.sample_title.set(f"Sample {index}")
        self.sample_status.set(status.upper())
        self.sample_status_label.configure(
            fg=self.SUCCESS if status in {"correct", "success"} else self.ERROR
        )
        self._set_text(self.prediction_text, str(sample.get("prediction") or "(no output)"))
        self._set_text(self.reference_text, _reference_text(sample))
        telemetry = sample.get("stats", {}) if isinstance(sample.get("stats"), dict) else {}
        evaluation = {
            key: value
            for key, value in sample.items()
            if key
            not in {
                "prediction",
                "stats",
                "prompt_labels",
                "valid_labels",
                "reference_answers",
                "reference_captions",
                "choices",
                "ground_truth_boxes",
                "predicted_boxes",
            }
        }
        text = "Evaluation\n" + json.dumps(evaluation, indent=2, ensure_ascii=False)
        text += "\n\nRuntime statistics\n" + json.dumps(telemetry, indent=2, ensure_ascii=False)
        self._set_text(self.telemetry_text, text)

    @staticmethod
    def _set_text(widget: ScrolledText, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state=tk.DISABLED)

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

    def _open_folder(self, path: Path) -> None:
        import os

        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


def _score_sort_value(run: ResultRun) -> float:
    if run.score is None:
        return float("inf")
    return run.score if run.score_is_error else -run.score


def _sample_status(sample: dict[str, Any]) -> str:
    correct = sample.get("correct")
    if isinstance(correct, bool):
        return "correct" if correct else "incorrect"
    stats = sample.get("stats")
    if isinstance(stats, dict) and stats.get("success") is False:
        return "failed"
    return "success"


def _sample_score_text(sample: dict[str, Any]) -> str:
    for key, percent in (("f1", True), ("bleu", True), ("absolute_error", False)):
        value = _as_float(sample.get(key))
        if value is not None:
            return f"{value * 100:.1f}%" if percent else f"{value:.2f}"
    correct = sample.get("correct")
    if isinstance(correct, bool):
        return "1" if correct else "0"
    return "-"


def _reference_text(sample: dict[str, Any]) -> str:
    sections: list[str] = []
    for key, label in (
        ("valid_labels", "Valid labels"),
        ("reference_answers", "Reference answers"),
        ("reference_captions", "Reference captions"),
        ("choices", "Choices"),
        ("prompt_labels", "Prompt labels"),
        ("ground_truth_boxes", "Ground-truth boxes"),
        ("predicted_boxes", "Predicted boxes"),
    ):
        value = sample.get(key)
        if value:
            rendered = json.dumps(value, indent=2, ensure_ascii=False)
            sections.append(f"{label}\n{rendered}")
    for key, label in (
        ("selected_choice", "Selected choice"),
        ("predicted_rating", "Predicted rating"),
        ("target_rating", "Target rating"),
    ):
        if sample.get(key) is not None:
            sections.append(f"{label}\n{sample[key]}")
    return "\n\n".join(sections) or "No reference fields were recorded for this sample."


def main() -> int:
    results_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    BenchmarkResultsBrowser(results_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
