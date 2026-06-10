from dataset import VisualCoT

from ._detection import DetectionBenchmark


class VisualCoTDetectionBenchmark(DetectionBenchmark):
    dataset_cls = VisualCoT
    benchmark_name = "visual_cot_detection"
    default_split = "train"

    def make_prompt(self, labels, row=None, image=None):
        del labels
        del image
        if row is None:
            raise ValueError("VisualCoTDetectionBenchmark requires a dataset row.")
        question = str(row.get("question", "")).strip()
        if not question:
            question = self.dataset.get_question_from_row(row)
        return (
            "USER: <image>\n\n"
            f"Question: {question}\n\n"
            "Locate the single visible image region that provides the visual evidence needed "
            "to answer the question.\n\n"
            "Return exactly one bounding box using this structure:\n\n"
            "[x, y, width, height]\n\n"
            "x and y are the normalized coordinates of the top-left corner. Width and height "
            "are normalized relative to the full image. Use decimal values from 0 to 1.\n"
            "Return only the box, without labels, prose, markdown, or confidence scores.\n\n"
            "ASSISTANT:"
        )
