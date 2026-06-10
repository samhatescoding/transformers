from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from PIL import Image

from dataset.mscoco_caption import MSCOCOCaption


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (3, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class MSCOCOCaptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = MSCOCOCaption.__new__(MSCOCOCaption)

    @patch("dataset.mscoco_caption.time.sleep")
    @patch("dataset.mscoco_caption.urlopen")
    def test_image_download_retries_transient_http_errors(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = _png_bytes()
        mock_urlopen.side_effect = [
            HTTPError(
                "https://images.cocodataset.org/train2014/example.jpg",
                503,
                "Service Unavailable",
                None,
                None,
            ),
            response,
        ]

        image = self.dataset.get_image_from_row(
            {"url": "http://images.cocodataset.org/train2014/example.jpg"}
        )

        self.assertEqual(image.size, (3, 2))
        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called_once_with(1)
        request = mock_urlopen.call_args_list[0].args[0]
        self.assertEqual(request.full_url.split(":", 1)[0], "https")
        self.assertEqual(request.get_header("User-agent"), "transformers-benchmark/1.0")

    @patch("dataset.mscoco_caption.time.sleep")
    @patch("dataset.mscoco_caption.urlopen")
    def test_image_download_does_not_retry_permanent_http_errors(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_urlopen.side_effect = HTTPError(
            "https://images.cocodataset.org/train2014/missing.jpg",
            404,
            "Not Found",
            None,
            None,
        )

        with self.assertRaises(HTTPError):
            self.dataset.get_image_from_row(
                {"url": "https://images.cocodataset.org/train2014/missing.jpg"}
            )

        mock_urlopen.assert_called_once()
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
