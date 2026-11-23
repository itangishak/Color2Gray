import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as module
from utils.color2gray_impl import color2gray_vanilla, color2gray_enhanced


class Color2GrayTests(unittest.TestCase):

    def test_algorithms(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :16] = [220, 50, 50]
        img[:, 16:] = [50, 120, 200]

        for method in (color2gray_vanilla, color2gray_enhanced):
            result = method(img)
            self.assertEqual(result.shape, (32, 32))
            self.assertEqual(result.dtype, np.uint8)

    def test_uniform_image(self):
        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        for method in (color2gray_vanilla, color2gray_enhanced):
            result = method(img)
            self.assertEqual(int(result.max()), int(result.min()))

    def test_solver_failure_detection(self):
        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        with patch(
            "utils.color2gray_impl.cg",
            return_value=(np.zeros(256), 1)
        ):
            with self.assertRaises(RuntimeError):
                color2gray_vanilla(img)

    def test_flask_workflow(self):
        old_upload = module.UPLOAD_FOLDER
        old_result = module.RESULT_FOLDER

        with tempfile.TemporaryDirectory() as tmp:
            try:
                module.UPLOAD_FOLDER = str(Path(tmp) / "uploads")
                module.RESULT_FOLDER = str(Path(tmp) / "results")
                Path(module.UPLOAD_FOLDER).mkdir()
                Path(module.RESULT_FOLDER).mkdir()

                client = module.app.test_client()

                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.post("/", data={}).status_code, 400)

                response = client.post(
                    "/",
                    data={"file": (io.BytesIO(b"invalid"), "bad.jpg")},
                    content_type="multipart/form-data"
                )
                self.assertEqual(response.status_code, 400)

                buffer = io.BytesIO()
                Image.new("RGB", (32, 32), "red").save(buffer, "JPEG")
                buffer.seek(0)

                response = client.post(
                    "/",
                    data={
                        "file": (buffer, "test.jpg"),
                        "method": "all",
                        "theta": "30",
                        "alpha": "4"
                    },
                    content_type="multipart/form-data"
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    len(list(Path(module.RESULT_FOLDER).glob("*.jpg"))), 3
                )
            finally:
                module.UPLOAD_FOLDER = old_upload
                module.RESULT_FOLDER = old_result


if __name__ == "__main__":
    unittest.main()
