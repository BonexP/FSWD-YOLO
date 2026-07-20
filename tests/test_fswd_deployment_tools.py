# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import fswd_deploy_common as common  # noqa: E402


class CommonUtilitiesTests(unittest.TestCase):
    def test_validate_weights_requires_existing_pt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"weights")

            with self.assertRaisesRegex(ValueError, "must use the .pt suffix"):
                common.validate_weights(path)

    def test_validate_weights_requires_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.pt"

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                common.validate_weights(path)

    def test_prepare_output_rejects_existing_file_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"existing")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                common.prepare_output(output, overwrite=False)

    def test_prepare_output_requires_expected_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.bin"

            with self.assertRaisesRegex(ValueError, "must use the .onnx suffix"):
                common.prepare_output(output, overwrite=False)

    def test_sha256_file_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(b"fswd")

            self.assertEqual(common.sha256_file(path), hashlib.sha256(b"fswd").hexdigest())

    def test_write_json_atomic_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"

            common.write_json_atomic(path, {"status": "ok"})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "ok"})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_write_json_atomic_respects_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                common.write_json_atomic(path, {"status": "ok"}, overwrite=False)

    def test_summarize_operators_counts_review_operations(self):
        summary = common.summarize_operators(["Conv", "Div", "Conv", "MatMul", "Transpose"])

        self.assertEqual(summary["histogram"], {"Conv": 2, "Div": 1, "MatMul": 1, "Transpose": 1})
        self.assertEqual(summary["review_operations"], {"Div": 1, "MatMul": 1, "Transpose": 1})

    def test_require_modules_reports_missing_dependency_without_installing(self):
        missing_name = "fswd_package_that_does_not_exist"

        with self.assertRaises(common.DependencyError) as context:
            common.require_modules({missing_name: "manual installation only"}, "unit test")

        message = str(context.exception)
        self.assertIn(missing_name, message)
        self.assertIn("manual installation only", message)
        self.assertIn(sys.executable, message)
        self.assertIn("does not install packages", message)


if __name__ == "__main__":
    unittest.main()
