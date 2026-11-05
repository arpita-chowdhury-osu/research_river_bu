"""Tests for download.fetch(): saving a verified download over an existing wrong file.

Run from the repository root (standard library only, no network):
    python tests/test_download_replace.py -v

Only fetch(row, out, retries=1) is called; main() is never run, so no split CSV is read and nothing is
downloaded. urllib.request.urlopen is replaced by a fake that returns small made-up bytes (they are not
images). Files are written to a real temporary directory; rename/replace/exists are not mocked, so the
check runs against the real file system of the machine.

Case c failed with the previous `tmp.rename(dest)` on Windows (FileExistsError, WinError 183): the
verified download stayed in <name>.part and the wrong file remained.
"""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("download", ROOT / "download.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

GOOD = b"made-up correct bytes\n" * 4
OLD = b"made-up old wrong bytes\n" * 3
BAD = b"made-up wrong download\n" * 5


def sha(b):
    return hashlib.sha256(b).hexdigest()


class FakeUrlopen:
    """Returns fixed bytes and counts calls instead of contacting a server."""

    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        payload = self.payload

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return payload
        return Response()


class FetchReplaceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.row = {"site": "TEST", "date": "2024-01-01", "filename": "made_up.jpg",
                    "url": "https://example.invalid/made_up.jpg", "sha256": sha(GOOD)}
        self.dest = self.out / "TEST" / "2024" / "made_up.jpg"

    def tearDown(self):
        self._tmp.cleanup()

    def fetch(self, payload):
        fake = FakeUrlopen(payload)
        with mock.patch.object(download.urllib.request, "urlopen", fake):
            status, name = download.fetch(self.row, self.out, retries=1)
        return status, fake.calls

    def put_existing(self, data):
        self.dest.parent.mkdir(parents=True)
        self.dest.write_bytes(data)

    def test_a_missing_target_is_saved(self):
        status, calls = self.fetch(GOOD)
        self.assertEqual(status, "downloaded")
        self.assertEqual(calls, 1)
        self.assertEqual(self.dest.read_bytes(), GOOD)
        self.assertFalse(self.dest.with_suffix(".part").exists())

    def test_b_correct_target_is_skipped_without_network(self):
        self.put_existing(GOOD)
        mtime = self.dest.stat().st_mtime_ns
        status, calls = self.fetch(BAD)
        self.assertEqual(status, "skipped")
        self.assertEqual(calls, 0)
        self.assertEqual(self.dest.read_bytes(), GOOD)
        self.assertEqual(self.dest.stat().st_mtime_ns, mtime)

    def test_c_wrong_target_is_replaced_then_skipped(self):
        self.put_existing(OLD)
        status, calls = self.fetch(GOOD)
        self.assertEqual(status, "downloaded")
        self.assertEqual(calls, 1)
        self.assertEqual(sha(self.dest.read_bytes()), sha(GOOD))
        self.assertFalse(self.dest.with_suffix(".part").exists())
        status, calls = self.fetch(BAD)
        self.assertEqual(status, "skipped")
        self.assertEqual(calls, 0)
        self.assertEqual(self.dest.read_bytes(), GOOD)

    def test_d_bad_download_is_rejected_and_old_target_kept(self):
        self.put_existing(OLD)
        status, calls = self.fetch(BAD)
        self.assertTrue(status.startswith("failed"), status)
        self.assertIn("checksum mismatch", status)
        self.assertEqual(calls, 1)
        self.assertEqual(self.dest.read_bytes(), OLD)
        self.assertFalse(self.dest.with_suffix(".part").exists())


if __name__ == "__main__":
    unittest.main()
