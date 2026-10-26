"""Keep source-integrity checks stable across Windows and Unix Git checkouts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bp_core.combined import PROCESSING_FILES, _code_hashes


ROOT = Path(__file__).resolve().parents[2]


class ProcessingSourceLineEndingTests(unittest.TestCase):
    def test_frozen_processing_sources_use_lf(self):
        for name in PROCESSING_FILES:
            with self.subTest(source=name):
                self.assertEqual((ROOT / "tools" / name).read_bytes().count(b"\r"), 0)

    @unittest.skipUnless(shutil.which("git"), "Git is required for checkout regression")
    def test_git_checkout_preserves_hashes_with_windows_autocrlf(self):
        attributes = (ROOT / ".gitattributes").read_bytes()
        for autocrlf in ("true", "false", "input"):
            with self.subTest(autocrlf=autocrlf), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ".gitattributes").write_bytes(attributes)
                for index, name in enumerate(PROCESSING_FILES):
                    source = root / "tools" / name
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_bytes(f"def feature_{index}():\n    return {index}\n".encode())

                def git(*arguments):
                    return subprocess.run(
                        [shutil.which("git"), "-c", f"core.autocrlf={autocrlf}", *arguments],
                        cwd=root, capture_output=True, text=True, check=True,
                    )

                git("init", "--quiet")
                git("add", "--", ".gitattributes", "tools")
                with patch("bp_core.combined.ROOT", root):
                    frozen_hashes = _code_hashes()
                    for name in PROCESSING_FILES:
                        source = root / "tools" / name
                        source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
                    # Only these isolated temporary fixtures are overwritten.
                    git("checkout-index", "--all", "--force")
                    self.assertEqual(_code_hashes(), frozen_hashes)
                    for name in PROCESSING_FILES:
                        self.assertEqual((root / "tools" / name).read_bytes().count(b"\r"), 0)

    def test_actual_source_edit_still_changes_integrity_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tools" / "fixture.py"
            source.parent.mkdir()
            source.write_bytes(b"def feature():\n    return 1\n")
            with patch("bp_core.combined.ROOT", root), patch(
                "bp_core.combined.PROCESSING_FILES", ("fixture.py",),
            ):
                frozen_hashes = _code_hashes()
                source.write_bytes(b"def feature():\n    return 2\n")
                self.assertNotEqual(_code_hashes(), frozen_hashes)


if __name__ == "__main__":
    unittest.main()
