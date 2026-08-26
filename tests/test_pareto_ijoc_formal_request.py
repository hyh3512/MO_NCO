from __future__ import annotations

import importlib.util
from pathlib import Path
import platform
import unittest


_BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v20"
    / "scripts"
    / "build_formal_freeze_request.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "ijoc_build_formal_freeze_request_contract", _BUILDER_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load the formal freeze-request builder.")
_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BUILDER)


class FormalFreezeRequestContractTests(unittest.TestCase):
    def test_python_version_matches_cold_runner_exact_version_gate(self) -> None:
        self.assertEqual(
            _BUILDER.formal_python_version(),
            platform.python_version(),
        )


if __name__ == "__main__":
    unittest.main()

