"""Run only workspace tests; no installed ComfyUI or external model needed."""
import pathlib
import sys
import unittest
root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(root))
suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
