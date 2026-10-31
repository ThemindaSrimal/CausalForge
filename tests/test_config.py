import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from causalforge.config import load_dotenv


class ConfigTests(unittest.TestCase):
    def test_loads_quoted_value_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text('# local settings\nMODEL_API_KEY="secret value"\n', encoding="utf-8")
            load_dotenv(path)
            self.assertEqual(os.environ["MODEL_API_KEY"], "secret value")

    def test_existing_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MODEL_API_KEY": "process-value"}, clear=True):
            path = Path(directory) / ".env"
            path.write_text("MODEL_API_KEY=file-value\n", encoding="utf-8")
            load_dotenv(path)
            self.assertEqual(os.environ["MODEL_API_KEY"], "process-value")
