import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import navai.settings as settings
from navai.settings import database_path


class ConfigTest(unittest.TestCase):
    def test_loads_current_config(self):
        config = settings.load_config()

        self.assertGreater(config["chunk_size"], 0)
        self.assertGreater(config["retrieval_k"], 0)
        self.assertGreater(config["crossencoder_k"], 0)
        self.assertLessEqual(config["crossencoder_k"], config["retrieval_k"])
        self.assertAlmostEqual(config["temperature"], 0.1)

    def test_rejects_out_of_range_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "chunk_size": 256,
                        "retrieval_k": 20,
                        "crossencoder_k": 4,
                        "temperature": 3,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(settings, "CONFIG_PATH", path):
                with self.assertRaises(ValueError):
                    settings.load_config()


class DatabasePathTest(unittest.TestCase):
    def test_256_keeps_existing_database(self):
        self.assertEqual(database_path(256).name, "chroma")

    def test_smaller_chunks_use_separate_databases(self):
        self.assertEqual(database_path(64).name, "chroma_64")
        self.assertEqual(database_path(128).name, "chroma_128")

    def test_rejects_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            database_path(0)


if __name__ == "__main__":
    unittest.main()
