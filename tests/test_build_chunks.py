import unittest

from navai.settings import chunk_file


class OutputPathTest(unittest.TestCase):
    def test_256_keeps_existing_chunk_file(self):
        self.assertEqual(chunk_file(256).name, "chunks.jsonl")

    def test_smaller_chunks_use_separate_files(self):
        self.assertEqual(chunk_file(64).name, "chunks_64.jsonl")
        self.assertEqual(chunk_file(128).name, "chunks_128.jsonl")


if __name__ == "__main__":
    unittest.main()
