import unittest

from chunking import chunk_text


class CharacterTokenizer:
    """One token per character, enough to exercise offset slicing."""

    def __call__(
        self, text, add_special_tokens=False, return_offsets_mapping=False
    ):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ChunkTextTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_fixed_windows_with_overlap(self):
        self.assertEqual(
            chunk_text("abcdefghij", self.tokenizer, size=4, overlap=1),
            [
                (0, 4, "abcd"),
                (3, 4, "defg"),
                (6, 4, "ghij"),
            ],
        )

    def test_short_final_window(self):
        self.assertEqual(
            chunk_text("abcdefg", self.tokenizer, size=4, overlap=2),
            [
                (0, 4, "abcd"),
                (2, 4, "cdef"),
                (4, 3, "efg"),
            ],
        )

    def test_text_shorter_than_window(self):
        self.assertEqual(
            chunk_text("abc", self.tokenizer, size=4, overlap=1),
            [(0, 3, "abc")],
        )

    def test_preserves_original_characters(self):
        self.assertEqual(
            chunk_text("NaV AI.", self.tokenizer, size=6, overlap=0)[0][2],
            "NaV AI",
        )

    def test_empty_text(self):
        self.assertEqual(chunk_text("", self.tokenizer, size=4, overlap=1), [])

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", self.tokenizer, size=4, overlap=4)

    def test_rejects_zero_size(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", self.tokenizer, size=0, overlap=0)


if __name__ == "__main__":
    unittest.main()
