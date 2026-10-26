import unittest

from navai.prompts import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    build_prompt,
    format_context,
    format_history,
    prompt_fingerprint,
)


class SystemInstructionTest(unittest.TestCase):
    def test_uses_current_prompt_version(self):
        self.assertEqual(PROMPT_VERSION, "v3")

    def test_stays_concise(self):
        self.assertLess(len(SYSTEM_INSTRUCTION.split()), 300)

    def test_uses_structured_sections(self):
        for section in (
            "<system_role>",
            "<context>",
            "<audience>",
            "<task>",
            "<format>",
            "<tone_and_constraints>",
        ):
            self.assertIn(section, SYSTEM_INSTRUCTION)

    def test_keeps_scope_and_refusal_rules(self):
        text = SYSTEM_INSTRUCTION.lower()

        self.assertIn("question's scope", text)
        self.assertIn("short, accurate, and helpful", text)
        self.assertIn("contacting navai", text)
        self.assertIn("question is unrelated", text)
        self.assertIn("do not add citations when refusing", text)


class PromptFingerprintTest(unittest.TestCase):
    def test_returns_short_sha256_digest(self):
        self.assertRegex(prompt_fingerprint(), r"^[0-9a-f]{12}$")

    def test_pins_version_to_prompt_digest(self):
        self.assertEqual(
            (PROMPT_VERSION, prompt_fingerprint()), ("v3", "3d613ceffaf4")
        )

    def test_changes_with_text(self):
        self.assertNotEqual(
            prompt_fingerprint("prompt one"),
            prompt_fingerprint("prompt two"),
        )


class HistoryTest(unittest.TestCase):
    def test_formats_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        self.assertEqual(
            format_history(history), "User: Hi\n\nAssistant: Hello"
        )

    def test_formats_empty_history(self):
        self.assertEqual(format_history([]), "")

    def test_builds_prompt_with_history(self):
        history = [{"role": "user", "content": "Hi"}]

        self.assertEqual(
            build_prompt("What is NavAI?", "CTX", history),
            "Conversation history:\nUser: Hi\n\n"
            "Context:\nCTX\n\nQuestion: What is NavAI?",
        )

    def test_builds_prompt_without_history(self):
        self.assertEqual(
            build_prompt("What is NavAI?", "CTX", []),
            "Context:\nCTX\n\nQuestion: What is NavAI?",
        )


class FormatContextTest(unittest.TestCase):
    def test_wraps_chunks_in_numbered_tags(self):
        chunks = [
            {
                "text": "First chunk.",
                "metadata": {"doc": "Doc A", "chunk_index": 0},
            },
            {
                "text": "Second chunk.",
                "metadata": {"doc": "Doc B", "chunk_index": 4},
            },
        ]

        self.assertEqual(
            format_context(chunks),
            "<1>\nDoc A (chunk 0)\nFirst chunk.\n</1>\n\n"
            "<2>\nDoc B (chunk 4)\nSecond chunk.\n</2>",
        )


if __name__ == "__main__":
    unittest.main()
