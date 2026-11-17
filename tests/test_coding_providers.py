from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wbc.coding.providers import PROVIDERS, key_from_file, write_catalog
from wbc.coding.run import codex_args, sandbox


class ProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_plain_and_markdown_key_files(self):
        path = self.root / "key"
        for content in ("sk-test_secret\n", "# API\nEndpoint: https://example.test/v1\nKey: `sk-test_secret`\n"):
            with self.subTest(content=content):
                path.write_text(content)
                self.assertEqual(key_from_file(path), "sk-test_secret")

    def test_invalid_file_errors_never_include_contents(self):
        path = self.root / "key"
        for content in ("", "private unrelated document", "sk-first_secret sk-second_secret"):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(ValueError) as caught:
                    key_from_file(path)
                self.assertEqual(str(caught.exception), "API key file must contain exactly one sk- credential")

    def test_selected_model_matches_codex_and_read_only_catalog(self):
        template = self.root / "template.json"
        original = {"models": [{"slug": "old", "display_name": "Old", "input_modalities": ["text", "image"]}]}
        template.write_text(json.dumps(original))
        catalog = self.root / "models.json"
        provider = PROVIDERS["dpsk"]
        write_catalog(template, catalog, provider)
        generated = json.loads(catalog.read_text())
        self.assertEqual(generated["models"][0]["slug"], "deepseek-v4.1-flash")
        self.assertEqual(generated["models"][0]["input_modalities"], ["text", "image"])
        self.assertEqual(json.loads(template.read_text()), original)
        args = codex_args("task", None, provider)
        self.assertEqual(args[args.index("-m") + 1], generated["models"][0]["slug"])
        mounts = sandbox(self.root, self.root / "socket", self.root / "codex", catalog)
        index = mounts.index(str(catalog))
        self.assertEqual(mounts[index - 1 : index + 2], ["--ro-bind", str(catalog), "/app/models.json"])


if __name__ == "__main__":
    unittest.main()
