"""Offline regression checks for context maintenance; no Aria dependencies needed."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_context.py"
SPEC = importlib.util.spec_from_file_location("check_context", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ContextChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.context = self.root / "docs/agents"
        self.context.mkdir(parents=True)
        self.config = {
            "areas": {"runtime": {"sources": ["app/"], "context": ["docs/agents/architecture.md"]}},
            "review_log": "docs/agents/reviews.md",
            "instruction_dirs": [""],
        }
        for name in ["project", "decisions", "code-intelligence", "maintenance", "reviews", "architecture"]:
            (self.context / f"{name}.md").write_text(f"# {name}\n")
        (self.root / "AGENTS.md").write_text("# Instructions\n")
        (self.root / "CLAUDE.md").write_text("# Shared project instructions\n\n@AGENTS.md\n")
        (self.context / "source-map.json").write_text(json.dumps(self.config))
        self.addCleanup(patch.stopall)
        patch.object(checker, "ROOT", self.root).start()
        patch.object(checker, "CONTEXT", self.context).start()

    def test_valid_context(self):
        self.assertEqual(checker.check_structure(self.config)[0], [])

    def test_missing_and_nonportable_links(self):
        (self.root / "AGENTS.md").write_text("[missing](missing.md)\n[outside](../outside.md)\n")
        errors, _ = checker.check_structure(self.config)
        self.assertTrue(any("missing link target" in e for e in errors))
        self.assertTrue(any("nonportable link" in e for e in errors))

    def test_spaced_link_and_external_url(self):
        (self.root / "with spaces.md").write_text("# Target\n")
        (self.root / "AGENTS.md").write_text("[local](<with spaces.md>)\n[web](https://example.org/docs)\n")
        self.assertEqual(checker.check_structure(self.config)[0], [])

    def test_duplicate_claude_policy_and_oversized_seed(self):
        (self.root / "CLAUDE.md").write_text("Different policy\n")
        (self.root / "AGENTS.md").write_text("Rule\n" * 51)
        errors, _ = checker.check_structure(self.config)
        self.assertTrue(any("wrapper" in e for e in errors))
        self.assertTrue(any("exceeds 50" in e for e in errors))

    def test_missing_required_file(self):
        (self.context / "project.md").unlink()
        self.assertTrue(any("Missing context file" in e for e in checker.check_structure(self.config)[0]))

    def test_source_change_requires_review(self):
        self.assertTrue(checker.check_coverage(self.config, {"app/main.py"}))
        self.assertEqual(checker.check_coverage(self.config, {"app/main.py", "docs/agents/architecture.md"}), [])
        self.assertEqual(checker.check_coverage(self.config, {"app/main.py", "docs/agents/reviews.md"}), [])
        self.assertEqual(checker.check_coverage(self.config, {"app/AGENTS.md"}), [])

    def test_git_changes_include_deleted_staged_unstaged_and_untracked(self):
        def run(*args):
            subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

        run("init", "-q")
        (self.root / "app").mkdir()
        for name in ["deleted.py", "staged.py", "unstaged.py"]:
            (self.root / "app" / name).write_text("# Initial\n")
        run("add", ".")
        run("-c", "user.name=Context Test", "-c", "user.email=context@example.invalid", "commit", "-qm", "Fixture")
        (self.root / "app/deleted.py").unlink()
        (self.root / "app/staged.py").write_text("# Staged\n")
        run("add", "app/staged.py")
        (self.root / "app/unstaged.py").write_text("# Unstaged\n")
        (self.root / "app/untracked.py").write_text("# New\n")
        self.assertEqual(checker.changed_paths("HEAD"), {
            "app/deleted.py", "app/staged.py", "app/unstaged.py", "app/untracked.py"
        })


if __name__ == "__main__":
    unittest.main()
