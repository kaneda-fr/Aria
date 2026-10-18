#!/usr/bin/env python3
"""Check portable LLM context without importing Aria or contacting services.

--base REF additionally checks that changed source areas have a context review.
This checks structure/review signals, not the semantic truth of documentation.
"""

import argparse
import fnmatch
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "docs/agents"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def changed_paths(base):
    ancestor = git("merge-base", "HEAD", base).strip()
    # A diff against a commit includes staged and unstaged changes, including deletions.
    changed = git("diff", "--name-only", "--no-renames", "-z", ancestor, "--")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    return set(filter(None, (changed + untracked).split("\0")))


def matches(path, pattern):
    return path.startswith(pattern) if pattern.endswith("/") else fnmatch.fnmatchcase(path, pattern)


def local_targets(text):
    # Maintained context uses inline Markdown links and standalone Claude imports.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    for match in re.finditer(r"\[[^\]]*\]\((<[^>]+>|[^)\s]+)\)", text):
        yield match.group(1).strip("<>")
    yield from re.findall(r"^@([^\s]+)\s*$", text, flags=re.M)


def check_structure(config):
    errors = []
    files = set(CONTEXT.glob("*.md"))
    for directory in config["instruction_dirs"]:
        agents = ROOT / directory / "AGENTS.md"
        claude = ROOT / directory / "CLAUDE.md"
        files.update([agents, claude])
        if claude.exists() and claude.read_text() != "# Shared project instructions\n\n@AGENTS.md\n":
            errors.append(f"{claude.relative_to(ROOT)}: keep the shared AGENTS.md import wrapper")
        if agents.exists():
            limit = 50 if not directory else 60
            if len(agents.read_text().splitlines()) > limit:
                errors.append(f"{agents.relative_to(ROOT)}: exceeds {limit} lines; move detail into links")
    required = {config["review_log"], "docs/agents/project.md", "docs/agents/decisions.md",
                "docs/agents/code-intelligence.md", "docs/agents/maintenance.md"}
    for area in config["areas"].values():
        required.update(area["context"])
    files.update(ROOT / name for name in required)
    for path in sorted(files):
        if not path.is_file():
            errors.append(f"Missing context file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text()
        if len(text.encode()) > 16000:
            errors.append(f"{path.relative_to(ROOT)}: exceeds 16 KB context budget")
        for target in local_targets(text):
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            dest = (path.parent / unquote(parts.path)).resolve()
            if not dest.is_relative_to(ROOT):
                errors.append(f"{path.relative_to(ROOT)}: nonportable link {target}")
            elif not dest.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link target {target}")
    return errors, len(files)


def check_coverage(config, changed):
    errors = []
    # Context files are not implementation changes. A reviewed no-change rationale
    # can be recorded in the common review log; humans must assess its adequacy.
    sources = {p for p in changed if Path(p).name not in {"AGENTS.md", "CLAUDE.md"}}
    for name, area in config["areas"].items():
        affected = sorted(p for p in sources if any(matches(p, pattern) for pattern in area["sources"]))
        if not affected:
            continue
        reviewed = changed.intersection([*area["context"], config["review_log"]])
        if not reviewed:
            errors.append(f"{name}: source changed ({', '.join(affected[:4])}); update relevant context "
                          f"or record a reason in {config['review_log']}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Git ref to compare from its merge base with HEAD")
    args = parser.parse_args()
    try:
        config = json.loads((CONTEXT / "source-map.json").read_text())
        errors, count = check_structure(config)
        if args.base:
            errors.extend(check_coverage(config, changed_paths(args.base)))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Context check could not complete: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("Context check failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    print(f"Context check passed: {count} maintained files; "
          + (f"source review coverage against {args.base}." if args.base else "structural checks only."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
