#!/usr/bin/env python3
"""Build a local-run variant of the veriflow-ple harness.

Copies the entry (this repo) to a target directory, swaps `model_api` to a
free Gemini model via `openai-compatible` + `model_base_url`, caps the
turn/output budgets so a free-tier dev run stays bounded, then makes the
target a git repository of its own (committed) so `kit check`'s files stage
sees a clean entry root.

Usage:
  python3 ci/make-local-variant.py <src-dir> <out-dir> [--max-turns N]

The real submission.yaml (google-cloud/claude-opus-5-5) is untouched; only
the copy is patched. No keys are written anywhere: the model key comes from
the environment / --env-file at run time.
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys

EXCLUDE = {".git", "__pycache__", ".venv", "runs", ".github", ".pytest_cache"}

MODEL_API = "openai-compatible/gemini-3-flash-preview"
MODEL_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--max-turns", default="60")
    ap.add_argument("--max-output-tokens", default="8192")
    ap.add_argument("--verify-sec", default="900")
    ap.add_argument("--wrap-up-sec", default="300")
    a = ap.parse_args()

    src = pathlib.Path(a.src).resolve()
    out = pathlib.Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(
        src,
        out,
        ignore=shutil.ignore_patterns(*EXCLUDE, "*.pyc"),
    )

    sub = out / "submission.yaml"
    text = sub.read_text(encoding="utf-8")
    # Swap the model: google-cloud (eval servers only) -> free Gemini model.
    new, n = re.subn(
        r"^model_api:.*$",
        f"model_api: {MODEL_API}",
        text,
        count=1,
        flags=re.M,
    )
    assert n == 1, "model_api line not found"
    text = new
    # model_base_url goes right after model_api (top-level key).
    text = text.replace(
        f"model_api: {MODEL_API}\n",
        f"model_api: {MODEL_API}\nmodel_base_url: {MODEL_BASE_URL}\n",
        1,
    )
    # Bound the run so a free-tier dev subset stays cheap and fast.
    for key, val in (
        ("MAX_TURNS", a.max_turns),
        ("VERIFY_SEC", a.verify_sec),
        ("WRAP_UP_SEC", a.wrap_up_sec),
    ):
        new, n = re.subn(
            rf'^  {key}: "[^"]*"$',
            f'  {key}: "{val}"',
            text,
            count=1,
            flags=re.M,
        )
        assert n == 1, f"env {key} not found"
        text = new
    text = text.replace(
        '  WRAP_UP_SEC: "%s"\n' % a.wrap_up_sec,
        '  WRAP_UP_SEC: "%s"\n  MAX_OUTPUT_TOKENS: "%s"\n'
        % (a.wrap_up_sec, a.max_output_tokens),
        1,
    )
    sub.write_text(text, encoding="utf-8")
    print("patched submission.yaml:")
    print(sub.read_text(encoding="utf-8"))

    # Make the variant a repository of its own, everything committed.
    env = dict(**__import__("os").environ)
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "ci@veriflow-ple.local"],
        ["git", "config", "user.name", "veriflow-ple CI"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "local-run variant (gemini-3-flash-preview via openai-compatible)"],
    ):
        subprocess.run(cmd, cwd=out, check=True, env=env)
    print("variant ready at", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
