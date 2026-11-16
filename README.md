# Veriflow — PLE agent harness

An autonomous engineering agent for Google's Agent Harness on
Physical-Engineering's Last Exam (Kaggle, $400K). It solves real structural /
civil / electrical / mechanical tasks in an isolated container and writes
graded deliverables (calculation packages, design reviews, drawing
revisions, memos).

## Idea

Frontier models already score 0.40–0.48 on PLE with their native harnesses;
the harness is where the edge is. The failure modes the Stage 1 rubric
names — misreading a drawing callout, picking the wrong code clause, unit
errors, stopping early — are all things a harness can guard against
deterministically. Veriflow does four things the minimal starter does not:

1. **Phases with a clock.** recon → solve → verify → deliver. The verify
   phase is forced by the clock even if the model never declares
   deliverables.
2. **Independent verification.** After deliverables are written, the agent
   re-reads them and recomputes every key number with a fresh script,
   checks units (SI base conversion) and code clauses, and fixes
   discrepancies before finishing.
3. **Deterministic deliverable check.** Paths the model declares are checked
   to exist and be non-empty by the harness itself; a missing or empty file
   bounces the model back to work instead of ending the run.
4. **MCP tools.** When a task exposes MCP servers (documents, spreadsheets,
   CAD, email), they are exposed as tools; the kit's smoke/MCP checks
   exercise them.

The model loop is provider-agnostic OpenAI Chat Completions, so
`model_api` can switch between the eight Google-sponsored models without
code changes. Default target: `google-cloud/claude-opus-5-5` (best
published PLE score, 0.483 with claude-code). Cost is tracked per call
(`usage` frames + a run-stats footer in the final message) for the
score-vs-cost plot the writeup needs.

## Layout

```
submission.yaml   entry name, model, install/serve commands
serve.py          the agent (phase machine + model loop + tools)
taskbridge.py     kit protocol plumbing (copied from google-research/ple)
prompts/          system.md (working rules), verify.md (verify phase)
requirements.txt  openai, websockets
tests/            mock evaluator + mock model + test runner (no keys, no Docker)
```

## Local test (no API key, no Docker)

```bash
pip install websockets openai
python -m unittest discover -s tests -v
```

The tests run `serve.py` against a mock evaluator that speaks the kit
protocol and a mock OpenAI-compatible model endpoint, covering the smoke
task (write hello.txt + done), a calculation task with a verification
round, an MCP task, and a cancel/timeout path.

## Full local check (needs Docker + uv + a model)

```bash
# from a clone of google-research/ple, with this entry in its own repo
uv run python -m kit check ../my-harness --env-file local.env
uv run python eval/run.py ../my-harness --env-file local.env  # 29 dev tasks
```

The dev tasks live in the gated Hugging Face dataset
`google/physical-engineerings-last-exam` (accept terms + `HF_TOKEN`).

## Status

Scaffold + protocol tests done. Not yet run against real dev tasks
(needs the HF dataset + a model key) and not yet through `kit check`
(needs Docker). See `../NOTES.md`.
