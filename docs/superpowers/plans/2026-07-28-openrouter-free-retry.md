# OpenRouter Free Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry an invalid `openrouter/free` summary up to 30 times per paper while continuing to publish the existing fallback fields after all attempts fail.

**Architecture:** Extract strict response parsing and retry orchestration into small functions in `ai/enhance.py`. The production chain requests a JSON object, while deterministic standard-library unit tests drive fake responses and inject a no-op sleeper so no test calls OpenRouter or waits.

**Tech Stack:** Python 3.12, LangChain `ChatOpenAI`, standard-library `unittest` and `unittest.mock`, `uv`.

---

### Task 1: Define strict AI-response acceptance

**Files:**
- Create: `tests/test_enhance.py`
- Modify: `ai/enhance.py:33-40`

- [ ] **Step 1: Write the failing parser tests**

Create `tests/test_enhance.py` with:

```python
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

import enhance


VALID_AI = {
    "tldr": "summary",
    "motivation": "motivation",
    "method": "method",
    "result": "result",
    "conclusion": "conclusion",
}


class ParseAiResponseTests(unittest.TestCase):
    def test_accepts_json_inside_markdown_fence(self):
        response = f"```json\n{json.dumps(VALID_AI)}\n```"

        self.assertEqual(enhance.parse_ai_response(response), VALID_AI)

    def test_rejects_non_json_response(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            enhance.parse_ai_response("User Safety: safe")

    def test_rejects_missing_required_field(self):
        incomplete = dict(VALID_AI)
        del incomplete["conclusion"]

        with self.assertRaisesRegex(ValueError, "conclusion"):
            enhance.parse_ai_response(json.dumps(incomplete))

    def test_rejects_empty_required_field(self):
        incomplete = dict(VALID_AI, method="   ")

        with self.assertRaisesRegex(ValueError, "method"):
            enhance.parse_ai_response(json.dumps(incomplete))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the parser tests and verify RED**

Run:

```bash
uv run python -m unittest tests.test_enhance.ParseAiResponseTests -v
```

Expected: four errors containing `module 'enhance' has no attribute 'parse_ai_response'`.

- [ ] **Step 3: Implement the minimal strict parser**

Add module constants and the parser above `process_single_item` in `ai/enhance.py`:

```python
REQUIRED_AI_FIELDS = (
    "tldr",
    "motivation",
    "method",
    "result",
    "conclusion",
)

DEFAULT_AI_FIELDS = {
    "tldr": "Summary generation failed",
    "motivation": "Motivation analysis unavailable",
    "method": "Method extraction failed",
    "result": "Result analysis unavailable",
    "conclusion": "Conclusion extraction failed",
}


def parse_ai_response(response) -> Dict[str, str]:
    response_text = str(response)
    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}")
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        raise ValueError("Response does not contain a JSON object")

    try:
        result = json.loads(response_text[start_idx:end_idx + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response contains invalid JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError("Response JSON must be an object")

    for field in REQUIRED_AI_FIELDS:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Required field '{field}' is missing or empty")

    return result
```

- [ ] **Step 4: Run the parser tests and verify GREEN**

Run:

```bash
uv run python -m unittest tests.test_enhance.ParseAiResponseTests -v
```

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Commit strict parsing**

```bash
git add ai/enhance.py tests/test_enhance.py
git commit -m "test: define valid AI summary responses"
```

### Task 2: Retry each paper through the random free router

**Files:**
- Modify: `tests/test_enhance.py`
- Modify: `ai/enhance.py:40-191`

- [ ] **Step 1: Write the failing retry tests**

Add below `VALID_AI` in `tests/test_enhance.py`:

```python
class SequenceChain:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
```

Add the following test class:

```python
class GenerateAiFieldsTests(unittest.TestCase):
    def setUp(self):
        self.valid_response = json.dumps(VALID_AI)
        self.inputs = {"language": "Chinese", "content": "paper abstract"}

    def test_first_valid_response_does_not_retry(self):
        chain = SequenceChain([self.valid_response])
        sleeps = []

        result = enhance.generate_ai_fields(
            chain, self.inputs, "paper-1", sleep_fn=sleeps.append
        )

        self.assertEqual(result, VALID_AI)
        self.assertEqual(chain.calls, 1)
        self.assertEqual(sleeps, [])

    def test_non_json_response_retries_then_succeeds(self):
        chain = SequenceChain(["User Safety: safe", self.valid_response])
        sleeps = []

        result = enhance.generate_ai_fields(
            chain, self.inputs, "paper-2", sleep_fn=sleeps.append
        )

        self.assertEqual(result, VALID_AI)
        self.assertEqual(chain.calls, 2)
        self.assertEqual(sleeps, [1])

    def test_api_exception_retries_then_succeeds(self):
        chain = SequenceChain([RuntimeError("rate limited"), self.valid_response])

        result = enhance.generate_ai_fields(
            chain, self.inputs, "paper-3", sleep_fn=lambda _: None
        )

        self.assertEqual(result, VALID_AI)
        self.assertEqual(chain.calls, 2)

    def test_incomplete_response_retries_then_succeeds(self):
        incomplete = json.dumps(dict(VALID_AI, method=""))
        chain = SequenceChain([incomplete, self.valid_response])

        result = enhance.generate_ai_fields(
            chain, self.inputs, "paper-4", sleep_fn=lambda _: None
        )

        self.assertEqual(result, VALID_AI)
        self.assertEqual(chain.calls, 2)

    def test_exhausts_initial_attempt_plus_thirty_retries(self):
        chain = SequenceChain(["..."] * 31)
        sleeps = []

        result = enhance.generate_ai_fields(
            chain, self.inputs, "paper-5", sleep_fn=sleeps.append
        )

        self.assertEqual(result, enhance.DEFAULT_AI_FIELDS)
        self.assertIsNot(result, enhance.DEFAULT_AI_FIELDS)
        self.assertEqual(chain.calls, 31)
        self.assertEqual(len(sleeps), 30)
        self.assertEqual(sleeps[:3], [1, 2, 2])
        self.assertTrue(all(delay == 2 for delay in sleeps[2:]))
```

- [ ] **Step 2: Run retry tests and verify RED**

Run:

```bash
uv run python -m unittest tests.test_enhance.GenerateAiFieldsTests -v
```

Expected: five errors containing `module 'enhance' has no attribute 'generate_ai_fields'`.

- [ ] **Step 3: Implement retry orchestration**

Add these constants and function below `parse_ai_response` in `ai/enhance.py`:

```python
MAX_AI_RETRIES = 30
MAX_RETRY_DELAY_SECONDS = 2


def generate_ai_fields(
    chain,
    inputs: Dict[str, str],
    paper_id: str,
    max_retries: int = MAX_AI_RETRIES,
    sleep_fn=time.sleep,
) -> Dict[str, str]:
    total_attempts = max_retries + 1
    last_error = None

    for attempt in range(1, total_attempts + 1):
        try:
            response = chain.invoke(inputs)
            result = parse_ai_response(response)
            print(
                f"Summary generated for {paper_id} on attempt "
                f"{attempt}/{total_attempts}",
                file=sys.stderr,
            )
            return result
        except Exception as exc:
            last_error = exc
            print(
                f"Summary attempt {attempt}/{total_attempts} failed for "
                f"{paper_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

        if attempt < total_attempts:
            retry_number = attempt
            delay = min(2 ** (retry_number - 1), MAX_RETRY_DELAY_SECONDS)
            sleep_fn(delay)

    print(
        f"All {total_attempts} summary attempts failed for {paper_id}; "
        f"using default values: {last_error}",
        file=sys.stderr,
    )
    return DEFAULT_AI_FIELDS.copy()
```

Replace the existing parsing loop and locally defined fallback dictionary inside `process_single_item` with:

```python
    item["AI"] = generate_ai_fields(
        chain,
        {
            "language": language,
            "content": item["summary"],
        },
        item.get("id", "unknown"),
    )
```

Change its signature from:

```python
def process_single_item(chains_to_try, item: Dict, language: str) -> Dict:
```

to:

```python
def process_single_item(chain, item: Dict, language: str) -> Dict:
```

In `process_all_items`, change the executor submission to pass `chain` rather than `[chain]`:

```python
executor.submit(process_single_item, chain, item, language): idx
```

- [ ] **Step 4: Run all retry and parser tests and verify GREEN**

Run:

```bash
uv run python -m unittest tests.test_enhance -v
```

Expected: `Ran 9 tests` and `OK`.

- [ ] **Step 5: Commit per-paper retry handling**

```bash
git add ai/enhance.py tests/test_enhance.py
git commit -m "fix: retry invalid OpenRouter summaries"
```

### Task 3: Require JSON-object-capable free routes

**Files:**
- Modify: `tests/test_enhance.py`
- Modify: `ai/enhance.py:193-220`

- [ ] **Step 1: Write the failing model-configuration test**

Add this import to `tests/test_enhance.py`:

```python
from unittest.mock import Mock, patch
```

Add this test class:

```python
class BuildLlmTests(unittest.TestCase):
    @patch.object(enhance, "ChatOpenAI")
    def test_requests_json_object_output_and_disables_hidden_retries(self, chat_openai):
        base_llm = Mock()
        bound_llm = Mock()
        chat_openai.return_value = base_llm
        base_llm.bind.return_value = bound_llm

        result = enhance.build_llm("openrouter/free")

        chat_openai.assert_called_once_with(
            model="openrouter/free",
            max_retries=0,
        )
        base_llm.bind.assert_called_once_with(
            response_format={"type": "json_object"}
        )
        self.assertIs(result, bound_llm)
```

- [ ] **Step 2: Run the model-configuration test and verify RED**

Run:

```bash
uv run python -m unittest tests.test_enhance.BuildLlmTests -v
```

Expected: one error containing `module 'enhance' has no attribute 'build_llm'`.

- [ ] **Step 3: Implement explicit JSON configuration**

Add above `process_all_items` in `ai/enhance.py`:

```python
def build_llm(model_name: str):
    llm = ChatOpenAI(model=model_name, max_retries=0)
    return llm.bind(response_format={"type": "json_object"})
```

Replace this line in `process_all_items`:

```python
llm = ChatOpenAI(model=model_name)
```

with:

```python
llm = build_llm(model_name)
```

- [ ] **Step 4: Run the full unit-test suite**

Run:

```bash
uv run python -m unittest discover -s tests -v
```

Expected: `Ran 10 tests` and `OK`.

- [ ] **Step 5: Run syntax and diff checks**

Run:

```bash
uv run python -m py_compile ai/enhance.py tests/test_enhance.py
git diff --check
```

Expected: both commands exit 0 with no output.

- [ ] **Step 6: Commit JSON routing constraints**

```bash
git add ai/enhance.py tests/test_enhance.py
git commit -m "fix: require JSON output from free model routes"
```

### Task 4: Final review and remote delivery

**Files:**
- Verify: `ai/enhance.py`
- Verify: `tests/test_enhance.py`
- Verify: `docs/superpowers/specs/2026-07-28-openrouter-free-retry-design.md`

- [ ] **Step 1: Re-run all verification from a clean process**

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m py_compile ai/enhance.py tests/test_enhance.py
git diff --check
git status --short --branch
```

Expected: 10 tests pass, compilation and diff checks exit 0, and the branch contains only the intended committed changes.

- [ ] **Step 2: Inspect the complete change against the refreshed remote base**

```bash
git fetch origin main
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- ai/enhance.py tests/test_enhance.py docs/superpowers
```

Expected: the diff contains the design, plan, retry implementation, and tests only; no generated paper data or workflow changes appear.

- [ ] **Step 3: Push after confirming the remote tip has not diverged**

```bash
test "$(git rev-parse origin/main)" = "$(git ls-remote origin refs/heads/main | cut -f1)"
git push origin HEAD:main
```

Expected: the tip check exits 0 and the push updates `main` without force.

- [ ] **Step 4: Verify the pushed SHA and variable**

```bash
test "$(git rev-parse HEAD)" = "$(git ls-remote origin refs/heads/main | cut -f1)"
test "$(gh variable get MODEL_NAME_BAK --repo zihu97/daily-arXiv-ai-enhanced)" = "openrouter/free"
```

Expected: both checks exit 0.
