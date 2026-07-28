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

if __name__ == "__main__":
    unittest.main()
