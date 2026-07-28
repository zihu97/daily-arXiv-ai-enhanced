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
