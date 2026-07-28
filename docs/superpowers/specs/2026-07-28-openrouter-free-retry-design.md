# OpenRouter Free Retry Design

## Goal

Improve per-paper summary reliability while keeping `MODEL_NAME_BAK=openrouter/free` and preserving the current behavior of publishing papers whose summaries ultimately fail.

## Scope

The change is limited to AI summary generation in `ai/enhance.py` and its automated tests. It does not select or pin a concrete fallback model, alter crawling or Markdown rendering, change repository variables, or regenerate historical data.

## Approach

Each paper is first sent to `openrouter/free`. If the request fails or its response cannot be accepted, the same request is retried through `openrouter/free` up to 30 additional times. Because OpenRouter chooses a free backing model for each request, every retry gives the paper another opportunity to reach a suitable instruction model without introducing a concrete model that may later be removed.

The model request will include `response_format={"type": "json_object"}`. This communicates the required response shape to OpenRouter and lets its router prefer compatible backing models before application-level validation runs.

## Acceptance and Retry Rules

A response is accepted only when it can be parsed into a JSON object containing all five required fields:

- `tldr`
- `motivation`
- `method`
- `result`
- `conclusion`

Every required field must contain a non-empty string. An API exception, a non-JSON response such as `User Safety: safe` or `...`, malformed JSON, a non-object JSON value, a missing field, or an empty field consumes one attempt and triggers another call.

There is one initial call plus at most 30 retries, for a maximum of 31 calls per paper. A successful response stops the loop immediately.

Retries use capped backoff. The delay grows after consecutive failures but stops growing at a small upper bound so a single paper cannot add unbounded sleep time. Tests inject a no-op sleep function and do not wait in real time.

## Exhausted Retries

If all 31 calls fail, processing continues. The paper receives the existing fallback values, including `Summary generation failed`, and remains eligible for Markdown conversion and publication. Exhaustion does not make the overall process or GitHub Actions job fail.

Logs identify the paper, attempt number, failure category, and final exhaustion. They must not include API keys or other secrets.

## Code Boundaries

Response parsing and validation will be separated from retry orchestration so both behaviors can be tested without making network calls. `process_single_item` will retain responsibility for sensitive-content checks and attaching either validated AI data or the existing fallback fields to the paper.

No unrelated cleanup or workflow policy changes are included.

## Tests

Automated tests will cover:

1. A valid first response succeeds without retrying.
2. A non-JSON response retries and then succeeds.
3. An API exception retries and then succeeds.
4. A response with a missing or empty required field retries and then succeeds.
5. Thirty retries after the initial failure are exhausted, exactly 31 calls are made, and existing fallback fields are returned without raising.
6. The model chain is configured to request JSON-object output.

All retry tests use deterministic fake chains and a no-op sleeper; they do not call OpenRouter.
