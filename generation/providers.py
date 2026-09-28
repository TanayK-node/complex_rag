"""
Two LLMProvider implementations:

- AnthropicProvider: a real implementation against api.anthropic.com. This is
  what production use is meant to run on, but it is NOT exercised by this
  repo's tests — there is no API key configured in this sandbox, and per the
  spec's own instruction ("do not claim something works unless it has
  actually been tested"), I'm not going to claim this path is verified when
  it hasn't been run. Wire in RAG_LLM_API_KEY (or ANTHROPIC_API_KEY) and run
  tests/test_generation.py with --live to actually exercise it.

- DeterministicMockProvider: a rule-based stand-in used by the test suite so
  the *pipeline* (prompting, citation checking, groundedness verification,
  retry logic) can be tested end-to-end without any API access. It is NOT a
  claim that this produces good answers — it is scaffolding to prove the
  plumbing around the LLM call is correct, so that swapping in a real model
  later is a one-line config change rather than a debugging exercise.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.request
import urllib.error

from config.settings import settings
from generation.llm_provider import LLMProvider, LLMResponse


_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
_PER_DAY_QUOTA_RE = re.compile(r'PerDay', re.IGNORECASE)


def _post_with_retry(req: urllib.request.Request, max_retries: int = 6, max_sleep: float = 65.0) -> dict:
    """Shared retry-with-backoff for both real providers. On a 429, prefer the
    server's own suggested wait time (Gemini's error body includes a precise
    "retryDelay", e.g. "10s") over a blind guess — free-tier quotas are counted
    per-minute, so a fixed exponential backoff (1s, 2s, 4s, 8s = 15s total) can
    still land inside the same rate-limit window and fail again. Falls back to
    exponential backoff when no retryDelay is present (e.g. a 503) or on
    Anthropic, whose errors don't include this field.

    A per-DAY quota (Gemini's quotaId contains "PerDay") is a fundamentally
    different failure than a per-minute one: no amount of waiting-and-retrying
    within this run will free it up (confirmed live: Gemini 3.8 Flash's free
    tier is 20 requests/day total, and a "retryDelay":"59s" on that error does
    NOT mean 59 seconds gets you a fresh quota — the daily cap resets at
    midnight Pacific, not in a minute). Retrying anyway just burns wall-clock
    time before failing the same way. Fail fast with an actionable message
    instead."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as e:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, max_sleep))
                last_error = RuntimeError(
                    f"API request timed out after 60 seconds (attempt {attempt + 1})"
                )
                continue
            raise RuntimeError(
                "API request timed out after repeated attempts. "
                "The evaluator will fall back to the mock provider."
            ) from e
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            if e.code == 429 and _PER_DAY_QUOTA_RE.search(body):
                raise RuntimeError(
                    "Gemini free-tier DAILY quota exhausted for this model (resets at "
                    "midnight Pacific time). Retrying won't help within this run. Fix: "
                    "switch to a Flash-Lite model, which gets a much higher daily free "
                    "quota on the same tier — set GEMINI_MODEL=gemini-3.5-flash-lite (or "
                    "your current Flash-Lite of choice; check "
                    "https://ai.google.dev/gemini-api/docs/rate-limits for today's exact "
                    f"figures). Original error: {body}"
                ) from e
            if e.code in (429, 503) and attempt < max_retries:
                m = _RETRY_DELAY_RE.search(body)
                wait = min(float(m.group(1)) + 1.0, max_sleep) if m else min(2 ** attempt, max_sleep)
                time.sleep(wait)
                last_error = RuntimeError(f"HTTP {e.code}: {body}")
                continue
            raise RuntimeError(f"API error {e.code}: {body}") from e
    raise last_error  # pragma: no cover — loop always returns or raises above


class AnthropicProvider(LLMProvider):
    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "No Anthropic API key configured. Set RAG_LLM_API_KEY or ANTHROPIC_API_KEY."
            )

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> LLMResponse:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        req = urllib.request.Request(
            self.API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        data = _post_with_retry(req)

        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=data.get("model", self.model),
            stopped_reason=data.get("stop_reason", ""),
        )


class GeminiProvider(LLMProvider):
    """
    Google Gemini, via the free-tier-eligible generateContent REST endpoint.
    Uses the stdlib only (no google-generativeai SDK dependency), same pattern
    as AnthropicProvider. NOT live-tested in this sandbox — this environment's
    network egress allowlist blocks generativelanguage.googleapis.com (confirmed:
    the request reaches the proxy and gets an explicit "Host not in allowlist"
    rejection, not a DNS/connection failure). Test with a real key in your own
    environment before relying on it. See the README's "Choosing an LLM
    provider" section for why you'd pick this over Anthropic (free tier) and
    its main caveat (rate limits on that free tier).
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.gemini_model
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            raise RuntimeError("No Gemini API key configured. Set GEMINI_API_KEY.")
        # Proactive pacing: free-tier Gemini quotas are ~5 requests/minute per
        # model (confirmed from a live 429 body: "limit: 5 ... FreeTier"), so a
        # batch eval run (one call per question, back-to-back) hits the wall
        # almost immediately if it doesn't space calls out itself — waiting to
        # react to a 429 and hoping the retryDelay is short isn't enough on its
        # own once several questions in a row all land in the same minute.
        # GEMINI_MIN_INTERVAL_SECONDS lets you tune this for a paid tier's
        # higher quota; 13s keeps you at ~4.6 req/min, just under the default
        # free-tier limit of 5.
        self.min_interval = settings.gemini_min_interval_seconds
        self._last_call_ts = 0.0

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> LLMResponse:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            data = _post_with_retry(req)
        finally:
            self._last_call_ts = time.time()

        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            model=self.model,
            stopped_reason=candidates[0].get("finishReason", "") if candidates else "",
        )


class DeterministicMockProvider(LLMProvider):
    """
    Rule-based "generation": pulls sentences from the user prompt's context
    section whose tokens overlap the question, attaches the citation that
    came with whichever context block they were drawn from, and explicitly
    says so when nothing in the context looks relevant. This is intentionally
    dumb — it exists only to give the rest of Phase 4 something deterministic
    to test against without network access.
    """

    _TOKEN_RE = re.compile(r"[a-zA-Z0-9$%]+")
    _STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "or",
                  "for", "on", "what", "does", "do", "under", "this", "that", "when"}

    def _tokens(self, text: str) -> set[str]:
        return {t.lower() for t in self._TOKEN_RE.findall(text)} - self._STOPWORDS

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> LLMResponse:
        q_match = re.search(r"Question:\s*(.+)", user)
        question = q_match.group(1).strip() if q_match else ""
        q_tokens = self._tokens(question)

        # isolate just the context portion of the prompt (before the "---\n\nQuestion:"
        # delimiter) so parsing below can't accidentally swallow the question/
        # instruction text that follows it as if it were retrieved content
        context_portion = user.split("\n\n---\n\nQuestion:")[0]

        # parse "[Document: ... | Section X ...]\n<text>" blocks out of the
        # context section of the prompt, keeping each block's citation
        blocks = re.split(r"\n\[Document:", context_portion)
        best_sentence, best_citation, best_overlap = None, None, 0

        for raw in blocks[1:]:
            header, _, body = ("[Document:" + raw).partition("]")
            sec_match = re.search(r"Section\s+([\d.]+)", header)
            doc_match = re.search(r"Document:\s*([^|]+)", header)
            citation = f"[Source: {doc_match.group(1).strip() if doc_match else 'Unknown'}" \
                       f"{', Section ' + sec_match.group(1) if sec_match else ''}]"
            for sentence in re.split(r"(?<=[.;])\s+", body.strip()):
                overlap = len(q_tokens & self._tokens(sentence))
                if overlap > best_overlap:
                    best_overlap, best_sentence, best_citation = overlap, sentence.strip(), citation

        if not best_sentence or best_overlap == 0:
            text = "INSUFFICIENT EVIDENCE: The provided context does not contain information that answers this question."
        else:
            text = f"{best_sentence} {best_citation}"

        return LLMResponse(text=text, input_tokens=len(user.split()), output_tokens=len(text.split()),
                            model="deterministic-mock")


def get_llm_provider() -> LLMProvider:
    """Single place that decides which provider is active: tries Anthropic first
    (if ANTHROPIC_API_KEY/RAG_LLM_API_KEY is set), then Gemini (if GEMINI_API_KEY
    is set), then falls back to the mock provider so the pipeline still runs
    end-to-end (with an honest, visibly-lower-quality answer) instead of
    crashing when no credentials are configured yet."""
    try:
        return AnthropicProvider()
    except RuntimeError:
        pass
    try:
        return GeminiProvider()
    except RuntimeError:
        pass
    return DeterministicMockProvider()


def describe_provider(llm: LLMProvider) -> str:
    """Human-readable label for CLI output — one place so ask.py/evaluate.py
    don't each hardcode their own isinstance checks."""
    if isinstance(llm, AnthropicProvider):
        return f"Anthropic ({llm.model}, live)"
    if isinstance(llm, GeminiProvider):
        return f"Gemini ({llm.model}, live)"
    return "deterministic mock (no API key set — see README 'Choosing an LLM provider')"