"""
Provider abstraction for generation (spec section 28: "keep model/provider
interfaces configurable"). Everything downstream (generation pipeline,
groundedness checker) talks to `LLMProvider.complete()`, never to a specific
vendor's SDK/HTTP shape directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stopped_reason: str = ""


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> LLMResponse:
        ...
