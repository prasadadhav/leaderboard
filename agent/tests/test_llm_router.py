"""
Tests for agent/llm/router.py — all LLM calls are mocked.

Run from leaderboard/agent/:
    python -m pytest tests/test_llm_router.py -v
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock

from llm.router import classify_with_llm, ALLOWED_INTENTS, _validate

_ALL_MODELS = ["GPT-6 Astra", "Llama 3.1 - 70B", "Qwen 2.5 - 7B", "Gemma 3 - 12B"]


def _llm_returning(text: str):
    """Return a mock LLM whose predict() returns `text`."""
    llm = MagicMock()
    llm.predict.return_value = text
    return llm


# ── classify_with_llm ─────────────────────────────────────────────────────────

class TestClassifyWithLlm:
    def test_returns_none_when_llm_is_none(self):
        result = classify_with_llm(None, "some message", None, [], _ALL_MODELS)
        assert result is None

    def test_happy_path(self):
        payload = json.dumps({
            "intent": "lookup_score",
            "models": ["GPT-6 Astra"],
            "cefr_level": "B2",
            "category": "Grammar",
            "top_n": None,
            "ascending": False,
            "needs_clarification": False,
        })
        llm = _llm_returning(payload)
        result = classify_with_llm(llm, "GPT-6 Astra B2 grammar", "lookup_score", _ALL_MODELS, _ALL_MODELS)
        assert result is not None
        assert result["intent"] == "lookup_score"
        assert result["models"] == ["GPT-6 Astra"]
        assert result["cefr_level"] == "B2"
        assert result["category"] == "Grammar"

    def test_strips_markdown_fences(self):
        payload = "```json\n" + json.dumps({
            "intent": "rank_models",
            "models": [],
            "cefr_level": "C1",
            "category": "Total",
            "top_n": 5,
            "ascending": False,
            "needs_clarification": False,
        }) + "\n```"
        llm = _llm_returning(payload)
        result = classify_with_llm(llm, "top 5 at C1", "rank_models", [], _ALL_MODELS)
        assert result is not None
        assert result["intent"] == "rank_models"

    def test_returns_none_on_invalid_json(self):
        llm = _llm_returning("not valid json at all")
        result = classify_with_llm(llm, "anything", None, [], _ALL_MODELS)
        assert result is None

    def test_returns_none_when_predict_raises(self):
        llm = MagicMock()
        llm.predict.side_effect = RuntimeError("network error")
        result = classify_with_llm(llm, "anything", None, [], _ALL_MODELS)
        assert result is None


# ── _validate ─────────────────────────────────────────────────────────────────

class TestValidate:
    def _base(self, **kwargs):
        base = {
            "intent": "lookup_score",
            "models": [],
            "cefr_level": None,
            "category": None,
            "top_n": None,
            "ascending": False,
            "needs_clarification": False,
        }
        base.update(kwargs)
        return base

    def test_valid_returns_dict(self):
        result = _validate(self._base(), _ALL_MODELS)
        assert result is not None
        assert result["intent"] == "lookup_score"

    def test_invalid_intent_rejected(self):
        result = _validate(self._base(intent="drop_table"), _ALL_MODELS)
        assert result is None

    def test_invalid_cefr_rejected(self):
        result = _validate(self._base(cefr_level="D1"), _ALL_MODELS)
        assert result is None

    def test_invalid_category_rejected(self):
        result = _validate(self._base(category="Pronunciation"), _ALL_MODELS)
        assert result is None

    def test_invented_model_name_rejected(self):
        result = _validate(self._base(models=["FakeModel - 99B"]), _ALL_MODELS)
        assert result is not None
        assert "FakeModel - 99B" not in result["models"]
        assert result["models"] == []

    def test_model_name_normalised_to_canonical_case(self):
        result = _validate(self._base(models=["gpt-6 astra"]), _ALL_MODELS)
        assert result is not None
        assert result["models"] == ["GPT-6 Astra"]

    def test_top_n_clamped(self):
        result = _validate(self._base(top_n=999), _ALL_MODELS)
        assert result is not None
        assert result["top_n"] == 5  # out-of-range → default 5

    def test_top_n_valid(self):
        result = _validate(self._base(top_n=10), _ALL_MODELS)
        assert result is not None
        assert result["top_n"] == 10

    def test_all_intents_allowed(self):
        for intent in ALLOWED_INTENTS:
            result = _validate(self._base(intent=intent), _ALL_MODELS)
            assert result is not None, f"Intent {intent!r} was unexpectedly rejected"

    def test_not_a_dict_returns_none(self):
        assert _validate(["lookup_score"], _ALL_MODELS) is None
        assert _validate(None, _ALL_MODELS) is None
