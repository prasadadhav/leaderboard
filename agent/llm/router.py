"""
Bounded LLM interpreter/router for the LëtzEval BAF assistant.

Purpose
-------
The PyTorch intent classifier handles unambiguous English queries.  This
module is called ONLY as a fallback when:
  - model-name extraction returns zero or >5 ambiguous candidates, OR
  - the intent confidence is low, OR
  - the user typed in a non-English language.

The LLM is given a tightly constrained prompt:
  • The literal user message (plain text, no injected metadata)
  • An allowlist of valid intent names
  • An allowlist of CEFR levels and categories
  • A SHORT list of candidate model names from the DB (never fabricated)

It must return ONLY a JSON object matching a strict schema.  No prose, no
extra keys, no invented model names.  The result is validated server-side
before any action is taken.

Security notes
--------------
• The LLM output is treated as untrusted data; it never executes SQL or
  triggers arbitrary code.
• Model names in the JSON must appear in `allowed_models`; any other name
  is rejected and the fallback is used.
• Intent names must appear in ALLOWED_INTENTS; any other value is rejected.
• A hard token budget (max_tokens=150) prevents runaway output.
• A wall-clock timeout (from LLM config) prevents stuck requests.
• Provider=none: this module always returns None immediately (no LLM call).
"""
import json
import logging
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical intent names the LLM may propose.
ALLOWED_INTENTS = frozenset({
    'explain_leaderboard',
    'explain_methodology',
    'explain_metric',
    'lookup_score',
    'rank_models',
    'compare_models',
    'data_coverage',
    'filter_results',
    'reset_view',
    'out_of_scope',
})

ALLOWED_CEFR   = frozenset({'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'All', None})
ALLOWED_CATS   = frozenset({'Grammar', 'LC', 'RC', 'Vocab', 'Total', 'Macro', 'All', None})

_SCHEMA_EXAMPLE = '''{
  "intent": "lookup_score",
  "models": ["<exact DB model name from the candidate list, or null>"],
  "cefr_level": "B2",
  "category": "Grammar",
  "top_n": null,
  "ascending": false,
  "needs_clarification": false
}'''

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a strict JSON extractor for the LëtzEval multilingual LLM leaderboard assistant.
    Your ONLY job is to parse a user message and return a JSON object — no prose, no explanation.

    Allowed intents: {intents}
    Allowed CEFR levels: A1, A2, B1, B2, C1, C2, All, null
    Allowed categories: Grammar, LC, RC, Vocab, Total, Macro, All, null
    Candidate model names (use ONLY these, never invent): {candidates}

    Return exactly this schema (replace <...> with actual values or null):
    {schema}

    Rules:
    - models: list of 0–3 names from the candidate list only; empty list if uncertain.
    - needs_clarification: true if you cannot determine the model name with confidence.
    - top_n: integer 1–25 or null.
    - ascending: true for "worst/lowest/bottom" queries, false otherwise.
    - If the message is off-topic, set intent="out_of_scope".
    - Output only the JSON object, no markdown fences.
""")


def _build_prompt(
    user_message: str,
    intent_hint: Optional[str],
    candidate_models: list[str],
) -> str:
    intents_str = ', '.join(sorted(ALLOWED_INTENTS))
    candidates_str = json.dumps(candidate_models[:10])  # hard cap to keep prompt short
    system = _SYSTEM_PROMPT.format(
        intents=intents_str,
        candidates=candidates_str,
        schema=_SCHEMA_EXAMPLE,
    )
    if intent_hint:
        system += f'\nHint from classifier: the intent is likely "{intent_hint}".'
    return system


def _validate(parsed: dict, allowed_models: list[str]) -> Optional[dict]:
    """Validate LLM JSON output against the schema; return None if invalid."""
    if not isinstance(parsed, dict):
        return None

    intent = parsed.get('intent')
    if intent not in ALLOWED_INTENTS:
        logger.warning("LLM router: invalid intent %r", intent)
        return None

    cefr = parsed.get('cefr_level')
    if cefr not in ALLOWED_CEFR:
        logger.warning("LLM router: invalid cefr_level %r", cefr)
        return None

    cat = parsed.get('category')
    if cat not in ALLOWED_CATS:
        logger.warning("LLM router: invalid category %r", cat)
        return None

    models = parsed.get('models')
    if models is None:
        models = []
    if not isinstance(models, list):
        logger.warning("LLM router: models is not a list")
        return None

    # Every proposed model name must appear in the DB candidate list (case-insensitive)
    allowed_lower = {m.lower() for m in allowed_models}
    clean_models = []
    for m in models:
        if not isinstance(m, str):
            continue
        if m.lower() in allowed_lower:
            # Return the canonical casing from allowed_models
            for canonical in allowed_models:
                if canonical.lower() == m.lower():
                    clean_models.append(canonical)
                    break
        else:
            logger.warning("LLM router: model name %r not in candidate list — rejected", m)

    top_n = parsed.get('top_n')
    if top_n is not None:
        try:
            top_n = int(top_n)
            if not (1 <= top_n <= 25):
                top_n = 5
        except (TypeError, ValueError):
            top_n = None

    ascending = bool(parsed.get('ascending', False))
    needs_clarification = bool(parsed.get('needs_clarification', False))

    return {
        'intent': intent,
        'models': clean_models,
        'cefr_level': cefr,
        'category': cat,
        'top_n': top_n,
        'ascending': ascending,
        'needs_clarification': needs_clarification,
    }


def classify_with_llm(
    llm,
    user_message: str,
    intent_hint: Optional[str],
    candidate_models: list[str],
    all_models: list[str],
) -> Optional[dict]:
    """Call the LLM interpreter and return a validated route dict, or None.

    Args:
        llm: BAF LLM instance (LLMOpenAI or LLMOllama), or None.
        user_message: the raw user input (plain text).
        intent_hint: intent name from the PyTorch classifier, if available.
        candidate_models: short list of plausible DB model names (≤10).
        all_models: full DB model name list for validation.

    Returns:
        dict with keys intent/models/cefr_level/category/top_n/ascending/
        needs_clarification, or None if LLM unavailable / output invalid.
    """
    if llm is None:
        return None

    system_prompt = _build_prompt(user_message, intent_hint, candidate_models)

    try:
        raw = llm.predict(
            message=user_message,
            parameters={'max_tokens': 150, 'temperature': 0},
            system_message=system_prompt,
        )
    except Exception as exc:
        logger.warning("LLM router: prediction failed (%s); using deterministic fallback.", exc)
        return None

    if not raw or not isinstance(raw, str):
        return None

    # Strip markdown fences if the model added them despite instructions
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-z]*\n?', '', text, flags=re.IGNORECASE)
        text = text.rstrip('`').strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM router: could not parse JSON from response: %r", text[:200])
        return None

    return _validate(parsed, all_models)


# Import re here to avoid importing at module level unnecessarily
import re
