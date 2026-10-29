"""
Entity extraction for CEFR level, task category, licensing, top-N, and model names.
Used inside BAF state bodies; standard library only (no external deps).

resolve_model_name() is a two-pass deterministic resolver:
  Pass 1 – exact casefold/normalized substring match against the DB name list.
  Pass 2 – similarity scoring (trigram Jaccard) with a minimum threshold and a
            specificity margin check to avoid silently choosing the wrong model.

Version/size tokens (e.g. "0.5B", "1.5B", "70B", "7B", "3.1", "2.5") are
preserved as meaningful discriminators so that Qwen 2 vs Qwen 2.5 and
0.5B vs 1.5B are never silently equated.
"""
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

CEFR_LEVELS = ('A1', 'A2', 'B1', 'B2', 'C1', 'C2')

_CATEGORY_ALIASES: dict[str, str] = {
    'grammar': 'Grammar',
    'grammer': 'Grammar',
    'reading comprehension': 'RC',
    'reading': 'RC',
    ' rc ': 'RC',
    'rc ': 'RC',
    ' rc': 'RC',
    'listening comprehension': 'LC',
    'listening': 'LC',
    ' lc ': 'LC',
    'lc ': 'LC',
    ' lc': 'LC',
    'vocabulary': 'Vocab',
    'vocab': 'Vocab',
    'words': 'Vocab',
    'total': 'Total',
    'overall': 'Total',
}

_LICENSING_ALIASES: dict[str, str] = {
    'open-source': 'Open_Source',
    'open source': 'Open_Source',
    'open': 'Open_Source',
    'free': 'Open_Source',
    'openly licensed': 'Open_Source',
    'proprietary': 'Proprietary',
    'closed-source': 'Proprietary',
    'closed source': 'Proprietary',
    'closed': 'Proprietary',
    'commercial': 'Proprietary',
}

# Similarity thresholds for fuzzy matching.
# ACCEPT: minimum score to consider a candidate at all.
# MARGIN: winner must beat the second-best by at least this margin to be
#         returned as a single unambiguous match.
_FUZZY_ACCEPT = 0.40
_FUZZY_MARGIN = 0.12


def extract_cefr_level(text: str) -> Optional[str]:
    upper = text.upper()
    for lvl in CEFR_LEVELS:
        if re.search(r'\b' + lvl + r'\b', upper):
            return lvl
    _spelled = {
        'beginner': 'A1', 'elementary': 'A2',
        'intermediate': 'B1', 'upper intermediate': 'B2',
        'advanced': 'C1', 'proficient': 'C2',
    }
    lower = text.lower()
    for word, lvl in _spelled.items():
        if word in lower:
            return lvl
    return None


def extract_category(text: str) -> Optional[str]:
    lower = ' ' + text.lower() + ' '
    for alias in sorted(_CATEGORY_ALIASES, key=len, reverse=True):
        if alias in lower:
            return _CATEGORY_ALIASES[alias]
    return None


def extract_licensing(text: str) -> Optional[str]:
    lower = ' ' + text.lower() + ' '
    for alias in sorted(_LICENSING_ALIASES, key=len, reverse=True):
        if alias in lower:
            return _LICENSING_ALIASES[alias]
    return None


def extract_top_n(text: str, default: int = 5) -> int:
    m = re.search(r'\b(?:top|bottom|best|worst|first|last|lowest|highest)[\s\-]?(\d+)\b', text.lower())
    if m:
        return min(int(m.group(1)), 25)
    m = re.search(r'\b(\d+)\b', text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 25:
            return n
    return default


def extract_ascending(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in ('bottom', 'lowest', 'worst', 'last', 'ascending', 'asc', 'least'))


# ---------------------------------------------------------------------------
# Internal normalization helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Unicode casefold + collapse separators + whitespace normalization.

    Preserves digits, letters, and a single space between tokens so that
    "Qwen 2 - 0.5B", "Qwen2-0.5b", and "Qwen2 0.5 B" all normalize
    identically while "Qwen2" and "Qwen 2.5" remain distinct.
    """
    # NFKD decomposition then ASCII casefold
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    # Replace separators (dash, underscore, slash, dot between non-digits) with space
    s = re.sub(r'[\-_/]', ' ', s)
    # Collapse repeated spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _norm_tokens(s: str) -> list[str]:
    """Tokenize a normalized string into individual tokens."""
    return _normalize(s).split()


def _trigrams(s: str) -> set[str]:
    """Character trigram set for a normalized string (minimum length = 3)."""
    padded = f'  {s}  '
    return {padded[i:i+3] for i in range(len(padded) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between character trigram sets."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _token_overlap(query_tokens: list[str], name_tokens: list[str]) -> float:
    """Fraction of query tokens (len ≥ 2) found in the name token list."""
    relevant = [t for t in query_tokens if len(t) >= 2]
    if not relevant:
        return 0.0
    hits = sum(1 for t in relevant if t in name_tokens)
    return hits / len(relevant)


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------

def resolve_model_name(
    query_text: str,
    all_names: list[str],
    max_candidates: int = 5,
) -> list[str]:
    """Deterministic model name resolver with two-pass matching.

    Pass 1 — exact normalized match:
        If the normalized DB name appears verbatim in the normalized query,
        return those matches (up to max_candidates).  No false-positive
        equating of version/size variants.

    Pass 2 — similarity scoring:
        Compute a combined score (trigram Jaccard + token overlap) for every
        DB name vs. a "query fragment" (the longest contiguous token run in
        the query).  Apply _FUZZY_ACCEPT threshold; if the top candidate
        clears _FUZZY_MARGIN above the next, return it alone; otherwise
        return all candidates above _FUZZY_ACCEPT (up to max_candidates).

    Returns:
        A list of canonical DB model names, empty if nothing confident found.
        Multiple results indicate ambiguity; callers should present the list
        and ask the user to choose.
    """
    if not all_names:
        return []

    norm_query = _normalize(query_text)
    norm_names = [_normalize(n) for n in all_names]

    # ------------------------------------------------------------------
    # Pass 1: normalized exact substring
    # ------------------------------------------------------------------
    exact: list[str] = []
    for canonical, norm in zip(all_names, norm_names):
        if norm in norm_query:
            exact.append(canonical)
    if exact:
        return exact[:max_candidates]

    # ------------------------------------------------------------------
    # Pass 2: similarity scoring
    # We score each DB name against the full normalized query and also
    # against shorter sub-sequences to handle partial mentions.
    # ------------------------------------------------------------------
    query_tokens = norm_query.split()

    scored: list[tuple[float, str]] = []
    for canonical, norm in zip(all_names, norm_names):
        name_tokens = norm.split()

        # Trigram Jaccard on full strings
        jac_full = _trigram_jaccard(norm_query, norm)

        # Token overlap: how many DB name tokens appear in the query
        tok_overlap = _token_overlap(name_tokens, query_tokens)

        # SequenceMatcher ratio (character-level) as a tie-breaker
        sm = SequenceMatcher(None, norm_query, norm, autojunk=False).ratio()

        # Combined score weighted toward token overlap (most discriminating)
        score = 0.40 * jac_full + 0.40 * tok_overlap + 0.20 * sm

        if score >= _FUZZY_ACCEPT:
            scored.append((score, canonical))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)

    # If the best candidate has a clear margin, return it alone
    if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= _FUZZY_MARGIN:
        return [scored[0][1]]

    # Otherwise return all candidates above threshold (up to max_candidates)
    return [name for _, name in scored[:max_candidates]]
