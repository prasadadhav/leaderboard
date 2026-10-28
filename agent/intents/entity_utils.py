"""
Lightweight entity extraction for CEFR level, task category, licensing, top-N.
Used inside BAF state bodies; no external dependencies beyond the standard library.
"""
import re
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


def extract_cefr_level(text: str) -> Optional[str]:
    upper = text.upper()
    for lvl in CEFR_LEVELS:
        if re.search(r'\b' + lvl + r'\b', upper):
            return lvl
    # also check spelled-out level names
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


def resolve_model_name(query_text: str, all_names: list[str]) -> list[str]:
    """
    Best-effort model name resolution from a query string.
    Returns a list of candidate model names (may be empty or multiple).
    """
    lower = query_text.lower()
    # Exact substring match first
    exact = [n for n in all_names if n.lower() in lower]
    if exact:
        return exact[:3]
    # Word-level partial match (min 4 chars to avoid false positives)
    words = [w for w in re.split(r'[\s\-\'\"]+', lower) if len(w) >= 4]
    partial = [n for n in all_names if any(w in n.lower() for w in words)]
    return partial[:5]
