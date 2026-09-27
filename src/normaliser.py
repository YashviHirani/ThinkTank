"""
Phase 1.1 — Normalisation
==========================
Multi-view normalisation for business names and addresses.
Never overwrites raw text; always creates new derived columns.

Views produced per record:
  name_raw           — original text (preserved)
  name_lower         — lowercased + unicode-normalised
  name_punct         — punctuation normalised (& → and, etc.)
  name_alpha         — compact alphanumeric only (no spaces)
  name_tokens        — sorted frozenset of lowercased name tokens
  name_tokens_str    — space-joined sorted tokens (for TF-IDF later)
  name_core          — name with legal suffixes removed
  name_suffix_set    — frozenset of detected legal suffixes

  addr_lower         — lowercased address
  addr_alpha         — compact alphanumeric
  addr_tokens        — frozenset of address tokens
  addr_numbers       — list[str] of all numeric tokens
  addr_house         — first numeric token (likely house/building number)
  addr_postcode      — likely postcode token (country-aware heuristic)
  country_norm       — stripped lowercased country

Usage:
    from src.normaliser import normalise_df
    df_norm = normalise_df(df)   # adds derived columns in-place copy
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from typing import FrozenSet, List, Optional

# ---------------------------------------------------------------------------
# Legal suffix dictionaries (English + French)
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES_EN = frozenset([
    "llc", "inc", "corp", "corporation", "ltd", "limited", "llp", "lp",
    "plc", "gmbh", "ag", "sa", "sl", "bv", "nv", "pty", "pvt", "private",
    "public", "company", "co", "group", "holdings", "holding", "enterprises",
    "enterprise", "services", "solutions", "associates", "association",
    "industries", "industry", "ventures", "venture", "partners", "partnership",
    "international", "intl", "global", "national", "trust", "foundation",
    "institute", "school", "college", "university", "hospital", "clinic",
    "bank", "fund", "capital", "management", "mgmt", "consulting", "consultants",
    "technologies", "technology", "tech", "systems", "network", "networks",
    "trading", "traders", "distributors", "distributor", "wholesale",
    "retail", "exports", "export", "imports", "import",
])

# French legal forms
_LEGAL_SUFFIXES_FR = frozenset([
    "sarl", "sas", "sa", "sasu", "eurl", "sci", "snc", "sc", "scp",
    "selarl", "selas", "sel", "earl", "gaec", "gie", "scop", "sca",
    "eurl", "ei", "ae",
])

# Indian legal forms
_LEGAL_SUFFIXES_IN = frozenset([
    "pvt", "private", "limited", "ltd", "llp", "opc",
])

ALL_LEGAL_SUFFIXES: FrozenSet[str] = (
    _LEGAL_SUFFIXES_EN | _LEGAL_SUFFIXES_FR | _LEGAL_SUFFIXES_IN
)

# ---------------------------------------------------------------------------
# Address abbreviation expansions (cautious set — only unambiguous mappings)
# ---------------------------------------------------------------------------

_ADDR_ABBREV = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway',
    r'\bsq\b': 'square',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bflr\b': 'floor',
    r'\bnr\b': 'near',
}

_ADDR_ABBREV_COMPILED = [(re.compile(pat, re.IGNORECASE), repl)
                          for pat, repl in _ADDR_ABBREV.items()]

# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

_AMP_RE = re.compile(r'\s*&\s*', re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_WS = re.compile(r'\s+')
_NON_ALNUM = re.compile(r'[^a-z0-9]')
_TOKEN_SPLIT = re.compile(r'\s+')

# Postcode patterns per country
_PC_US = re.compile(r'\b\d{5}(?:-\d{4})?\b')
_PC_IN = re.compile(r'\b\d{6}\b')
_PC_FR = re.compile(r'\b\d{5}\b')
_PC_GENERIC = re.compile(r'\b\d{5,6}\b')


def _unicode_norm(text: str) -> str:
    """NFKC normalise + strip."""
    return unicodedata.normalize("NFKC", text).strip()


def _name_lower(raw: str) -> str:
    """Lowercase + unicode-normalised."""
    return _unicode_norm(raw).lower()


def _name_punct(lower: str) -> str:
    """Normalise punctuation: & → and, collapse whitespace."""
    s = _AMP_RE.sub(' and ', lower)
    s = _PUNCT_RE.sub(' ', s)
    return _MULTI_WS.sub(' ', s).strip()


def _name_alpha(punct: str) -> str:
    """Compact alphanumeric only (no whitespace)."""
    return _NON_ALNUM.sub('', punct)


def _tokenise(punct: str) -> frozenset:
    """Frozenset of tokens from the punctuation-normalised name."""
    tokens = _TOKEN_SPLIT.split(punct.strip())
    return frozenset(t for t in tokens if t)


def _extract_name_core_and_suffixes(tokens: frozenset):
    """Separate core tokens from legal suffix tokens.

    Returns:
        core_str:   space-joined non-suffix tokens (sorted for determinism)
        suffix_set: frozenset of detected legal suffix tokens
    """
    suffix_hits = frozenset(t for t in tokens if t in ALL_LEGAL_SUFFIXES)
    core_tokens = tokens - suffix_hits
    core_str = ' '.join(sorted(core_tokens)) if core_tokens else ' '.join(sorted(tokens))
    return core_str, suffix_hits


# ---------------------------------------------------------------------------
# Address normalisation helpers
# ---------------------------------------------------------------------------

def _addr_lower(raw: str) -> str:
    return _unicode_norm(raw).lower()


def _addr_punct(lower: str) -> str:
    s = _AMP_RE.sub(' and ', lower)
    s = _PUNCT_RE.sub(' ', s)
    return _MULTI_WS.sub(' ', s).strip()


def _addr_abbrev_expand(punct: str) -> str:
    for pat, repl in _ADDR_ABBREV_COMPILED:
        punct = pat.sub(repl, punct)
    return _MULTI_WS.sub(' ', punct).strip()


def _addr_tokens(punct: str) -> frozenset:
    tokens = _TOKEN_SPLIT.split(punct.strip())
    return frozenset(t for t in tokens if t)


def _addr_numbers(tokens: frozenset) -> List[str]:
    return sorted(t for t in tokens if t.isdigit())


def _addr_house(numbers: List[str]) -> str:
    """Return first numeric token as a proxy for building/house number."""
    return numbers[0] if numbers else ''


def _addr_postcode(raw_addr: str, country_norm: str) -> str:
    """Extract a postcode using country-appropriate regex."""
    if country_norm in ('us', 'france', 'fr'):
        m = _PC_US.search(raw_addr) if country_norm == 'us' else _PC_FR.search(raw_addr)
    elif country_norm in ('india', 'in'):
        m = _PC_IN.search(raw_addr)
    else:
        m = _PC_GENERIC.search(raw_addr)
    return m.group(0) if m else ''


# ---------------------------------------------------------------------------
# Country normalisation
# ---------------------------------------------------------------------------

def _country_norm(raw: str) -> str:
    return _unicode_norm(raw).lower().strip()


# ---------------------------------------------------------------------------
# Vectorised per-DataFrame normalisation
# ---------------------------------------------------------------------------

def normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised/derived columns to a copy of df.

    Fully vectorized — uses Pandas string ops to avoid per-row Python overhead.

    Columns added (never overwrites originals):
      name_lower, name_punct, name_alpha, name_tokens_str,
      name_core, name_suffix_str, country_norm,
      addr_lower, addr_punct, addr_alpha,
      addr_house, addr_postcode
    """
    out = df.copy()

    # ------------------------------------------------------------------ country
    out['country_norm'] = (
        out['country']
        .str.normalize('NFKC')
        .str.strip()
        .str.lower()
    )

    # ------------------------------------------------------------------ name pipeline
    # 1. Lowercase + NFKC
    out['name_lower'] = out['business_name'].str.normalize('NFKC').str.strip().str.lower()

    # 2. & → and
    out['name_punct'] = out['name_lower'].str.replace(r'\s*&\s*', ' and ', regex=True)
    # Remove punctuation except alphanumeric/space
    out['name_punct'] = out['name_punct'].str.replace(r"[^a-z0-9\s]", ' ', regex=True)
    # Collapse whitespace
    out['name_punct'] = out['name_punct'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 3. Alpha (no spaces)
    out['name_alpha'] = out['name_punct'].str.replace(r'[^a-z0-9]', '', regex=True)

    # 4. Tokens, core, suffix — per-row apply is unavoidable for set ops but we do one pass
    def _name_fields(punct: str):
        tokens = frozenset(_TOKEN_SPLIT.split(punct.strip())) - frozenset([''])
        core, suffixes = _extract_name_core_and_suffixes(tokens)
        return ' '.join(sorted(tokens)), core, ' '.join(sorted(suffixes))

    _name_tuples = out['name_punct'].apply(_name_fields)
    out['name_tokens_str'] = _name_tuples.apply(lambda x: x[0])
    out['name_core']       = _name_tuples.apply(lambda x: x[1])
    out['name_suffix_str'] = _name_tuples.apply(lambda x: x[2])

    # ------------------------------------------------------------------ address pipeline
    # 1. Lowercase
    out['addr_lower'] = out['business_address'].str.normalize('NFKC').str.strip().str.lower()

    # 2. Punctuation normalise
    out['addr_punct'] = out['addr_lower'].str.replace(r'\s*&\s*', ' and ', regex=True)
    out['addr_punct'] = out['addr_punct'].str.replace(r"[^a-z0-9\s]", ' ', regex=True)
    out['addr_punct'] = out['addr_punct'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 3. Abbreviation expansion (vectorized per pattern)
    for pat, repl in _ADDR_ABBREV_COMPILED:
        out['addr_punct'] = out['addr_punct'].str.replace(pat.pattern, repl, regex=True, case=False)
    out['addr_punct'] = out['addr_punct'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 4. Alpha
    out['addr_alpha'] = out['addr_punct'].str.replace(r'[^a-z0-9]', '', regex=True)

    # 5. House number: first numeric token in addr_punct
    def _extract_house(addr_punct: str) -> str:
        if not addr_punct:
            return ''
        tokens = addr_punct.split()
        for t in tokens:
            if t.isdigit():
                return t
        return ''

    out['addr_house'] = out['addr_punct'].apply(_extract_house)

    # 6. Postcode: country-aware regex on original business_address
    # Vectorized per country group
    out['addr_postcode'] = ''

    for country_val in out['country_norm'].unique():
        mask = out['country_norm'] == country_val
        addrs = out.loc[mask, 'business_address']

        if country_val == 'us':
            extracted = addrs.str.extract(r'(\b\d{5}(?:-\d{4})?\b)', expand=False)
        elif country_val in ('india', 'in'):
            extracted = addrs.str.extract(r'(\b\d{6}\b)', expand=False)
        elif country_val in ('france', 'fr'):
            extracted = addrs.str.extract(r'(\b\d{5}\b)', expand=False)
        else:
            extracted = addrs.str.extract(r'(\b\d{5,6}\b)', expand=False)

        out.loc[mask, 'addr_postcode'] = extracted.fillna('')

    return out



# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_data = pd.DataFrame([
        {
            'entity_id': 'S1-001',
            'business_name': 'ABC Corp & Co., LLC',
            'business_address': '123 Main St, Springfield, IL 62701',
            'country': 'US',
        },
        {
            'entity_id': 'S2-001',
            'business_name': 'Abc Corp and Company Limited',
            'business_address': '123 Main Road, Springfield 62701',
            'country': 'US',
        },
        {
            'entity_id': 'S2-002',
            'business_name': 'Sharma Pvt Ltd',
            'business_address': 'Plot 45 Industrial Area Pune 411018',
            'country': 'India',
        },
        {
            'entity_id': 'S2-003',
            'business_name': 'Dupont SAS',
            'business_address': '12 Rue de la Paix 75001 Paris',
            'country': 'France',
        },
    ])
    norm = normalise_df(test_data)
    for col in ['name_lower', 'name_punct', 'name_alpha', 'name_core',
                'name_suffix_str', 'addr_house', 'addr_postcode', 'country_norm']:
        print(f"\n--- {col} ---")
        print(norm[['entity_id', col]].to_string(index=False))
    print("\n[OK] Normaliser smoke test passed")
