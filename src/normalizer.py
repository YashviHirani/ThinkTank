"""
Phase 1, Step 5.1: Text Normalisation Module
=============================================
Provides high-performance, deterministic text normalisation and token
extraction for business names and addresses across US, India, and France.

Key Capabilities:
  - Unicode NFKD decomposition: strips Latin diacritics (é -> e, ç -> c) while
    safely preserving non-Latin scripts (e.g. Indic Devanagari).
  - Multi-jurisdiction legal suffix separation (US, India, France) handling both
    trailing and leading positions.
  - Sorted token views for word-order invariance.
  - Address cleaning and numeric token extraction (house/plot/postcodes).
  - Fast vectorized DataFrame processing or single-string transformations.

Usage:
    # Run test suite:
    python src/normalizer.py --test
"""

import os
import re
import sys
import unicodedata
import pandas as pd
from typing import Tuple, List, Set, Dict, Any


# ---------------------------------------------------------------------------
# Precompiled Regexes and Constants
# ---------------------------------------------------------------------------

RE_SYMBOLS = [
    (re.compile(r"&"), " and "),
    (re.compile(r"@"), " at "),
    (re.compile(r"\+"), " plus "),
]

# Common road/address abbreviations to expand
ADDRESS_EXPANSIONS = [
    (re.compile(r"\bst\b", re.IGNORECASE), "street"),
    (re.compile(r"\brd\b", re.IGNORECASE), "road"),
    (re.compile(r"\bave\b", re.IGNORECASE), "avenue"),
    (re.compile(r"\bdr\b", re.IGNORECASE), "drive"),
    (re.compile(r"\bblvd\b", re.IGNORECASE), "boulevard"),
    (re.compile(r"\bln\b", re.IGNORECASE), "lane"),
    (re.compile(r"\bct\b", re.IGNORECASE), "court"),
    (re.compile(r"\bp\.?\s*o\.?\s*box\b", re.IGNORECASE), "pobox"),
    (re.compile(r"\b(h\.?\s*no|house\s*no|kh\.?\s*no|plot\s*no)\b", re.IGNORECASE), "no"),
]

# Legal suffixes across US, India, and France (ordered from multi-word to single-word)
LEGAL_SUFFIXES = [
    # India / UK multi-word
    "private limited",
    "pvt limited",
    "private ltd",
    "pvt ltd",
    # French forms
    "sarl",
    "eurl",
    "sasu",
    "scop",
    "snc",
    "sci",
    "gie",
    "sas",
    "sa",
    # US / General forms
    "incorporated",
    "corporation",
    "limited",
    "enterprise",
    "enterprises",
    "proprietorship",
    "pllc",
    "corp",
    "inc",
    "llc",
    "llp",
    "ltd",
    "pvt",
    "co",
]

# Precompile regex to match legal suffix at end of clean name
# e.g. "acme widgets pvt ltd" -> ("acme widgets", "pvt ltd")
SUFFIX_END_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")$",
    re.IGNORECASE
)

# Precompile regex to match legal suffix at start of clean name
# e.g. "llc moncada learning center" -> ("moncada learning center", "llc")
SUFFIX_START_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")\b",
    re.IGNORECASE
)

RE_NON_ALPHANUMERIC = re.compile(r"[^\w\s\u0900-\u0D7F]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\d+")
RE_POSTCODE = re.compile(r"\b\d{5,6}\b")


# ---------------------------------------------------------------------------
# String Transformation Functions
# ---------------------------------------------------------------------------

def strip_latin_diacritics(text: str) -> str:
    """
    Decompose accents using NFKD, stripping only combining diacritical marks
    (U+0300 to U+036F) while keeping Indic and other non-Latin scripts intact.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    # Strip characters in Combining Diacritical Marks block
    return "".join(c for c in decomposed if not (0x0300 <= ord(c) <= 0x036F))


def normalize_general_text(text: str) -> str:
    """
    Base cleaner:
      1. Strip Latin diacritics (é -> e, ç -> c)
      2. Lowercase
      3. Standardize symbols (& -> and, @ -> at, + -> plus)
      4. Remove website noise (.com, .org, www)
      5. Remove punctuation
      6. Collapse consecutive spaces
    """
    if not text or not isinstance(text, str):
        return ""

    text = strip_latin_diacritics(text).lower()

    # Strip domain endings and www
    text = re.sub(r"\bwww\b", " ", text)
    text = re.sub(r"\.(com|co|in|org|net|io|edu|gov)\b", " ", text)

    for pattern, replacement in RE_SYMBOLS:
        text = pattern.sub(replacement, text)

    # Remove punctuation characters (preserving all alphanumeric letters in any script)
    text = RE_NON_ALPHANUMERIC.sub(" ", text)
    # Collapse multiple whitespaces
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def extract_core_name(clean_name: str) -> Tuple[str, str]:
    """
    Detaches recognized legal suffixes from clean_name.
    Also strips leading 'the' if followed by substantive name.
    Checks end of string first, then start of string.

    Returns:
        (core_name, detected_suffix)
    """
    if not clean_name:
        return "", ""

    suffix = ""
    core = clean_name

    # Check suffix at the end (most common: "Custom Wealth Services LLC")
    match_end = SUFFIX_END_PATTERN.search(clean_name)
    if match_end:
        suffix = match_end.group(1).lower()
        sub = clean_name[: match_end.start()].strip()
        if sub:
            core = sub

    # Check suffix at the start (e.g. "LLC Moncada Learning Center")
    if not suffix:
        match_start = SUFFIX_START_PATTERN.search(clean_name)
        if match_start:
            suffix = match_start.group(1).lower()
            sub = clean_name[match_start.end() :].strip()
            if sub:
                core = sub

    # Strip leading 'the ' if remaining core has at least 4 chars
    if core.startswith("the ") and len(core) >= 8:
        core = core[4:].strip()

    return core, suffix


def get_sorted_tokens(clean_text: str) -> str:
    """
    Returns alphabetically sorted tokens as a single string.
    Makes 'Starbucks Coffee' and 'Coffee Starbucks' identical.
    """
    if not clean_text:
        return ""
    tokens = clean_text.lower().split()
    tokens.sort()
    return " ".join(tokens)


def get_compact_alphanumeric(clean_text: str) -> str:
    """
    Removes all spaces for compact prefix indexing and typo resilience.
    """
    if not clean_text:
        return ""
    return clean_text.replace(" ", "")


def normalize_address(address_str: str) -> str:
    """
    Cleans address and expands standard road/building abbreviations.
    """
    if not address_str or not isinstance(address_str, str):
        return ""

    clean_addr = normalize_general_text(address_str)
    for pattern, replacement in ADDRESS_EXPANSIONS:
        clean_addr = pattern.sub(replacement, clean_addr)
    return RE_WHITESPACE.sub(" ", clean_addr).strip()


def extract_numeric_tokens(address_str: str) -> List[str]:
    """
    Extracts all digit tokens from an address (building numbers, plot numbers, postcodes).
    """
    if not address_str:
        return []
    return RE_NUMBERS.findall(address_str)


def extract_postcode_candidate(address_str: str) -> str:
    """
    Extracts the first standalone 5 or 6 digit number, or empty string if none.
    """
    if not address_str:
        return ""
    match = RE_POSTCODE.search(address_str)
    return match.group(0) if match else ""


# ---------------------------------------------------------------------------
# High-Level Entity Normalisation Pipeline
# ---------------------------------------------------------------------------

def normalize_entity_record(name: str, address: str, country: str) -> Dict[str, Any]:
    """
    Normalizes a single entity record and returns all deterministic views.
    """
    clean_name = normalize_general_text(name)
    core_name, suffix = extract_core_name(clean_name)
    sorted_name = get_sorted_tokens(clean_name)
    compact_name = get_compact_alphanumeric(clean_name)

    clean_addr = normalize_address(address)
    num_tokens = extract_numeric_tokens(clean_addr)
    postcode = extract_postcode_candidate(clean_addr)

    norm_country = country.strip().upper() if country else ""

    return {
        "clean_name": clean_name,
        "core_name": core_name,
        "legal_suffix": suffix,
        "sorted_name": sorted_name,
        "compact_name": compact_name,
        "clean_address": clean_addr,
        "numeric_tokens": num_tokens,
        "postcode": postcode,
        "country": norm_country,
    }


def normalize_dataframe(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Vectorized normalization on a pandas DataFrame with standard columns:
    ['entity_id', 'business_name', 'business_address', 'country']

    Appends normalized views without overwriting original columns.
    """
    if not inplace:
        df = df.copy()

    # 1. Clean names
    print("    Normalising business names...", end=" ", flush=True)
    df["clean_name"] = df["business_name"].fillna("").apply(normalize_general_text)

    # Core name & suffix
    core_and_suffix = df["clean_name"].apply(extract_core_name)
    df["core_name"] = [cs[0] for cs in core_and_suffix]
    df["legal_suffix"] = [cs[1] for cs in core_and_suffix]

    # Sorted tokens
    df["sorted_name"] = df["clean_name"].apply(get_sorted_tokens)
    print("done.")

    # 2. Clean addresses & extract numbers
    print("    Normalising addresses & numbers...", end=" ", flush=True)
    df["clean_address"] = df["business_address"].fillna("").apply(normalize_address)
    df["numeric_tokens"] = df["clean_address"].apply(extract_numeric_tokens)
    df["postcode"] = df["clean_address"].apply(extract_postcode_candidate)
    print("done.")

    # 3. Country
    df["country_clean"] = df["country"].fillna("").str.strip().str.upper()

    return df


# ---------------------------------------------------------------------------
# Unit Test Suite
# ---------------------------------------------------------------------------

def run_tests() -> bool:
    """Runs automated verification suite for normalizer functions."""
    passed = 0
    failed = 0

    def assert_eq(test_name: str, got: Any, expected: Any):
        nonlocal passed, failed
        if got == expected:
            print(f"  [PASS] {test_name}")
            passed += 1
        else:
            enc = sys.stdout.encoding or "ascii"
            got_safe = str(got).encode(enc, errors="backslashreplace").decode(enc)
            exp_safe = str(expected).encode(enc, errors="backslashreplace").decode(enc)
            print(f"  [FAIL] {test_name}")
            print(f"         Got:      {got_safe!r}")
            print(f"         Expected: {exp_safe!r}")
            failed += 1

    print("\n--- Running Normalizer Unit Tests ---\n")

    # 1. Diacritics stripping (French)
    res = normalize_general_text("Café Étoile Société SARL")
    assert_eq("French diacritics removal", res, "cafe etoile societe sarl")

    # 2. Non-Latin preservation (Indic script)
    hi_text = "सुरेश एंटरप्राइज प्राइवेट लिमिटेड"
    res_hi = normalize_general_text(hi_text)
    assert_eq("Indic script preserved", res_hi, hi_text)

    # 3. Symbol expansion (&, @, +)
    res_sym = normalize_general_text("Barnes & Noble @ Home + Work")
    assert_eq("Symbol expansion", res_sym, "barnes and noble at home plus work")

    # 4. Trailing legal suffix detachment
    core, suffix = extract_core_name("custom wealth services llc")
    assert_eq("Trailing suffix - core", core, "custom wealth services")
    assert_eq("Trailing suffix - suffix", suffix, "llc")

    # 5. Multi-word Indian suffix detachment
    core_in, suffix_in = extract_core_name("acme international pvt ltd")
    assert_eq("Multi-word suffix - core", core_in, "acme international")
    assert_eq("Multi-word suffix - suffix", suffix_in, "pvt ltd")

    # 6. Leading legal suffix detachment
    core_lead, suffix_lead = extract_core_name("llc moncada learning center")
    assert_eq("Leading suffix - core", core_lead, "moncada learning center")
    assert_eq("Leading suffix - suffix", suffix_lead, "llc")

    # 7. French suffix detachment (SARL)
    core_fr, suffix_fr = extract_core_name("boulangerie des alpes sarl")
    assert_eq("French suffix - core", core_fr, "boulangerie des alpes")
    assert_eq("French suffix - suffix", suffix_fr, "sarl")

    # 8. Sorted tokens (word-order invariance)
    s1 = get_sorted_tokens("Starbucks Coffee")
    s2 = get_sorted_tokens("Coffee Starbucks")
    assert_eq("Sorted tokens identical", s1, s2)
    assert_eq("Sorted tokens value", s1, "coffee starbucks")

    # 9. Address expansion & cleaning
    addr = normalize_address("1795 Westchester Dr., Apt. 4B, High Point, NC")
    assert_eq("Address road expansion", addr, "1795 westchester drive apt 4b high point nc")

    # 10. Numeric tokens extraction
    nums = extract_numeric_tokens("1795 Westchester Drive, Apt 4B, High Point, NC 27262")
    assert_eq("Numeric tokens extracted", nums, ["1795", "4", "27262"])

    # 11. Postcode extraction candidate
    pc = extract_postcode_candidate("1795 Westchester Drive, High Point, NC 27262")
    assert_eq("Postcode extracted", pc, "27262")

    # 12. Empty string safety
    empty_res = normalize_entity_record("", "", "")
    assert_eq("Empty record safety", empty_res["core_name"], "")

    print(f"\n--- Results: {passed} passed, {failed} failed ---")
    return failed == 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  python src/normalizer.py --test   Run normalizer test suite")
