"""Response validator -- the B.3 discipline brought forward from Phase
2 (`02_AGENT_TOOL_CONTRACTS.md` design principle 1: "the agent
narrates facts, it does not invent them"). Every numeric figure in a
draft agent reply must resolve to a fact actually returned by a tool
call this turn. An uncited figure is a hard failure; callers
(`agent_loop.py`) regenerate a fresh draft or fall back to a fixed
refusal, never let it reach the user as-is.

Design: walk every tool result produced this turn, flatten it, and
collect every numeric leaf value into a fact ledger. Extract every
standalone numeric figure from the draft reply and check it against
the ledger (allowing both the raw value and its x100 percentage form,
plus a small tolerance for display rounding). ISO 8601 dates/
datetimes are recognised and excluded from the number-extraction pass
first -- a date is not a "figure" in the sense this discipline
targets, and naively extracting "2026", "07", "30" from "2026-07-30"
would make stating a timestamp impossible.

A separate, independent check rejects champion/winner/best claims
outright, regardless of citation -- Phase 1's gates are all PENDING
(no champion assigned yet), and this must hold even if the system
prompt's instruction to that effect were ignored or bypassed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_ISO_DATETIME_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?\b")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")

ABS_TOLERANCE = 0.01
REL_TOLERANCE = 0.02

CHAMPION_CLAIM_RE = re.compile(r"\b(champion|winner|winning model|best model|is the best|has won)\b", re.IGNORECASE)


class ChampionClaimError(Exception):
    """Raised when a draft reply characterises any model as champion,
    winner, or best -- not permitted while gate evaluation is
    PENDING (reports/GATE_REPORT.md), regardless of citation."""


@dataclass
class ValidationResult:
    passed: bool
    uncited_figures: list[str] = field(default_factory=list)
    champion_claim_found: bool = False
    known_facts: list[float] = field(default_factory=list)

    @property
    def failure_reason(self) -> str | None:
        if self.champion_claim_found:
            return "reply characterises a model as champion/winner/best while gate evaluation is PENDING"
        if self.uncited_figures:
            return f"reply contains uncited figure(s): {self.uncited_figures}"
        return None


def _flatten_numeric_leaves(obj) -> list[float]:
    """Recursively collect every int/float leaf value from a tool
    result payload (dicts, lists, nested structures, dataclasses
    already converted to plain data) -- these are the ONLY numbers an
    agent reply is permitted to cite."""
    values: list[float] = []
    if isinstance(obj, bool):
        return values  # bools are not "figures"
    if isinstance(obj, (int, float)):
        values.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            values.extend(_flatten_numeric_leaves(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            values.extend(_flatten_numeric_leaves(v))
    return values


def build_fact_ledger(tool_results: list[object]) -> list[float]:
    """tool_results: the raw Python return values of every tool called
    this turn (before JSON serialisation) -- the full set of numbers
    the agent is permitted to state in its final reply."""
    facts: list[float] = []
    for result in tool_results:
        facts.extend(_flatten_numeric_leaves(result))
    return facts


def _extract_candidate_figures(text: str) -> list[str]:
    without_dates = _ISO_DATETIME_RE.sub(" ", text)
    return _NUMBER_RE.findall(without_dates)


def _figure_matches_ledger(figure: str, ledger: list[float]) -> bool:
    is_percent = figure.endswith("%")
    raw = figure.rstrip("%").replace(",", "")
    if raw in ("", "-", "."):
        return True  # not a real number (e.g. a stray "-" from a range dash)
    try:
        value = float(raw)
    except ValueError:
        return True  # not parseable as a number -- not this validator's concern

    candidates = [value]
    # the underlying fact might be stored as a fraction (0.15) and
    # cited as a percentage ("15%"), or vice versa -- check both forms.
    candidates.append(value / 100.0 if is_percent else value * 100.0)

    for fact in ledger:
        for candidate in candidates:
            if math.isclose(fact, candidate, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE):
                return True
    return False


def validate_reply(draft_reply: str, tool_results: list[object]) -> ValidationResult:
    ledger = build_fact_ledger(tool_results)
    candidates = _extract_candidate_figures(draft_reply)
    uncited = [c for c in candidates if not _figure_matches_ledger(c, ledger)]
    champion_claim = bool(CHAMPION_CLAIM_RE.search(draft_reply))
    return ValidationResult(
        passed=not uncited and not champion_claim,
        uncited_figures=uncited,
        champion_claim_found=champion_claim,
        known_facts=ledger,
    )
