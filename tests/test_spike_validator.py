"""spike/validator.py -- built and tested FIRST, per the bounded
agent-loop spike's instructions. Every case here is a fabricated
draft reply + fabricated tool results, no live agent/API involved."""
from __future__ import annotations

from spike.validator import build_fact_ledger, validate_reply


def test_cited_figure_passes():
    tool_results = [{"wmape": 0.7302, "n_series": 26304}]
    reply = "AutoETS scored a WMAPE of 0.7302 across 26,304 series."
    result = validate_reply(reply, tool_results)
    assert result.passed
    assert result.uncited_figures == []


def test_uncited_figure_fails():
    tool_results = [{"wmape": 0.7302, "n_series": 26304}]
    reply = "AutoETS scored a WMAPE of 0.65, which is quite good."
    result = validate_reply(reply, tool_results)
    assert not result.passed
    assert "0.65" in result.uncited_figures


def test_entirely_fabricated_reply_with_no_tool_results_fails():
    reply = "The comparator achieved a WMAPE of 0.71 on fold 3."
    result = validate_reply(reply, tool_results=[])
    assert not result.passed
    assert "0.71" in result.uncited_figures


def test_percentage_form_of_a_fractional_fact_passes():
    tool_results = [{"relative_difference": 0.0031}]
    reply = "ADVI and NUTS agreed within 0.31% relative difference."
    result = validate_reply(reply, tool_results)
    assert result.passed


def test_fractional_form_of_a_percentage_style_fact_passes():
    tool_results = [{"pct_improvement": 15.0}]
    reply = "That is a 0.15 relative improvement."
    result = validate_reply(reply, tool_results)
    assert result.passed


def test_rounded_display_of_a_fact_within_tolerance_passes():
    tool_results = [{"fold_id": 2, "wmape": 0.814510608294286}]
    reply = "Fold 2 scored a WMAPE of approximately 0.8145."
    result = validate_reply(reply, tool_results)
    assert result.passed


def test_grossly_different_number_fails_even_if_a_similar_fact_exists():
    tool_results = [{"wmape": 0.8145}]
    reply = "Fold 2 scored a WMAPE of 0.95."
    result = validate_reply(reply, tool_results)
    assert not result.passed
    assert "0.95" in result.uncited_figures


def test_iso_date_in_reply_is_never_treated_as_an_uncited_figure():
    tool_results = [{"generated_at": "2026-07-30T10:15:00Z"}]
    reply = "This forecast was generated at 2026-07-30T10:15:00Z."
    result = validate_reply(reply, tool_results)
    assert result.passed  # "2026", "07", "30" etc. must not be flagged


def test_nested_tool_result_facts_are_all_reachable():
    tool_results = [
        {
            "metrics": {"seasonal_naive": {"per_fold_wmape": [0.8572, 0.8656], "mean_wmape": 0.8630}},
            "fold_ids": [1, 2],
        }
    ]
    reply = "Seasonal naive scored 0.8572 in fold 1 and 0.8656 in fold 2, averaging 0.8630."
    result = validate_reply(reply, tool_results)
    assert result.passed


def test_multiple_tool_results_pool_into_one_ledger():
    tool_results = [{"wmape": 0.73}, {"n_series": 1000}]
    reply = "WMAPE was 0.73 across 1000 series."
    result = validate_reply(reply, tool_results)
    assert result.passed


def test_champion_claim_fails_even_when_every_number_is_cited():
    tool_results = [{"wmape": 0.73}]
    reply = "auto_ets is the champion model, with a WMAPE of 0.73."
    result = validate_reply(reply, tool_results)
    assert not result.passed
    assert result.champion_claim_found
    assert "champion" in result.failure_reason


def test_winner_claim_also_fails():
    tool_results = [{"wmape": 0.73}]
    reply = "auto_ets is the clear winner here, scoring 0.73."
    result = validate_reply(reply, tool_results)
    assert not result.passed
    assert result.champion_claim_found


def test_best_model_claim_also_fails():
    tool_results = [{"wmape": 0.73}]
    reply = "This is the best model available, at 0.73 WMAPE."
    result = validate_reply(reply, tool_results)
    assert not result.passed
    assert result.champion_claim_found


def test_reply_with_no_numbers_and_no_champion_claim_passes_trivially():
    result = validate_reply("I don't have that information available right now.", tool_results=[])
    assert result.passed


def test_failure_reason_reports_uncited_figures_when_no_champion_claim():
    tool_results = [{"wmape": 0.73}]
    reply = "The real number is 0.99."
    result = validate_reply(reply, tool_results)
    assert "uncited figure" in result.failure_reason


def test_build_fact_ledger_ignores_boolean_leaves():
    tool_results = [{"licensed": True, "wmape": 0.73}]
    ledger = build_fact_ledger(tool_results)
    assert 1.0 not in ledger  # True must not be flattened into a numeric fact
    assert 0.73 in ledger
