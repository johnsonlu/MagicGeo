import pytest

from geo.Auxiliary_function import filter_conditions_with_llm


@pytest.mark.llm_condition_check
def test_filter_conditions_skips_trusted_types(monkeypatch):
    calls = []

    def fake_batch_check(text, to_check):
        calls.append(to_check)
        return {}

    monkeypatch.setattr(
        "geo.Auxiliary_function._llm_batch_check", fake_batch_check
    )

    conditions = [
        ["ortho", ["A", "B", "B", "C"]],
        ["dist", ["A", "B", 1.0]],
    ]
    kept = filter_conditions_with_llm("题目文本", conditions)

    assert kept == conditions
    assert calls == []


@pytest.mark.llm_condition_check
def test_filter_conditions_batches_parallel_verdicts(monkeypatch):
    calls = []

    def fake_batch_check(text, to_check):
        calls.append((text, to_check))
        return {"c0": "yes", "c1": "no"}

    monkeypatch.setattr(
        "geo.Auxiliary_function._llm_batch_check", fake_batch_check
    )

    conditions = [
        ["parallel", ["A", "B", "D", "C"]],
        ["parallel", ["A", "D", "B", "C"]],
        ["ortho", ["A", "B", "A", "D"]],
    ]
    kept = filter_conditions_with_llm("矩形ABCD", conditions)

    assert kept == [
        ["parallel", ["A", "B", "D", "C"]],
        ["ortho", ["A", "B", "A", "D"]],
    ]
    assert len(calls) == 1
    assert len(calls[0][1]) == 2


@pytest.mark.llm_condition_check
def test_filter_conditions_keeps_all_on_json_parse_failure(monkeypatch):
    def fake_batch_check(text, to_check):
        return None

    monkeypatch.setattr(
        "geo.Auxiliary_function._llm_batch_check", fake_batch_check
    )

    conditions = [["parallel", ["A", "B", "C", "D"]]]
    kept = filter_conditions_with_llm("题目", conditions)

    assert kept == conditions


@pytest.mark.llm_condition_check
def test_filter_conditions_keeps_missing_verdict_keys(monkeypatch):
    def fake_batch_check(text, to_check):
        return {"c0": "yes"}

    monkeypatch.setattr(
        "geo.Auxiliary_function._llm_batch_check", fake_batch_check
    )

    conditions = [
        ["parallel", ["A", "B", "D", "C"]],
        ["parallel", ["A", "D", "B", "C"]],
    ]
    kept = filter_conditions_with_llm("题目", conditions)

    assert kept == conditions
