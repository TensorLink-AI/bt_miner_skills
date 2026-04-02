"""Tests for the backtest evidence extraction system."""

import pytest
from ralph_loop.evidence import (
    extract_crps_scores,
    extract_asset_scores,
    extract_baseline_scores,
    has_synth_api_evidence,
    has_emulator_validation,
    has_false_confidence,
    extract_evidence,
    build_evidence_summary,
    build_evidence_gate_warning,
)
from ralph_loop.state import LoopState


class TestExtractCrpsScores:
    def test_basic_crps(self):
        text = "CRPS: 145.32"
        assert extract_crps_scores(text) == [145.32]

    def test_crps_equals(self):
        text = "crps = 200.5"
        assert extract_crps_scores(text) == [200.5]

    def test_crps_sum(self):
        text = "crps_sum: 98.7"
        assert extract_crps_scores(text) == [98.7]

    def test_overall_crps(self):
        text = "overall_crps = 112.3"
        assert extract_crps_scores(text) == [112.3]

    def test_multiple_scores(self):
        text = "CRPS: 100.0\ncrps_sum: 200.0\nprompt_score: 0.85"
        scores = extract_crps_scores(text)
        assert 100.0 in scores
        assert 200.0 in scores
        assert 0.85 in scores

    def test_no_scores(self):
        text = "Writing validator_emulator.py to disk..."
        assert extract_crps_scores(text) == []

    def test_code_not_output(self):
        # Code that defines CRPS but doesn't produce scores
        text = "def calculate_crps(predictions, actual):\n    return crps_score"
        assert extract_crps_scores(text) == []


class TestExtractAssetScores:
    def test_asset_crps(self):
        text = "BTC CRPS: 145.0\nETH CRPS: 200.3"
        scores = extract_asset_scores(text)
        assert scores["BTC"] == 145.0
        assert scores["ETH"] == 200.3

    def test_bracket_format(self):
        text = "CRPS[SOL]: 88.5"
        scores = extract_asset_scores(text)
        assert scores["SOL"] == 88.5

    def test_no_assets(self):
        text = "Training complete."
        assert extract_asset_scores(text) == {}


class TestExtractBaselineScores:
    def test_gbm_baseline(self):
        text = "GBM CRPS: 250.0"
        scores = extract_baseline_scores(text)
        assert scores["GBM"] == 250.0

    def test_historical_sim(self):
        text = "historical_simulation score: 180.5"
        scores = extract_baseline_scores(text)
        assert scores["historical_sim"] == 180.5

    def test_generic_baseline(self):
        text = "baseline CRPS: 300.0"
        scores = extract_baseline_scores(text)
        assert scores["baseline"] == 300.0


class TestSynthApiEvidence:
    def test_synth_api_present(self):
        text = "Fetching from synth API... live CRPS for BTC: 150"
        assert has_synth_api_evidence(text) is True

    def test_validation_endpoint(self):
        text = "GET /validation/scores/latest returned network score 145"
        assert has_synth_api_evidence(text) is True

    def test_no_api_evidence(self):
        text = "Wrote model to disk."
        assert has_synth_api_evidence(text) is False


class TestFalseConfidence:
    def test_ready_to_deploy(self):
        assert has_false_confidence("Model is ready to deploy!") is True

    def test_deployment_ready(self):
        assert has_false_confidence("STATUS: deployment ready") is True

    def test_pipeline_complete(self):
        assert has_false_confidence("Pipeline is complete, moving to deploy") is True

    def test_normal_status(self):
        assert has_false_confidence("STATUS: training model iteration 3") is False


class TestExtractEvidence:
    def test_real_backtest_output(self):
        exec_output = """
=== BACKTEST RESULTS ===
Model: dlinear_v1
Overall CRPS: 145.32
BTC CRPS: 130.5
ETH CRPS: 160.2
GBM baseline CRPS: 250.0
=== END RESULTS ===
"""
        evidence = extract_evidence(exec_output, "STATUS: Backtest complete", iteration=5)
        assert evidence["has_real_scores"] is True
        assert evidence["claims_ready_without_evidence"] is False
        assert 145.32 in evidence["crps_scores_found"]
        assert evidence["asset_scores"]["BTC"] == 130.5
        assert evidence["baseline_scores"]["GBM"] == 250.0

    def test_no_evidence_but_claims_ready(self):
        exec_output = "Written: model.py (1234 bytes)"
        llm_response = "STATUS: Pipeline is complete and ready to deploy"
        evidence = extract_evidence(exec_output, llm_response, iteration=7)
        assert evidence["has_real_scores"] is False
        assert evidence["claims_ready"] is True
        assert evidence["claims_ready_without_evidence"] is True

    def test_code_writing_only(self):
        exec_output = """
Written: validator_emulator.py (5432 bytes)
Written: train.py (3210 bytes)
Written: data_pipeline.py (2100 bytes)
"""
        llm_response = "STATUS: Building the training pipeline"
        evidence = extract_evidence(exec_output, llm_response, iteration=3)
        assert evidence["has_real_scores"] is False
        assert evidence["claims_ready"] is False


class TestBuildEvidenceSummary:
    def test_no_results(self):
        state = LoopState()
        assert build_evidence_summary(state) == ""

    def test_with_scores(self):
        state = LoopState(
            iteration_count=5,
            has_backtest_scores=True,
            has_baseline_comparison=False,
            backtest_results=[{
                "has_real_scores": True,
                "crps_scores_found": [145.0],
                "asset_scores": {"BTC": 130.0, "ETH": 160.0},
            }],
        )
        summary = build_evidence_summary(state)
        assert "Backtest Evidence Tracker" in summary
        assert "1/5" in summary
        assert "MISSING: Baseline comparison" in summary
        assert "BTC" in summary
        assert "Assets MISSING" in summary  # only 2 of 9 assets

    def test_no_scores_warning(self):
        state = LoopState(
            iteration_count=7,
            has_backtest_scores=False,
            backtest_results=[{"has_real_scores": False}],
        )
        summary = build_evidence_summary(state)
        assert "WARNING" in summary
        assert "No actual CRPS scores" in summary


class TestEvidenceGateWarning:
    def test_no_warning_when_no_results(self):
        state = LoopState()
        assert build_evidence_gate_warning(state) == ""

    def test_warning_on_false_readiness(self):
        state = LoopState(
            backtest_results=[{"claims_ready_without_evidence": True}],
        )
        warning = build_evidence_gate_warning(state)
        assert "EVIDENCE GATE" in warning
        assert "ACTION REQUIRED" in warning

    def test_no_warning_when_legit(self):
        state = LoopState(
            backtest_results=[{"claims_ready_without_evidence": False}],
        )
        assert build_evidence_gate_warning(state) == ""
