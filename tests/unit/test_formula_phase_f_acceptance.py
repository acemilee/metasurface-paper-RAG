from evals.run_formula_phase_f_acceptance import run_acceptance


def test_formula_phase_f_runner_reports_automatic_and_manual_boundaries(tmp_path) -> None:
    result = run_acceptance(output_dir=tmp_path)

    assert result["parser"]["kubo_parts"] == ["1a", "1b", "1c"]
    assert result["parser"]["equation_2_not_truncated"] is True
    assert result["direct_answer"]["stable_runs"] == 20
    assert result["direct_answer"]["unique_signatures"] == 1
    assert result["production_backfill"]["status"] == "not_run"
    assert result["manual_review"]["status"] == "pending"
    assert result["manual_review"]["required_papers"] == 30
    assert result["manual_review"]["required_formulas"] == 100
