import json

from evals.run_domain_admission_acceptance import (
    build_report,
    validate_manifest,
)


def case(sample_id: str, sha256: str, decision: str) -> dict:
    run = {
        "decision": decision,
        "decision_code": (
            "positive_evidence_quorum"
            if decision == "accepted"
            else "insufficient_independent_regions"
        ),
        "region_ids": [f"{sample_id}-region-1", f"{sample_id}-region-2"],
        "duration_ms": 10,
    }
    return {
        "sample_id": sample_id,
        "sha256": sha256,
        "runs": [dict(run) for _ in range(3)],
    }


def test_private_report_uses_only_sample_ids_and_hashes() -> None:
    report = build_report(
        positive_cases=[case("POS-01", "a" * 64, "accepted")],
        negative_cases=[case("NEG-01", "b" * 64, "review_required")],
        metamorphic_cases=[],
        expected_positive=1,
        expected_negative=1,
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert "private-book.pdf" not in serialized
    assert report["negative_cases"][0]["sample_id"] == "NEG-01"
    assert report["negative_cases"][0]["sha256"] == "b" * 64


def test_release_gate_requires_all_59_positive_and_13_negative_cases() -> None:
    report = build_report(
        positive_cases=[
            case(f"POS-{index:02d}", f"{index:064x}", "accepted")
            for index in range(1, 60)
        ],
        negative_cases=[
            case(
                f"NEG-{index:02d}",
                f"{index + 100:064x}",
                "review_required",
            )
            for index in range(1, 14)
        ],
        metamorphic_cases=[
            case(
                f"META-{index:02d}",
                f"{index + 200:064x}",
                "review_required",
            )
            for index in range(1, 37)
        ],
        expected_positive=59,
        expected_negative=13,
    )

    assert report["release_gate"] == "PASS"
    assert report["metrics"]["positive_accepted"] == 59
    assert report["metrics"]["negative_accepted"] == 0
    assert report["metrics"]["negative_review_required"] == 13
    assert report["metrics"]["metamorphic_total"] == 36


def test_repeatability_mismatch_fails_release_gate() -> None:
    unstable = case("POS-01", "a" * 64, "accepted")
    unstable["runs"][1]["decision_code"] = "different_code"
    report = build_report(
        positive_cases=[unstable],
        negative_cases=[case("NEG-01", "b" * 64, "review_required")],
        metamorphic_cases=[],
        expected_positive=1,
        expected_negative=1,
    )

    assert report["release_gate"] == "FAIL"
    assert report["metrics"]["repeatability_failures"] == 1


def test_report_preserves_reproducibility_metadata() -> None:
    report = build_report(
        positive_cases=[case("POS-01", "a" * 64, "accepted")],
        negative_cases=[case("NEG-01", "b" * 64, "review_required")],
        metamorphic_cases=[],
        expected_positive=1,
        expected_negative=1,
        classifier_version="positive-admission-v2",
        embedding_model_id="BAAI/bge-m3",
        config_fingerprint="f" * 64,
    )

    assert report["classifier_version"] == "positive-admission-v2"
    assert report["embedding_model_id"] == "BAAI/bge-m3"
    assert report["config_fingerprint"] == "f" * 64


def test_manifest_separates_private_and_database_negative_samples() -> None:
    manifest = {
        "positive": [
            {
                "sample_id": f"POS-{index:02d}",
                "sha256": f"{index:064x}",
                "document_id": f"00000000-0000-0000-0000-{index:012d}",
            }
            for index in range(1, 60)
        ],
        "negative": [
            {
                "sample_id": f"NEG-{index:02d}",
                "sha256": f"{index + 100:064x}",
            }
            for index in range(1, 13)
        ]
        + [
            {
                "sample_id": "NEG-13",
                "sha256": "f" * 64,
                "document_id": "10000000-0000-0000-0000-000000000013",
            }
        ],
    }

    summary = validate_manifest(manifest)

    assert summary == {
        "positive": 59,
        "private_negative": 12,
        "database_negative": 1,
        "negative": 13,
    }
