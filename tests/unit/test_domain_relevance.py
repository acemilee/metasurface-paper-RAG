import unicodedata

import pytest

from paper_rag.config import Settings
from paper_rag.models.document import DomainStatus
from paper_rag.services.domain_admission import AdmissionPage, evaluate_domain_admission


class PositiveAdmissionProvider:
    model_id = "positive-admission-stub"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0]
            if any(
                marker in unicodedata.normalize("NFKC", text).lower()
                for marker in (
                    "domain-positive",
                    "metasurface",
                    "metamaterial",
                    "frequency-selective",
                    "frequency selective",
                    "fss",
                    "tunable radar absorber",
                    "tunable absorber",
                    "wideband tunable",
                    "repcm",
                    "pcm",
                    "mma",
                    "surface conductivity",
                    "超表面",
                    "超材料",
                    "频率选择表面",
                )
            )
            else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FailingAdmissionProvider(PositiveAdmissionProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise TimeoutError("embedding timeout")


class EmptyAdmissionProvider(PositiveAdmissionProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []


class CountingAdmissionProvider(PositiveAdmissionProvider):
    model_id = "counting-prototype-stub"

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return super().embed_documents(texts)


class WrongDimensionAdmissionProvider(PositiveAdmissionProvider):
    model_id = "wrong-dimension-admission-stub"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _text in texts]


class NonFiniteAdmissionProvider(PositiveAdmissionProvider):
    model_id = "non-finite-admission-stub"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float("nan"), 0.0] for _text in texts]


def admission_settings() -> Settings:
    return Settings(
        embedding_provider="hash",
        domain_region_min_chars=80,
        domain_region_target_chars=600,
        domain_region_max_count=12,
        domain_min_evidence_regions=2,
        domain_semantic_support_min=0.48,
        domain_parse_quality_min=0.65,
    )


def test_two_independent_metasurface_regions_are_accepted() -> None:
    pages = [
        AdmissionPage(
            1,
            "domain-positive A graphene metasurface controls electromagnetic "
            "absorption and polarization. " * 3,
            0.98,
            None,
        ),
        AdmissionPage(
            4,
            "domain-positive The metasurface unit cells tune surface impedance "
            "and terahertz reflection phase. " * 3,
            0.97,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.ACCEPTED
    assert result.decision_code == "positive_evidence_quorum"
    assert len(result.evidence_regions) == 2


def test_single_high_scoring_region_requires_review() -> None:
    pages = [
        AdmissionPage(
            1,
            "domain-positive A metasurface controls electromagnetic reflection "
            "and phase. " * 5,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "Administrative procedures and employee schedules. " * 8,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.decision_code == "insufficient_independent_regions"


def test_reference_only_domain_mentions_require_review() -> None:
    pages = [
        AdmissionPage(
            1,
            "This handbook covers general scientific writing and citation "
            "management. " * 4,
            0.98,
            None,
        ),
        AdmissionPage(
            9,
            "References\n"
            "[1] A domain-positive metasurface controls electromagnetic absorption.\n"
            "[2] Tunable metasurface phase control. " * 4,
            0.98,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.decision_code == "reference_only_evidence"


def test_duplicated_positive_paragraph_counts_once() -> None:
    repeated = (
        "domain-positive A graphene metasurface controls electromagnetic absorption. "
        * 5
    )
    pages = [
        AdmissionPage(2, repeated, 0.98, None),
        AdmissionPage(8, repeated, 0.98, None),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert len(result.evidence_regions) <= 1


def test_single_page_short_paper_can_supply_two_independent_windows() -> None:
    text = (
        "domain-positive We design a metasurface whose unit cells control "
        "electromagnetic absorption. " * 8
        + "\n\nMethods\n"
        + "domain-positive Metasurface resonators tune surface impedance and "
        "terahertz reflection phase. " * 8
    )

    result = evaluate_domain_admission(
        [AdmissionPage(1, text, 0.99, None)],
        PositiveAdmissionProvider(),
        admission_settings(),
    )

    assert result.decision == DomainStatus.ACCEPTED
    assert len({region.region_id for region in result.evidence_regions}) >= 2


def test_region_cap_keeps_distributed_body_evidence() -> None:
    settings = admission_settings().model_copy(
        update={"domain_region_max_count": 4}
    )
    pages = [
        AdmissionPage(
            page_number,
            (
                "domain-positive A metasurface controls electromagnetic "
                "absorption. " * 5
                if page_number == 1
                else "domain-positive Metasurface unit cells tune surface "
                "impedance and terahertz reflection phase. " * 5
                if page_number == 80
                else "General supporting discussion without an admission "
                "relationship. " * 5
            ),
            0.98,
            None,
        )
        for page_number in range(1, 81)
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), settings
    )

    assert result.decision == DomainStatus.ACCEPTED
    evidence_pages = {
        page
        for region in result.evidence_regions
        for page in region.page_numbers
    }
    assert {1, 80} <= evidence_pages


def test_repeated_page_headers_cannot_become_domain_evidence() -> None:
    header = "Metasurface controls electromagnetic absorption"
    pages = [
        AdmissionPage(
            page_number,
            f"{header}\n"
            + (
                f"Administrative section {page_number} covers schedules, forms, "
                "approvals, and general office procedures. " * 3
            ),
            0.98,
            None,
        )
        for page_number in range(1, 5)
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.evidence_regions == ()


def test_general_materials_and_frequency_language_requires_review() -> None:
    pages = [
        AdmissionPage(
            1,
            "domain-positive This graphene material model reports "
            "frequency-dependent mechanical properties. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            3,
            "domain-positive The structure is simulated and experimental data "
            "agree with the model. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert "domain_identity" in result.failed_requirements


def test_single_metasurface_keyword_injection_requires_review() -> None:
    pages = [
        AdmissionPage(
            1,
            "domain-positive Employee scheduling and office procedures. "
            "metasurface " * 6,
            0.99,
            None,
        ),
        AdmissionPage(
            4,
            "domain-positive Payroll approval and administrative reporting. " * 6,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert "domain_relationship" in result.failed_requirements


def test_embedding_failure_is_fail_closed() -> None:
    pages = [
        AdmissionPage(
            1,
            "A metasurface controls electromagnetic absorption. " * 5,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "Metasurface unit cells tune terahertz reflection phase. " * 5,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, FailingAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.decision_code == "gate_dependency_unavailable"
    assert result.evidence_regions == ()


def test_empty_embedding_response_is_fail_closed() -> None:
    pages = [
        AdmissionPage(
            1,
            "A metasurface controls electromagnetic absorption. " * 5,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "Metasurface unit cells tune terahertz reflection phase. " * 5,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, EmptyAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.decision_code == "gate_dependency_unavailable"


def test_invalid_page_number_is_fail_closed_internal_error() -> None:
    result = evaluate_domain_admission(
        [
            AdmissionPage(
                0,
                "A metasurface controls electromagnetic absorption. " * 5,
                0.99,
                None,
            )
        ],
        PositiveAdmissionProvider(),
        admission_settings(),
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.decision_code == "gate_internal_error"
    assert result.evidence_regions == ()


def test_low_parse_quality_cannot_be_rescued_by_domain_language() -> None:
    pages = [
        AdmissionPage(
            1,
            "A metasurface controls electromagnetic absorption. " * 5,
            0.40,
            0.40,
        ),
        AdmissionPage(
            2,
            "Metasurface unit cells tune terahertz reflection phase. " * 5,
            0.40,
            0.40,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.decision_code == "insufficient_parse_evidence"
    assert result.evidence_regions == ()
    assert result.parse_quality == 0.40


def test_positive_prototypes_are_embedded_once_per_model() -> None:
    provider = CountingAdmissionProvider()
    pages = [
        AdmissionPage(
            1,
            "A metasurface controls electromagnetic absorption. " * 5,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "Metasurface unit cells tune terahertz reflection phase. " * 5,
            0.99,
            None,
        ),
    ]

    first = evaluate_domain_admission(pages, provider, admission_settings())
    second = evaluate_domain_admission(pages, provider, admission_settings())

    assert first.decision == second.decision == DomainStatus.ACCEPTED
    assert len(provider.embedded_texts) == 8


@pytest.mark.parametrize(
    "provider",
    [WrongDimensionAdmissionProvider(), NonFiniteAdmissionProvider()],
)
def test_invalid_embedding_vectors_are_fail_closed(provider) -> None:
    pages = [
        AdmissionPage(
            1,
            "A metasurface controls electromagnetic absorption. " * 5,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "Metasurface unit cells tune terahertz reflection phase. " * 5,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(pages, provider, admission_settings())

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.decision_code == "gate_dependency_unavailable"


def test_fss_radiator_and_reflector_regions_are_domain_evidence() -> None:
    pages = [
        AdmissionPage(
            1,
            "A frequency-selective surface (FSS) operates as a low-band radiator "
            "and high-band reflector at 0.8 and 3.5 GHz. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            3,
            "The FSS unit cells provide band-stop characteristics and control "
            "the radiation pattern in the microwave bands. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.ACCEPTED


def test_engineered_tunable_radar_absorber_is_domain_evidence() -> None:
    pages = [
        AdmissionPage(
            1,
            "A wideband tunable radar absorber controls electromagnetic wave "
            "absorption through an impedance-matched patterned surface at microwave "
            "frequencies. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            4,
            "The tunable absorber uses resonant unit cells and adjustable surface "
            "impedance to change microwave reflectivity. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.ACCEPTED


def test_acoustic_tweezers_transducer_requires_review() -> None:
    pages = [
        AdmissionPage(
            1,
            "Spiraling transducer acoustical tweezers generate focused acoustic "
            "vortices for selective particle manipulation. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "The transducer vibration field reflects an acoustic wave and traps "
            "cells near the vortex center. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert "domain_identity" in result.failed_requirements


def test_duplicated_identity_paragraph_cannot_unlock_generic_em_relation() -> None:
    generic_relation = (
        "domain-positive Resonant unit cells tune terahertz reflection phase "
        "through adjustable surface impedance. " * 5
    )
    injected = (
        "A metasurface controls electromagnetic absorption and reflection "
        "through designed unit cells."
    )
    text = generic_relation + "\n\n" + injected + "\n\n" + injected

    result = evaluate_domain_admission(
        [AdmissionPage(1, text, 0.99, None)],
        PositiveAdmissionProvider(),
        admission_settings(),
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert "domain_identity_regions" in result.failed_requirements


def test_document_scoped_metasurface_aliases_inherit_domain_identity() -> None:
    pages = [
        AdmissionPage(
            1,
            "A reconfigurable polarization-conversion metasurface controls the "
            "reflection phase at microwave frequencies. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "The RePCM changes the co-polarized reflection coefficient in its "
            "conversion mode. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            3,
            "Surface currents on the PCM rotate polarization and produce the "
            "cross-polarized reflected wave. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.ACCEPTED


def test_nfkc_aliases_inherit_a_prior_explicit_domain_identity() -> None:
    pages = [
        AdmissionPage(
            1,
            "A reconfigurable polarization-conversion metasurface controls "
            "microwave reflection and polarization. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "The fullwidth ＰＣＭ changes the reflected electromagnetic wave and "
            "its co-polarized reflection coefficient. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            3,
            "Surface currents on the ReＰＣＭ produce cross-polarized microwave "
            "reflection in conversion mode. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.ACCEPTED


def test_domain_aliases_adjacent_to_cjk_text_inherit_identity() -> None:
    pages = [
        AdmissionPage(
            1,
            "可重构极化转换超表面能够调控微波反射与偏振。" * 8,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "可重构ＰＣＭ的表面电流产生交叉偏振反射电磁波。" * 8,
            0.99,
            None,
        ),
        AdmissionPage(
            3,
            "ReＰＣＭ在转换模式下调控反射相位与微波偏振。" * 8,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.ACCEPTED


def test_aliases_before_an_explicit_domain_identity_do_not_inherit_it() -> None:
    pages = [
        AdmissionPage(
            1,
            "The PCM changes the co-polarized reflection coefficient and "
            "rotates the incident electromagnetic wave. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "Surface currents on the MMA tune impedance and microwave "
            "absorption over the measured frequency band. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            3,
            "A metasurface is mentioned only later as a possible comparison "
            "for electromagnetic reflection control. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert "domain_identity_regions" in result.failed_requirements


def test_isolated_aliases_cannot_form_domain_identity() -> None:
    pages = [
        AdmissionPage(
            1,
            "The PCM changes electromagnetic reflection and polarization over "
            "the measured microwave frequency range. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            2,
            "The MMA tunes surface impedance and absorption through resonant "
            "current distributions. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert "domain_identity" in result.failed_requirements


def test_graphene_surface_conductivity_wave_model_is_domain_evidence() -> None:
    pages = [
        AdmissionPage(
            1,
            "A surface conductivity model of graphene gives electromagnetic "
            "plane-wave reflection and transmission through dyadic Green functions. "
            * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            3,
            "The graphene sheet surface conductivity controls guided TE and TM "
            "surface-wave propagation at infrared frequencies. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.ACCEPTED


def test_engineered_em_absorber_relationships_form_domain_evidence() -> None:
    pages = [
        AdmissionPage(
            1,
            "domain-positive A wide band polarization-insensitive absorber uses "
            "a symmetric resistive unit cell for microwave absorption. " * 4,
            0.99,
            None,
        ),
        AdmissionPage(
            3,
            "domain-positive Surface currents in the designed absorber create "
            "electromagnetic resonance and high absorption at 15 GHz. " * 4,
            0.99,
            None,
        ),
    ]

    result = evaluate_domain_admission(
        pages, PositiveAdmissionProvider(), admission_settings()
    )

    assert result.decision == DomainStatus.ACCEPTED
