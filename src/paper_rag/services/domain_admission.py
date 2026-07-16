from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import ceil, isfinite
from threading import Lock
from time import perf_counter

from paper_rag.config import Settings
from paper_rag.models.document import DomainStatus
from paper_rag.services.embeddings import EmbeddingProvider

CLASSIFIER_VERSION = "positive-admission-v2"
PROTOTYPE_VERSION = "metasurface-positive-prototypes-v1"


class EmbeddingContractError(RuntimeError):
    pass


_prototype_cache: dict[tuple[str, int, str], list[list[float]]] = {}
_prototype_cache_lock = Lock()

POSITIVE_PROTOTYPES = (
    "Metasurface unit cells control electromagnetic absorption, reflection, "
    "transmission, phase, polarization or beam direction.",
    "Tunable artificial electromagnetic surfaces use resonant structures, "
    "surface impedance or reconfigurable materials at microwave, terahertz "
    "or optical frequencies.",
    "超表面单元结构通过谐振或表面阻抗调控电磁波的吸收、反射、透射、相位、偏振或波束。",
    "可调谐人工电磁表面在微波、太赫兹或光学频段利用材料或器件改变电磁响应。",
)

_CONCEPT_PATTERNS = {
    "domain_identity": re.compile(
        r"\bmeta-?surfaces?\b|\bmetamaterials?\b|"
        r"\bfrequency[- ]selective (?:surfaces?|absorbers?|structures?)\b|"
        r"\br?fss\b|"
        r"\b(?:wideband|broadband|tunable|radar|microwave|electromagnetic)"
        r"(?:[- ]+(?:wideband|broadband|tunable|radar|microwave|electromagnetic))*"
        r"[- ]+absorbers?\b|"
        r"\bartificial impedance surfaces?\b|"
        r"\bsurface conductivity(?: model)? of graphene\b|"
        r"\bgraphene(?: sheet)? surface conductivity\b|"
        r"超表面|超材料|频率选择表面|频选表面|人工阻抗表面|"
        r"石墨烯.{0,16}表面电导|表面电导.{0,16}石墨烯|"
        r"(?:宽带|可调谐?|雷达|微波|电磁)(?:宽带|可调谐?|雷达|微波|电磁)*吸波(?:器|体)",
        re.IGNORECASE,
    ),
    "em_function": re.compile(
        r"\babsorption\b|\breflection\b|\btransmission\b|"
        r"\bpolarization\b|\bphase(?: control)?\b|"
        r"\babsorbers?\b|\breflect(?:ivity|ors?)\b|"
        r"\bradiat(?:ion|ors?|ing)\b|\bpassbands?\b|\bband[- ]stop\b|"
        r"\b(?:guided |surface[- ])?wave propagation\b|"
        r"\bbeam(?:forming| steering)\b|"
        r"吸收|吸波|反射|透射|偏振|相位调控|表面波传播|导波|"
        r"通带|阻带|带通|"
        r"波束(?:赋形|偏转|调控)",
        re.IGNORECASE,
    ),
    "implementation": re.compile(
        r"\bunit cells?\b|\bperiodic arrays?\b|\bsurface impedance\b|"
        r"\bresonan(?:ce|t)\b|\breconfigurable\b|\btunable\b|"
        r"单元结构|周期阵列|表面阻抗|谐振|可重构|可调谐",
        re.IGNORECASE,
    ),
    "operating_context": re.compile(
        r"\bmicrowave\b|\bmillimeter[- ]wave\b|\bterahertz\b|"
        r"\binfrared\b|\boptical\b|\belectromagnetic\b|"
        r"\b(?:mhz|ghz|thz)\b|"
        r"微波|毫米波|太赫兹|红外|光学|电磁",
        re.IGNORECASE,
    ),
}
_REFERENCE_HEADING = re.compile(
    r"(?im)^\s*(references|bibliography|参考文献)\s*$"
)
_IDENTITY_ALIAS_PATTERNS = (
    re.compile(
        r"(?<![a-z0-9])(?:re[- ]?pcm|pcm)(?![a-z0-9])", re.IGNORECASE
    ),
    re.compile(r"(?<![a-z0-9])mmas?(?![a-z0-9])", re.IGNORECASE),
)
_ABSORBER_TERM = re.compile(r"\babsorbers?\b", re.IGNORECASE)


def _concept_families(
    text: str, identity_aliases: tuple[re.Pattern[str], ...] = ()
) -> set[str]:
    text = unicodedata.normalize("NFKC", text)
    families = {
        family
        for family, pattern in _CONCEPT_PATTERNS.items()
        if pattern.search(text)
    }
    if (
        _ABSORBER_TERM.search(text)
        and {"em_function", "implementation", "operating_context"}
        <= families
    ):
        families.add("domain_identity")
    if any(pattern.search(text) for pattern in identity_aliases):
        families.add("domain_identity")
    return families


def _identity_aliases_before(
    pages: list[AdmissionPage], page: AdmissionPage, start_offset: int
) -> tuple[re.Pattern[str], ...]:
    prefix = "\n".join(
        candidate.text
        for candidate in sorted(pages, key=lambda item: item.page_number)
        if candidate.page_number < page.page_number
    )
    prefix = f"{prefix}\n{page.text[:start_offset]}"
    if "domain_identity" in _concept_families(prefix):
        return _IDENTITY_ALIAS_PATTERNS
    return ()


def _relations(families: set[str]) -> tuple[str, ...]:
    relations: list[str] = []
    if {"domain_identity", "em_function"} <= families:
        relations.append("domain_structure_controls_em_wave")
    if {"implementation", "em_function", "operating_context"} <= families:
        relations.append("implementation_controls_response_in_context")
    return tuple(relations)


@dataclass(frozen=True)
class AdmissionPage:
    page_number: int
    text: str
    quality_score: float
    ocr_confidence: float | None


@dataclass(frozen=True)
class EvidenceRegion:
    region_id: str
    page_numbers: tuple[int, ...]
    section_role: str
    concept_families: tuple[str, ...]
    relations: tuple[str, ...]
    semantic_support: float
    content_hash: str
    excerpt: str


@dataclass(frozen=True)
class DomainAdmissionResult:
    decision: DomainStatus
    decision_code: str
    evidence_regions: tuple[EvidenceRegion, ...]
    passed_requirements: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    parse_quality: float
    classifier_version: str
    embedding_model_id: str
    config_fingerprint: str
    duration_ms: int
    evaluated_at: datetime


def _result(
    *,
    started: float,
    pages: list[AdmissionPage],
    provider: EmbeddingProvider,
    settings: Settings,
    decision: DomainStatus,
    decision_code: str,
    evidence_regions: tuple[EvidenceRegion, ...] = (),
    passed_requirements: tuple[str, ...] = (),
    failed_requirements: tuple[str, ...] = (),
) -> DomainAdmissionResult:
    return DomainAdmissionResult(
        decision=decision,
        decision_code=decision_code,
        evidence_regions=evidence_regions,
        passed_requirements=passed_requirements,
        failed_requirements=failed_requirements,
        parse_quality=min((page.quality_score for page in pages), default=0.0),
        classifier_version=CLASSIFIER_VERSION,
        embedding_model_id=provider.model_id,
        config_fingerprint=_config_fingerprint(settings),
        duration_ms=round((perf_counter() - started) * 1000),
        evaluated_at=datetime.now().astimezone(),
    )


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _validate_vectors(
    vectors: list[list[float]], expected_count: int, provider: EmbeddingProvider
) -> None:
    if len(vectors) != expected_count:
        raise EmbeddingContractError(
            f"embedding count mismatch: expected={expected_count} actual={len(vectors)}"
        )
    if any(
        len(vector) != provider.dimension
        or any(not isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise EmbeddingContractError("embedding vector metadata is invalid")


def _get_prototype_vectors(provider: EmbeddingProvider) -> list[list[float]]:
    cache_key = (provider.model_id, provider.dimension, PROTOTYPE_VERSION)
    with _prototype_cache_lock:
        cached = _prototype_cache.get(cache_key)
        if cached is not None:
            return cached
        vectors = provider.embed_documents(list(POSITIVE_PROTOTYPES))
        _validate_vectors(vectors, len(POSITIVE_PROTOTYPES), provider)
        _prototype_cache[cache_key] = vectors
        return vectors


def _config_fingerprint(settings: Settings) -> str:
    payload = {
        "domain_gate_max_retries": settings.domain_gate_max_retries,
        "domain_gate_safe_mode": settings.domain_gate_safe_mode,
        "domain_min_evidence_regions": settings.domain_min_evidence_regions,
        "domain_parse_quality_min": settings.domain_parse_quality_min,
        "domain_region_max_count": settings.domain_region_max_count,
        "domain_region_min_chars": settings.domain_region_min_chars,
        "domain_region_target_chars": settings.domain_region_target_chars,
        "domain_semantic_support_min": settings.domain_semantic_support_min,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate_pages(
    pages: list[AdmissionPage], settings: Settings
) -> list[AdmissionPage]:
    eligible = [
        page
        for page in sorted(pages, key=lambda item: item.page_number)
        if page.page_number >= 1
        and page.quality_score >= settings.domain_parse_quality_min
        and (
            page.ocr_confidence is None
            or page.ocr_confidence >= settings.domain_parse_quality_min
        )
        and len(" ".join(page.text.split())) >= settings.domain_region_min_chars
    ]
    limit = settings.domain_region_max_count
    if len(eligible) <= limit:
        return eligible
    if limit <= 1:
        return eligible[:1]
    indexes = [
        round(position * (len(eligible) - 1) / (limit - 1))
        for position in range(limit)
    ]
    return [eligible[index] for index in dict.fromkeys(indexes)]


def _strip_reference_suffix(text: str) -> tuple[str, str | None]:
    match = _REFERENCE_HEADING.search(text)
    if match is None:
        return text, None
    return text[: match.start()], text[match.start() :]


def _normalize_line(line: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", line).casefold().split())


def _strip_repeated_page_edges(pages: list[AdmissionPage]) -> list[AdmissionPage]:
    if len(pages) < 2:
        return pages
    edge_counts: Counter[str] = Counter()
    page_lines: list[tuple[AdmissionPage, list[str], list[int]]] = []
    for page in pages:
        lines = page.text.splitlines()
        nonempty_indexes = [
            index for index, line in enumerate(lines) if line.strip()
        ]
        page_lines.append((page, lines, nonempty_indexes))
        edge_indexes = nonempty_indexes[:2] + nonempty_indexes[-2:]
        edge_lines = [lines[index] for index in edge_indexes]
        edge_counts.update(set(_normalize_line(line) for line in edge_lines))
    minimum_count = max(2, ceil(len(pages) / 2))
    repeated = {
        line for line, count in edge_counts.items() if line and count >= minimum_count
    }
    cleaned: list[AdmissionPage] = []
    for page, lines, nonempty_indexes in page_lines:
        edge_indexes = set(nonempty_indexes[:2] + nonempty_indexes[-2:])
        retained = [
            line
            for index, line in enumerate(lines)
            if not (
                index in edge_indexes and _normalize_line(line) in repeated
            )
        ]
        cleaned.append(
            AdmissionPage(
                page.page_number,
                "\n".join(retained),
                page.quality_score,
                page.ocr_confidence,
            )
        )
    return cleaned


def _evaluate_regions(
    pages: list[AdmissionPage],
    provider: EmbeddingProvider,
    settings: Settings,
) -> list[EvidenceRegion]:
    candidates: list[tuple[AdmissionPage, int, str]] = []
    for page in _candidate_pages(pages, settings):
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", page.text)
            if paragraph.strip()
        ]
        windows = paragraphs if len(paragraphs) > 1 else [page.text]
        search_from = 0
        for text in windows:
            start_offset = page.text.find(text, search_from)
            if start_offset < 0:
                start_offset = search_from
            search_from = start_offset + len(text)
            identity_aliases = _identity_aliases_before(
                pages, page, start_offset
            )
            if (
                len(" ".join(text.split())) >= settings.domain_region_min_chars
                and _relations(_concept_families(text, identity_aliases))
            ):
                candidates.append((page, start_offset, text))
    unique_candidates: list[tuple[AdmissionPage, int, str, str]] = []
    seen_hashes: set[str] = set()
    for page, start_offset, text in candidates:
        normalized = " ".join(
            unicodedata.normalize("NFKC", text).casefold().split()
        )
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        unique_candidates.append((page, start_offset, text, content_hash))
    unique_candidates = unique_candidates[: settings.domain_region_max_count]
    if not unique_candidates:
        return []
    prototype_vectors = _get_prototype_vectors(provider)
    candidate_vectors = provider.embed_documents(
        [text for _page, _start, text, _hash in unique_candidates]
    )
    _validate_vectors(candidate_vectors, len(unique_candidates), provider)
    regions: list[EvidenceRegion] = []
    for (page, start_offset, text, content_hash), vector in zip(
        unique_candidates, candidate_vectors, strict=True
    ):
        support = max(_dot(vector, prototype) for prototype in prototype_vectors)
        if support < settings.domain_semantic_support_min:
            continue
        families = _concept_families(text, identity_aliases)
        regions.append(
            EvidenceRegion(
                region_id=(
                    f"page-{page.page_number}-offset-{start_offset}-"
                    f"{content_hash[:16]}"
                ),
                page_numbers=(page.page_number,),
                section_role="body",
                concept_families=tuple(sorted(families)),
                relations=_relations(families),
                semantic_support=support,
                content_hash=content_hash,
                excerpt=" ".join(text.split())[:240],
            )
        )
    return regions


def evaluate_domain_admission(
    pages: list[AdmissionPage],
    provider: EmbeddingProvider,
    settings: Settings,
) -> DomainAdmissionResult:
    started = perf_counter()
    if settings.domain_gate_safe_mode == "review_all":
        return _result(
            started=started,
            pages=pages,
            provider=provider,
            settings=settings,
            decision=DomainStatus.REVIEW_REQUIRED,
            decision_code="gate_safe_mode",
            failed_requirements=("safe_mode",),
        )
    if any(page.page_number < 1 for page in pages):
        return _result(
            started=started,
            pages=pages,
            provider=provider,
            settings=settings,
            decision=DomainStatus.REVIEW_REQUIRED,
            decision_code="gate_internal_error",
            failed_requirements=("page_contract",),
        )
    cleaned_pages: list[AdmissionPage] = []
    reference_only_evidence = False
    for page in _strip_repeated_page_edges(pages):
        body, reference_suffix = _strip_reference_suffix(page.text)
        if (
            reference_suffix
            and _relations(_concept_families(reference_suffix))
        ):
            reference_only_evidence = True
        cleaned_pages.append(
            AdmissionPage(
                page.page_number,
                body,
                page.quality_score,
                page.ocr_confidence,
            )
        )
    if not _candidate_pages(cleaned_pages, settings):
        return _result(
            started=started,
            pages=pages,
            provider=provider,
            settings=settings,
            decision=DomainStatus.REVIEW_REQUIRED,
            decision_code="insufficient_parse_evidence",
            failed_requirements=("parse_quality",),
        )
    document_text = "\n".join(page.text for page in cleaned_pages)
    regions: list[EvidenceRegion] | None = None
    for attempt in range(settings.domain_gate_max_retries + 1):
        try:
            regions = _evaluate_regions(cleaned_pages, provider, settings)
            break
        except (TimeoutError, EmbeddingContractError):
            if attempt == settings.domain_gate_max_retries:
                return _result(
                    started=started,
                    pages=pages,
                    provider=provider,
                    settings=settings,
                    decision=DomainStatus.REVIEW_REQUIRED,
                    decision_code="gate_dependency_unavailable",
                    failed_requirements=("embedding_provider",),
                )
    assert regions is not None
    document_families = _concept_families(document_text)
    has_domain_identity = "domain_identity" in document_families
    identity_region_count = sum(
        "domain_identity" in region.concept_families for region in regions
    )
    has_identity_quorum = (
        identity_region_count >= settings.domain_min_evidence_regions
    )
    accepted = (
        has_domain_identity
        and has_identity_quorum
        and len(regions) >= settings.domain_min_evidence_regions
    )
    decision_code = (
        "positive_evidence_quorum"
        if accepted
        else "reference_only_evidence"
        if reference_only_evidence and not regions
        else "missing_domain_relationship"
        if not has_domain_identity or not regions
        else "insufficient_independent_regions"
        if len(regions) < settings.domain_min_evidence_regions
        else "inconsistent_positive_evidence"
    )
    failed_requirements: list[str] = []
    if not has_domain_identity:
        failed_requirements.append("domain_identity")
    if not has_identity_quorum:
        failed_requirements.append("domain_identity_regions")
    if not regions:
        failed_requirements.append("domain_relationship")
    if len(regions) < settings.domain_min_evidence_regions:
        failed_requirements.append("independent_regions")
    return _result(
        started=started,
        pages=pages,
        provider=provider,
        settings=settings,
        decision=DomainStatus.ACCEPTED if accepted else DomainStatus.REVIEW_REQUIRED,
        decision_code=decision_code,
        evidence_regions=tuple(regions),
        passed_requirements=(
            (
                "domain_identity",
                "domain_identity_regions",
                "domain_relationship",
                "independent_regions",
            )
            if accepted
            else ()
        ),
        failed_requirements=tuple(failed_requirements),
    )
