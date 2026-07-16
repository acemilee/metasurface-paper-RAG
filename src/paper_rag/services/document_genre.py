from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import Lock

from paper_rag.models.document import DocumentGenre
from paper_rag.services.embeddings import EmbeddingProvider


GENRE_CLASSIFIER_VERSION = "document-genre-v2"
GENRE_ANCHORS = {
    DocumentGenre.RESEARCH_PAPER: (
        "Original research article reporting one specific theoretical, numerical, fabricated, measured, or experimental study with methods and results.",
        "原创研究论文，报告一项具体理论、仿真、制备、测量或实验研究，包含方法和结果。",
    ),
    DocumentGenre.REVIEW_PAPER: (
        "Review or survey article systematically classifying prior studies, comparing approaches, and discussing challenges, perspectives, or trends.",
        "综述或调研文章，系统分类已有研究、比较技术路线并讨论挑战、展望或趋势。",
    ),
    DocumentGenre.THESIS: (
        "University master's thesis or doctoral dissertation submitted for a degree with advisor, degree declaration, chapters, acknowledgements, and original contributions.",
        "大学硕士或博士学位论文，包含导师、学位声明、分章、致谢和原创贡献。",
    ),
    DocumentGenre.CONFERENCE_PAPER: (
        "Conference proceedings paper published for a named conference, symposium, workshop, or technical session.",
        "会议论文集中的会议、研讨会、workshop 或技术分会论文。",
    ),
}
SEGMENT_BUDGETS = {
    "front_matter": 2500,
    "abstract": 2500,
    "section_headings": 2000,
    "method_experiment": 2500,
    "conclusion": 2000,
    "tail_matter": 1500,
}
SEGMENT_WEIGHTS = {
    "front_matter": 0.30,
    "section_headings": 0.25,
    "abstract": 0.20,
    "method_experiment": 0.15,
    "conclusion": 0.10,
}

_genre_anchor_cache: dict[tuple[str, int], tuple[list[DocumentGenre], list[list[float]]]] = {}
_genre_anchor_lock = Lock()


@dataclass(frozen=True)
class DocumentGenreResult:
    genre: DocumentGenre
    score: float
    margin: float
    decision_source: str
    scores: dict[str, float]
    evidence: list[dict] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    classifier_version: str = GENRE_CLASSIFIER_VERSION


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _anchor_vectors(provider: EmbeddingProvider) -> tuple[list[DocumentGenre], list[list[float]]]:
    key = (provider.model_id, provider.dimension)
    with _genre_anchor_lock:
        cached = _genre_anchor_cache.get(key)
    if cached is not None:
        return cached
    genres = []
    texts = []
    for genre, anchors in GENRE_ANCHORS.items():
        for anchor in anchors:
            genres.append(genre)
            texts.append(anchor)
    result = (genres, provider.embed_documents(texts))
    with _genre_anchor_lock:
        return _genre_anchor_cache.setdefault(key, result)


def _compact(text: str) -> str:
    return " ".join(text.split())


def _before_references(text: str) -> str:
    match = re.search(r"(?im)^\s*(?:references|bibliography|参考文献)\s*$", text)
    return text[:match.start()] if match else text


def _extract_abstract(page_texts: list[str]) -> str:
    head = "\n".join(page_texts[:3])
    match = re.search(
        r"(?is)(?:^|\n)\s*(?:abstract|摘要)\s*[:：]?\s*(.+?)(?=\n\s*(?:1\.?\s*)?(?:introduction|引言|绪论)\b)",
        head,
    )
    return _compact(match.group(1)) if match else ""


def _extract_headings(page_texts: list[str]) -> str:
    headings = []
    for page_number, text in enumerate(page_texts, start=1):
        for line in text.splitlines():
            compact = _compact(line)
            if not 2 <= len(compact) <= 100:
                continue
            if re.match(r"^(?:\d+(?:\.\d+)*\.?\s+|[一二三四五六七八九十]+[、.])", compact) or compact.lower() in {
                "abstract", "introduction", "methods", "materials and methods", "results", "discussion",
                "conclusion", "conclusions", "data availability", "acknowledgements", "references",
                "摘要", "引言", "绪论", "研究方法", "实验", "结果", "讨论", "结论", "总结与展望", "致谢", "参考文献",
            }:
                headings.append(f"p{page_number}: {compact}")
    return "\n".join(dict.fromkeys(headings))


def _pages_matching(page_texts: list[str], pattern: str) -> str:
    selected = []
    regex = re.compile(pattern, re.I)
    for page_number, text in enumerate(page_texts, start=1):
        body = _before_references(text)
        if regex.search(body):
            selected.append(f"Page {page_number}: {_compact(body)}")
    return "\n".join(selected)


def build_genre_segments(filename: str, page_texts: list[str]) -> dict[str, str]:
    first_page = page_texts[0] if page_texts else ""
    tail = "\n".join(_before_references(text) for text in page_texts[-2:])
    segments = {
        "front_matter": f"Filename: {filename}\nPage count: {len(page_texts)}\n{_compact(first_page)}",
        "abstract": _extract_abstract(page_texts),
        "section_headings": _extract_headings(page_texts),
        "method_experiment": _pages_matching(
            page_texts,
            r"\b(method|experiment|fabricat|measur|simulation|structure design|prototype|setup)\w*\b|研究方法|实验|仿真|结构设计|样机|测试|测量",
        ),
        "conclusion": _pages_matching(page_texts, r"\bconclusions?\b|结论|总结与展望"),
        "tail_matter": _compact(tail),
    }
    return {name: text[:SEGMENT_BUDGETS[name]] for name, text in segments.items()}


def build_genre_sample(filename: str, page_texts: list[str], max_chars: int = 12000) -> str:
    segments = build_genre_segments(filename, page_texts)
    return "\n".join(f"[{name}]\n{text}" for name, text in segments.items())[:max_chars]


def _explicit_type_evidence(filename: str, page_texts: list[str]) -> dict[DocumentGenre, list[dict]]:
    first = _compact(page_texts[0] if page_texts else "")[:3000]
    degree_pages = _compact("\n".join(page_texts[:5]))[:8000]
    source = f"{filename}\n{first}"
    patterns = {
        DocumentGenre.REVIEW_PAPER: r"\breview article\b|\bsystematic review\b|\bsurvey article\b|\bresearch progress\b|综述文章|文献综述|研究进展",
        DocumentGenre.RESEARCH_PAPER: r"\bresearch article\b|\boriginal article\b|\boriginal research\b",
        DocumentGenre.CONFERENCE_PAPER: r"\bconference paper\b|\bproceedings of\b|\binternational conference on\b|\bjournal of physics:\s*conference series\b|会议论文集",
    }
    found: dict[DocumentGenre, list[dict]] = {}
    for genre, pattern in patterns.items():
        match = re.search(pattern, source, re.I)
        if match:
            found[genre] = [{
                "page": 1,
                "type": "explicit_publication_type",
                "text": match.group(0),
                "weight": 1.0,
            }]
    thesis_match = re.search(
        r"doctoral dissertation|master(?:'s)? thesis|博士学位论文|硕士学位论文|学位授予|指导教师",
        f"{filename}\n{degree_pages}",
        re.I,
    )
    if thesis_match:
        found[DocumentGenre.THESIS] = [{
            "page": 1,
            "type": "explicit_degree_type",
            "text": thesis_match.group(0),
            "weight": 1.0,
        }]
    return found


def _structure_evidence(page_texts: list[str]) -> dict[DocumentGenre, list[dict]]:
    body = _compact("\n".join(_before_references(text) for text in page_texts)).lower()
    markers = {
        DocumentGenre.RESEARCH_PAPER: (
            "in this work", "in this paper, we present", "in our paper, we have developed", "we propose",
            "we experimentally propose", "is proposed", "prototype is fabricated",
            "fabricated and measured", "experiment results", "measured results", "experimental demonstration",
            "structure design", "data availability", "simulation and experiment", "仿真与实验", "样机", "测试结果",
        ),
        DocumentGenre.REVIEW_PAPER: (
            "this review", "we review", "systematic review", "taxonomy", "research progress", "challenges and perspectives",
            "研究进展", "本文综述", "系统综述", "分类讨论", "挑战与展望",
        ),
        DocumentGenre.THESIS: (
            "doctoral dissertation", "master's thesis", "degree of", "指导教师", "学位论文", "攻读学位期间", "原创性声明",
        ),
        DocumentGenre.CONFERENCE_PAPER: (),
    }
    result: dict[DocumentGenre, list[dict]] = {}
    for genre, terms in markers.items():
        hits = [term for term in terms if term in body]
        if hits:
            result[genre] = [{"page": None, "type": "structure_marker", "text": term, "weight": 0.2} for term in hits[:6]]
    return result


def _semantic_scores(segments: dict[str, str], provider: EmbeddingProvider) -> dict[DocumentGenre, float]:
    names = [name for name in SEGMENT_WEIGHTS if segments.get(name)]
    if not names:
        return {genre: 0.0 for genre in GENRE_ANCHORS}
    segment_vectors = provider.embed_documents([segments[name] for name in names])
    genres, anchor_vectors = _anchor_vectors(provider)
    total_weight = sum(SEGMENT_WEIGHTS[name] for name in names)
    scores = {}
    for genre in GENRE_ANCHORS:
        weighted = 0.0
        for name, segment_vector in zip(names, segment_vectors, strict=True):
            similarity = max(
                _dot(segment_vector, anchor)
                for current_genre, anchor in zip(genres, anchor_vectors, strict=True)
                if current_genre == genre
            )
            weighted += SEGMENT_WEIGHTS[name] * similarity
        scores[genre] = weighted / total_weight
    return scores


def classify_document_genre(
    filename: str,
    page_texts: list[str],
    provider: EmbeddingProvider,
) -> DocumentGenreResult:
    explicit = _explicit_type_evidence(filename, page_texts)
    semantic = _semantic_scores(build_genre_segments(filename, page_texts), provider)
    raw_scores = {genre.value: score for genre, score in semantic.items()}
    if len(explicit) == 1:
        genre = next(iter(explicit))
        return DocumentGenreResult(genre, 0.98, 1.0, "explicit_publication_type", raw_scores, explicit[genre])
    if len(explicit) > 1:
        conflicts = [f"同时检测到显式类型：{', '.join(genre.value for genre in explicit)}"]
        evidence = [item for items in explicit.values() for item in items]
        return DocumentGenreResult(DocumentGenre.UNCLASSIFIED, 0.30, 0.0, "explicit_type_conflict", raw_scores, evidence, conflicts)

    structure = _structure_evidence(page_texts)
    structure_counts = {genre: len(items) for genre, items in structure.items()}
    ranked_structure = sorted(structure_counts.items(), key=lambda item: item[1], reverse=True)
    if ranked_structure and ranked_structure[0][1] >= 2:
        top_genre, top_count = ranked_structure[0]
        second_count = ranked_structure[1][1] if len(ranked_structure) > 1 else 0
        if top_count - second_count >= 1:
            confidence = min(0.90, 0.72 + top_count * 0.03)
            return DocumentGenreResult(top_genre, confidence, (top_count - second_count) / 10, "document_structure", raw_scores, structure[top_genre])

    ranked = sorted(semantic.items(), key=lambda item: item[1], reverse=True)
    genre, top_score = ranked[0]
    margin = top_score - ranked[1][1]
    evidence = [item for items in structure.values() for item in items]
    if margin < 0.08:
        return DocumentGenreResult(DocumentGenre.UNCLASSIFIED, max(0.0, min(0.54, top_score)), margin, "low_margin", raw_scores, evidence)
    confidence = max(0.55, min(0.90, 0.55 + margin * 2))
    return DocumentGenreResult(genre, confidence, margin, "bge_segmented", raw_scores, evidence)
