from paper_rag.models.job import JobState


def test_phase35_job_states_cover_automatic_indexing_pipeline() -> None:
    assert [
        JobState.PARSING.value,
        JobState.CHUNKING.value,
        JobState.EMBEDDING.value,
        JobState.INDEXING.value,
        JobState.COMPLETED.value,
    ] == ["parsing", "chunking", "embedding", "indexing", "completed"]
