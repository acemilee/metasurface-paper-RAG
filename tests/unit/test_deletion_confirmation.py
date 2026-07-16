from uuid import uuid4

from paper_rag.services.deletion_confirmation import DeletionConfirmationStore


def test_confirmation_requires_exact_filename_and_is_single_use() -> None:
    store = DeletionConfirmationStore()
    document_id = uuid4()
    token = store.issue(document_id, "paper.pdf", 60)

    assert not store.consume(token, document_id, "Paper.pdf")
    assert not store.consume(token, document_id, "paper.pdf")


def test_confirmation_is_bound_to_document() -> None:
    store = DeletionConfirmationStore()
    token = store.issue(uuid4(), "paper.pdf", 60)

    assert not store.consume(token, uuid4(), "paper.pdf")


def test_expired_confirmation_is_rejected() -> None:
    store = DeletionConfirmationStore()
    document_id = uuid4()
    token = store.issue(document_id, "paper.pdf", -1)

    assert not store.consume(token, document_id, "paper.pdf")


def test_confirmation_accepts_exact_unicode_filename_once() -> None:
    store = DeletionConfirmationStore()
    document_id = uuid4()
    filename = "从零开始学MT4编程.pdf"
    token = store.issue(document_id, filename, 60)

    assert store.consume(token, document_id, filename)
    assert not store.consume(token, document_id, filename)
