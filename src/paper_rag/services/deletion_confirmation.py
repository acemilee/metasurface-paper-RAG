from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID


@dataclass(frozen=True)
class DeletionConfirmation:
    document_id: UUID
    original_filename: str
    expires_at: datetime


class DeletionConfirmationStore:
    def __init__(self) -> None:
        self._tokens: dict[str, DeletionConfirmation] = {}
        self._lock = threading.Lock()

    def issue(self, document_id: UUID, original_filename: str, ttl_seconds: int) -> str:
        token = secrets.token_urlsafe(32)
        confirmation = DeletionConfirmation(
            document_id=document_id,
            original_filename=original_filename,
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
        )
        with self._lock:
            self._purge_expired()
            self._tokens[token] = confirmation
        return token

    def consume(self, token: str, document_id: UUID, confirm_filename: str) -> bool:
        with self._lock:
            self._purge_expired()
            confirmation = self._tokens.pop(token, None)
        return bool(
            confirmation
            and confirmation.document_id == document_id
            and secrets.compare_digest(
                confirmation.original_filename.encode("utf-8"),
                confirm_filename.encode("utf-8"),
            )
        )

    def _purge_expired(self) -> None:
        now = datetime.now(UTC)
        expired = [token for token, item in self._tokens.items() if item.expires_at <= now]
        for token in expired:
            self._tokens.pop(token, None)


deletion_confirmations = DeletionConfirmationStore()
