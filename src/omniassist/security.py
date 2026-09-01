from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class User:
    """Authenticated user identity and document-access groups."""

    user_id: str
    roles: frozenset[str]
    groups: frozenset[str]


class AccessPolicy:
    """Small, explicit authorization layer for retrieved enterprise documents.

    Document metadata may contain ``allowed_roles`` and ``allowed_groups``.
    A document with neither field is public to authenticated OmniAssist users.
    """

    def can_access(self, user: User, metadata: dict) -> bool:
        allowed_roles = set(metadata.get("allowed_roles", []))
        allowed_groups = set(metadata.get("allowed_groups", []))

        if allowed_roles and user.roles.intersection(allowed_roles):
            return True
        if allowed_groups and user.groups.intersection(allowed_groups):
            return True
        return not allowed_roles and not allowed_groups

    def filter_documents(self, user: User, documents: Iterable) -> list:
        return [
            document
            for document in documents
            if self.can_access(user, getattr(document, "metadata", {}) or {})
        ]


DEFAULT_USER = User(
    user_id="demo-user",
    roles=frozenset({"employee"}),
    groups=frozenset(),
)

policy = AccessPolicy()
