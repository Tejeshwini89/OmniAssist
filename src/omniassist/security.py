from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class User:
    """Authenticated identity used for document-level authorization."""

    user_id: str
    roles: frozenset[str]
    groups: frozenset[str]


class AccessPolicy:
    """Explicit document authorization policy.

    Documents may declare ``allowed_roles`` and/or ``allowed_groups`` in
    metadata. A restricted document is accessible only when the user matches
    at least one declared role/group. Documents without restrictions are
    available to authenticated users.
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
