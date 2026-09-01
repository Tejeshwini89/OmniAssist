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


def user_from_headers(
    user_id: str | None,
    roles: str | None,
    groups: str | None,
) -> User:
    """Build a user from an identity gateway's normalized headers.

    These headers are an integration seam for a trusted reverse proxy/IdP.
    The application must not be exposed directly with client-controlled
    identity headers in production.
    """
    resolved_user_id = (user_id or "demo-user").strip()
    resolved_roles = frozenset(
        value.strip() for value in (roles or "employee").split(",") if value.strip()
    )
    resolved_groups = frozenset(
        value.strip() for value in (groups or "").split(",") if value.strip()
    )
    return User(
        user_id=resolved_user_id,
        roles=resolved_roles,
        groups=resolved_groups,
    )


DEFAULT_USER = User(
    user_id="demo-user",
    roles=frozenset({"employee"}),
    groups=frozenset(),
)

policy = AccessPolicy()
