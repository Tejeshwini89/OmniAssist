from langchain_core.documents import Document

from src.omniassist.security import AccessPolicy, User, user_from_headers


def test_role_or_group_authorization():
    policy = AccessPolicy()
    manager = User("alice", frozenset({"manager"}), frozenset())
    finance = User("bob", frozenset({"employee"}), frozenset({"finance"}))
    employee = User("eve", frozenset({"employee"}), frozenset())

    assert policy.can_access(manager, {"allowed_roles": ["manager"]})
    assert policy.can_access(finance, {"allowed_groups": ["finance"]})
    assert not policy.can_access(employee, {"allowed_groups": ["finance"]})
    assert policy.can_access(employee, {})


def test_filter_documents_enforces_acl():
    user = User("alice", frozenset({"employee"}), frozenset({"finance"}))
    docs = [
        Document("public", metadata={}),
        Document("finance", metadata={"allowed_groups": ["finance"]}),
        Document("hr", metadata={"allowed_groups": ["hr"]}),
    ]

    result = AccessPolicy().filter_documents(user, docs)
    assert [doc.page_content for doc in result] == ["public", "finance"]


def test_user_from_headers_normalizes_identity():
    user = user_from_headers(" alice ", "employee, manager", " finance, analytics ")
    assert user.user_id == "alice"
    assert user.roles == frozenset({"employee", "manager"})
    assert user.groups == frozenset({"finance", "analytics"})
