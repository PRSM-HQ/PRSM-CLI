from prsm.shared.models.session import is_default_session_name


def test_default_session_name_matches_canonical_form() -> None:
    assert is_default_session_name("Session 1")
    assert is_default_session_name("(Forked) Session 2")


def test_default_session_name_is_case_insensitive() -> None:
    assert is_default_session_name("session 2")
    assert is_default_session_name("(Forked) session 3")


def test_default_session_name_rejects_non_default_patterns() -> None:
    assert not is_default_session_name("Session1")
    assert not is_default_session_name("Session abc")
    assert not is_default_session_name("My task")


def test_default_session_name_trims_outer_whitespace() -> None:
    assert is_default_session_name("  Session 4  ")
    assert is_default_session_name("  (Forked) session 5 ")
