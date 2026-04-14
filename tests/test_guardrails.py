import pytest

from backend.guardrails import GuardrailViolation, sanitize_user_input


def test_sanitize_trims_and_rejects_empty() -> None:
    assert sanitize_user_input("  hello  ") == "hello"
    with pytest.raises(GuardrailViolation):
        sanitize_user_input("")


def test_sanitize_rejects_too_long() -> None:
    with pytest.raises(GuardrailViolation):
        sanitize_user_input("x" * 300000)
