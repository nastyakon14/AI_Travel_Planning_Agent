"""Простая авторизация для Streamlit: пароль из env или secrets."""

from __future__ import annotations

import hashlib
import os

import streamlit as st


def _expected_password() -> str | None:
    try:
        sec = st.secrets
        if sec is not None and "STREAMLIT_APP_PASSWORD" in sec:
            return str(sec["STREAMLIT_APP_PASSWORD"])
    except Exception:
        pass
    v = os.getenv("STREAMLIT_APP_PASSWORD") or os.getenv("APP_PASSWORD")
    return v.strip() if v else None


def _hash_user_id(token: str) -> str:
    return "u-" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def ensure_session_user() -> str:
    """
    Если задан пароль приложения — показывает поле ввода и возвращает стабильный user_id после входа.
    Иначе — анонимный идентификатор сессии.
    """
    if "session_uid" not in st.session_state:
        st.session_state.session_uid = os.urandom(8).hex()

    pwd = _expected_password()
    if not pwd:
        return f"anon-{st.session_state.session_uid}"

    if st.session_state.get("auth_ok"):
        return str(st.session_state.get("user_id") or f"anon-{st.session_state.session_uid}")

    with st.sidebar:
        st.subheader("Вход")
        entered = st.text_input("Пароль приложения", type="password", key="app_password_field")
        if st.button("Войти"):
            if entered == pwd:
                st.session_state.auth_ok = True
                st.session_state.user_id = _hash_user_id(entered + st.session_state.session_uid)
                st.rerun()
            else:
                st.error("Неверный пароль.")
    st.warning("Введите пароль в боковой панели, чтобы продолжить.")
    st.stop()
    return "pending"


def get_optional_user_id() -> str | None:
    """Для вызовов вне Streamlit или без пароля."""
    if st.session_state.get("auth_ok"):
        return str(st.session_state.get("user_id"))
    return None
