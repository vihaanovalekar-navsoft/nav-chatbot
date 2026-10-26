"""Streamlit app for the NavAI assistant."""

import os
import sys

import streamlit as st

from navai import pipeline
from navai.answers import format_source
from navai.settings import ROOT

# Allow Streamlit Cloud secrets to satisfy the pipeline's environment lookups.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(str(_key), str(_value))
except Exception:
    pass


MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 12000


# Keep a module-level alias so tests can patch ``app.NavAIChat`` and so the
# class is resolvable when Streamlit executes this file as a script.
_ORIGINAL_NAVAI_CHAT = pipeline.NavAIChat
NavAIChat = pipeline.NavAIChat


def _resolve_navai_chat():
    """Return the chat class, honoring patches on app or pipeline."""
    app_candidate = globals().get("NavAIChat")
    if app_candidate is not None and app_candidate is not _ORIGINAL_NAVAI_CHAT:
        return app_candidate
    pipeline_candidate = getattr(pipeline, "NavAIChat", None)
    if (
        pipeline_candidate is not None
        and pipeline_candidate is not _ORIGINAL_NAVAI_CHAT
    ):
        return pipeline_candidate
    return _ORIGINAL_NAVAI_CHAT


def trim_history(messages):
    """Return recent messages within the configured size limits."""
    history = []
    total = 0
    for message in reversed(messages):
        content = message["content"]
        if len(history) >= MAX_HISTORY_MESSAGES:
            break
        if total + len(content) > MAX_HISTORY_CHARS:
            break
        history.insert(0, {"role": message["role"], "content": content})
        total += len(content)
    return history


def valid_message(message):
    """Return an error string when `message` is invalid, otherwise None."""
    if not message or not message.strip():
        return "Message cannot be empty."
    if len(message) > MAX_MESSAGE_CHARS:
        return f"Message must be at most {MAX_MESSAGE_CHARS} characters."
    return None


def sources_markdown(sources):
    """Return a compact text block describing citation sources."""
    if not sources:
        return ""
    return "Sources:\n" + "\n".join(
        format_source(source) for source in sources
    )


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat" not in st.session_state:
        st.session_state.chat = None


def get_chat():
    if st.session_state.chat is None:
        st.session_state.chat = _resolve_navai_chat()()
    return st.session_state.chat


def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for source in sources:
            st.caption(format_source(source))


def main():
    st.set_page_config(page_title="NavAI Assistant", page_icon="N", layout="centered")
    init_state()

    st.title("NavAI Assistant")
    st.caption("Retrieval-grounded answers from the NavAI documentation.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    if st.sidebar.button("New chat"):
        st.session_state.messages = []
        st.session_state.chat = None
        st.rerun()

    prompt = st.chat_input("Ask about NavAI")
    if not prompt:
        return

    error = valid_message(prompt)
    if error:
        st.error(error)
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = trim_history(st.session_state.messages[:-1])
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = get_chat().answer(prompt.strip(), history)
            except Exception as exc:  # noqa: BLE001 - surface runtime errors in UI
                st.error(str(exc))
                return
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
