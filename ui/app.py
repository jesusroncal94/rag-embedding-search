from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from ui.client import Answer, ApiClient, Failure, create_client
from ui.demo import DEMO_QUESTIONS

INJECTION_TAG = "instruction-like text · logged"
FAILURE_HINT = (
    "Nothing was answered. Try again in a moment; if it keeps failing, "
    "search the api logs for the correlation id below."
)
UNREACHABLE_HINT = "Nothing was answered. Check that the api service is running."


@st.cache_resource
def _client() -> ApiClient:
    return create_client()


def _correlation_id(value: str) -> None:
    st.caption("correlation id")
    st.code(value, language=None)


def _sources(answer: Answer) -> None:
    with st.expander("Sources in context"):
        st.dataframe(
            [
                {
                    "source": citation.source_path,
                    "chunk": citation.chunk_ordinal,
                    "similarity": citation.similarity,
                    "text": citation.snippet,
                    "flag": INJECTION_TAG if citation.injection_suspected else "",
                }
                for citation in answer.citations
            ],
            hide_index=True,
            column_order=("source", "chunk", "similarity", "flag", "text"),
            column_config={
                "flag": st.column_config.TextColumn(width="medium"),
                "text": st.column_config.TextColumn(width="large"),
            },
        )


def _render_answer(answer: Answer) -> None:
    st.markdown(answer.text)
    if answer.grounded:
        st.badge(f"Answered · {len(answer.citations)} sources in context", color="green")
        _sources(answer)
    else:
        st.badge("Refused · no model call", color="gray")
        st.caption(
            "Nothing retrieved scored above the similarity floor, "
            "so the question was never sent to the model."
        )
    _correlation_id(answer.correlation_id)


def _render_failure(failure: Failure) -> None:
    if failure.correlation_id:
        st.error(FAILURE_HINT, title=failure.message)
        _correlation_id(failure.correlation_id)
    else:
        st.error(UNREACHABLE_HINT, title=failure.message)


def _render_turn(question: str, result: Answer | Failure) -> None:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if isinstance(result, Answer):
            _render_answer(result)
        else:
            _render_failure(result)


def _sidebar(client: ApiClient) -> None:
    with st.sidebar:
        st.subheader("Try these")
        st.caption("Each question is answered on its own; there is no follow-up memory.")
        for question in DEMO_QUESTIONS:
            if st.button(question, key=question, width="stretch"):
                st.session_state["pending"] = question
        health = client.health()
        state = "healthy" if health.reachable else "unreachable"
        st.caption(f"api · {state} · {health.chunk_count} chunks")


st.set_page_config(page_title="Documentation Q&A", page_icon="📄", layout="wide")

try:
    client = _client()
except ValidationError:
    st.error("Set API_BASE_URL to the address of the api service.", title="The UI is not configured.")
    st.stop()

_sidebar(client)
st.title("Documentation Q&A")
st.caption(
    "Answers come only from the policy corpus. "
    "Every answer shows the sources that were in its context."
)

history = st.session_state.setdefault("history", [])
for past_question, past_result in history:
    _render_turn(past_question, past_result)

question = st.chat_input("Ask about the policies…") or st.session_state.pop("pending", None)
if question:
    result = client.ask(question)
    history.append((question, result))
    _render_turn(question, result)
