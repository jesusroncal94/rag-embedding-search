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
WAITING = "Searching the policy documents and drafting an answer…"


@st.cache_resource
def _client() -> ApiClient:
    return create_client()


def _correlation_id(value: str) -> None:
    st.caption("correlation id")
    st.code(value, language=None)


def _sources(answer: Answer, *, expanded: bool = False) -> None:
    with st.expander("Sources in context", expanded=expanded):
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
    if answer.grounded and answer.generated:
        st.badge(f"Answered · {len(answer.citations)} sources in context", color="green")
        _sources(answer)
    elif answer.grounded:
        st.badge("Sources only · model unavailable", color="orange")
        st.caption(
            "The model has failed repeatedly, so it is not being called for now. "
            "These are the matching sections, unedited."
        )
        _sources(answer, expanded=True)
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


def _render_question(question: str) -> None:
    with st.chat_message("user"):
        st.markdown(question)


def _render_result(result: Answer | Failure) -> None:
    if isinstance(result, Answer):
        _render_answer(result)
    else:
        _render_failure(result)


def _render_turn(question: str, result: Answer | Failure) -> None:
    _render_question(question)
    with st.chat_message("assistant"):
        _render_result(result)


def _ask_and_render(client: ApiClient, question: str) -> Answer | Failure:
    _render_question(question)
    with st.chat_message("assistant"):
        with st.spinner(WAITING, show_time=True):
            result = client.ask(question)
        _render_result(result)
    return result


def _take_suggestion() -> None:
    st.session_state["pending"] = st.session_state["suggestion"]
    st.session_state["suggestion"] = None


def _suggestions(*, busy: bool, first: bool) -> None:
    st.pills(
        "Try one of these" if first else "Or try another",
        DEMO_QUESTIONS,
        key="suggestion",
        on_change=_take_suggestion,
        disabled=busy,
    )


def _sidebar(client: ApiClient) -> None:
    with st.sidebar:
        st.subheader("About")
        st.caption("Each question is answered on its own; there is no follow-up memory.")
        health = client.health()
        state = "healthy" if health.reachable else "unreachable"
        st.caption(f"api · {state} · {health.chunk_count} chunks")


st.set_page_config(page_title="Documentation Q&A", page_icon="📄", layout="wide")

try:
    client = _client()
except ValidationError:
    st.error(
        "Set API_BASE_URL to the address of the api service.", title="The UI is not configured."
    )
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

suggested = st.session_state.pop("pending", None)
typed = st.chat_input(
    "Ask about the policies…", submit_mode="disable", disabled=suggested is not None
)
question = typed or suggested
current_turn = st.container()
_suggestions(busy=bool(question), first=not history)
if question:
    with current_turn:
        history.append((question, _ask_and_render(client, question)))
    st.rerun()
