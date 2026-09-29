from __future__ import annotations

import logging

import pytest
from google.genai import _extra_utils

from app.providers import GeminiEmbeddingProvider, GeminiGenerationProvider, RetryBudget
from tests.conftest import FakeGenaiClient

pytestmark = pytest.mark.unit


def test_generation_turns_off_function_calling_without_logging(base_config, caplog):
    client = FakeGenaiClient(answer="An answer.")
    GeminiGenerationProvider(base_config, client=client).generate("A prompt.")

    with caplog.at_level(logging.INFO, logger=""):
        caplog.clear()
        disabled = _extra_utils.should_disable_afc(client.models.last_config)

    assert disabled
    assert caplog.records == []


def test_both_provider_calls_carry_the_configured_timeout(base_config):
    client = FakeGenaiClient(answer="An answer.")
    GeminiEmbeddingProvider(base_config, client=client).embed_query("A question.")
    GeminiGenerationProvider(base_config, client=client).generate("A prompt.")

    timeout = base_config.runtime.request_timeout_ms
    assert client.models.last_embed_config.http_options.timeout == timeout
    assert client.models.last_config.http_options.timeout == timeout


def test_the_retry_budget_refills_on_success_up_to_its_size():
    budget = RetryBudget(max_tokens=4, token_ratio=0.5)
    assert budget.spend()
    assert not budget.spend()

    for _ in range(100):
        budget.earn()

    assert budget.spend()
    assert not budget.spend()
