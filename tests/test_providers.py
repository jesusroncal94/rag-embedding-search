from __future__ import annotations

import logging

import pytest
from google.genai import _extra_utils

from app.providers import GeminiGenerationProvider
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
