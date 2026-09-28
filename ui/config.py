from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field


class UiSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_base_url: str = Field(min_length=1)
    request_timeout_s: float = Field(default=120.0, gt=0)
    connect_timeout_s: float = Field(default=3.0, gt=0)
    health_timeout_s: float = Field(default=3.0, gt=0)

    @classmethod
    def from_env(cls) -> UiSettings:
        return cls(
            api_base_url=os.environ.get("API_BASE_URL", "").strip(),
            request_timeout_s=os.environ.get("REQUEST_TIMEOUT_S", 120.0),
            connect_timeout_s=os.environ.get("CONNECT_TIMEOUT_S", 3.0),
            health_timeout_s=os.environ.get("HEALTH_TIMEOUT_S", 3.0),
        )
