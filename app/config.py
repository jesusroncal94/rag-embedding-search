from __future__ import annotations

import os
from dataclasses import asdict, field
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass

REDACTED = "***"

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonEmpty = Annotated[str, Field(min_length=1)]

_frozen = ConfigDict(extra="forbid", frozen=True)


class ConfigError(RuntimeError):
    pass


@dataclass(config=_frozen)
class CorpusConfig:
    source_dir: str
    supported_extensions: list[str]


@dataclass(config=_frozen)
class ChunkingConfig:
    paragraph_delimiter: NonEmpty
    min_chunk_chars: PositiveInt
    max_chunk_chars: PositiveInt

    def __post_init__(self) -> None:
        if self.min_chunk_chars >= self.max_chunk_chars:
            raise ValueError(
                "'chunking.min_chunk_chars' must be less than 'chunking.max_chunk_chars' "
                f"({self.min_chunk_chars} >= {self.max_chunk_chars})"
            )


@dataclass(config=_frozen)
class EmbeddingConfig:
    model: NonEmpty
    dimensions: PositiveInt
    embed_batch_size: PositiveInt
    document_task_type: str
    query_task_type: str


@dataclass(config=_frozen)
class RetrievalConfig:
    top_k: PositiveInt
    max_top_k: PositiveInt
    min_similarity: Annotated[float, Field(ge=-1.0, le=1.0)]
    distance_metric: Literal["cosine"]

    def __post_init__(self) -> None:
        if self.top_k > self.max_top_k:
            raise ValueError(
                "'retrieval.top_k' must not exceed 'retrieval.max_top_k' "
                f"({self.top_k} > {self.max_top_k})"
            )


@dataclass(config=_frozen)
class GenerationConfig:
    model: NonEmpty
    temperature: Annotated[float, Field(ge=0.0, le=2.0)]
    max_output_tokens: PositiveInt
    prompt_template_path: str


@dataclass(config=_frozen)
class RuntimeConfig:
    max_retries: PositiveInt
    backoff_base_ms: PositiveInt
    request_timeout_ms: PositiveInt
    max_question_chars: PositiveInt


@dataclass(config=_frozen)
class LoggingConfig:
    level: str
    log_question_text: bool
    log_answer_text: bool


@dataclass(config=_frozen)
class DatabaseConfig:
    url: str
    collection_name: str


@dataclass(config=_frozen)
class Config:
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    database: DatabaseConfig
    gemini_api_key: str = field(repr=False, default="")

    def redacted(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("gemini_api_key"):
            data["gemini_api_key"] = REDACTED
        return data


_CONFIG_ADAPTER = TypeAdapter(Config)


def load_config(path: str | Path | None = None) -> Config:
    raw = _read_yaml(Path(path or os.environ.get("CONFIG_PATH", "config/config.yaml")))
    raw["gemini_api_key"] = _required_env("GEMINI_API_KEY")
    raw.setdefault("database", {})
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        raw["database"] = {**raw["database"], "url": database_url}
    try:
        return _CONFIG_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise ConfigError(_describe(exc)) from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            "Set it in .env (see .env.example); it must never be written to config.yaml."
        )
    return value


def _describe(exc: ValidationError) -> str:
    problems = []
    for error in exc.errors():
        key = ".".join(str(part) for part in error["loc"])
        problems.append(f"'{key}': {error['msg']}" if key else error["msg"])
    return "Invalid configuration: " + "; ".join(problems)
