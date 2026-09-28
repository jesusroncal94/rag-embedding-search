"""Configuration loading and validation.

FR-CFG-01 hyperparameters live in YAML; FR-CFG-02 secrets come from the environment;
FR-CFG-03/03a we fail fast, loudly, and name the offending key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

REDACTED = "***"


class ConfigError(RuntimeError):
    """Raised when configuration or secrets are absent or invalid (FR-CFG-03/03a)."""


@dataclass(frozen=True)
class CorpusConfig:
    source_dir: str
    supported_extensions: list[str]


@dataclass(frozen=True)
class ChunkingConfig:
    paragraph_delimiter: str
    min_chunk_chars: int
    max_chunk_chars: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dimensions: int
    embed_batch_size: int
    document_task_type: str
    query_task_type: str


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    max_top_k: int
    min_similarity: float
    distance_metric: str


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    temperature: float
    max_output_tokens: int
    prompt_template_path: str


@dataclass(frozen=True)
class RuntimeConfig:
    max_retries: int
    backoff_base_ms: int
    request_timeout_ms: int
    max_question_chars: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_question_text: bool
    log_answer_text: bool


@dataclass(frozen=True)
class DatabaseConfig:
    url: str
    collection_name: str


@dataclass(frozen=True)
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
        """Config as a dict with secrets masked — for FR-LOG-01."""
        data = asdict(self)
        if data.get("gemini_api_key"):
            data["gemini_api_key"] = REDACTED
        return data


def _require(section: dict[str, Any], key: str, path: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required configuration key: {path}.{key}")
    return section[key]


def _positive(value: Any, key: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Invalid configuration: '{key}' must be a positive number, got {value!r}")
    return value


def _validate(cfg: Config) -> None:
    """FR-CFG-03a: range-validate every value, naming the offending key."""
    _positive(cfg.chunking.min_chunk_chars, "chunking.min_chunk_chars")
    _positive(cfg.chunking.max_chunk_chars, "chunking.max_chunk_chars")
    if cfg.chunking.min_chunk_chars >= cfg.chunking.max_chunk_chars:
        raise ConfigError(
            "Invalid configuration: 'chunking.min_chunk_chars' must be less than "
            f"'chunking.max_chunk_chars' ({cfg.chunking.min_chunk_chars} >= "
            f"{cfg.chunking.max_chunk_chars})"
        )
    if not cfg.chunking.paragraph_delimiter:
        raise ConfigError("Invalid configuration: 'chunking.paragraph_delimiter' must not be empty")

    _positive(cfg.embedding.dimensions, "embedding.dimensions")
    _positive(cfg.embedding.embed_batch_size, "embedding.embed_batch_size")
    if not cfg.embedding.model:
        raise ConfigError("Invalid configuration: 'embedding.model' must not be empty")

    _positive(cfg.retrieval.top_k, "retrieval.top_k")
    _positive(cfg.retrieval.max_top_k, "retrieval.max_top_k")
    if cfg.retrieval.top_k > cfg.retrieval.max_top_k:
        raise ConfigError(
            "Invalid configuration: 'retrieval.top_k' must not exceed 'retrieval.max_top_k' "
            f"({cfg.retrieval.top_k} > {cfg.retrieval.max_top_k})"
        )
    if not -1.0 <= cfg.retrieval.min_similarity <= 1.0:
        raise ConfigError(
            "Invalid configuration: 'retrieval.min_similarity' must be within [-1.0, 1.0], "
            f"got {cfg.retrieval.min_similarity}"
        )
    if cfg.retrieval.distance_metric != "cosine":
        raise ConfigError(
            "Invalid configuration: 'retrieval.distance_metric' only 'cosine' is supported, "
            f"got {cfg.retrieval.distance_metric!r}"
        )

    if not 0.0 <= cfg.generation.temperature <= 2.0:
        raise ConfigError(
            f"Invalid configuration: 'generation.temperature' must be within [0.0, 2.0], "
            f"got {cfg.generation.temperature}"
        )
    _positive(cfg.generation.max_output_tokens, "generation.max_output_tokens")
    if not cfg.generation.model:
        raise ConfigError("Invalid configuration: 'generation.model' must not be empty")

    _positive(cfg.runtime.max_retries, "runtime.max_retries")
    _positive(cfg.runtime.backoff_base_ms, "runtime.backoff_base_ms")
    _positive(cfg.runtime.request_timeout_ms, "runtime.request_timeout_ms")
    _positive(cfg.runtime.max_question_chars, "runtime.max_question_chars")


def load_config(path: str | Path | None = None, *, require_api_key: bool = True) -> Config:
    """Load, validate, and return configuration.

    Raises ConfigError on anything missing or out of range (FR-CFG-03/03a).
    """
    cfg_path = Path(path or os.environ.get("CONFIG_PATH", "config/config.yaml"))
    if not cfg_path.is_file():
        raise ConfigError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    try:
        corpus = CorpusConfig(**raw["corpus"])
        chunking = ChunkingConfig(**raw["chunking"])
        embedding = EmbeddingConfig(**raw["embedding"])
        retrieval = RetrievalConfig(**raw["retrieval"])
        generation = GenerationConfig(**raw["generation"])
        runtime = RuntimeConfig(**raw["runtime"])
        logging_cfg = LoggingConfig(**raw["logging"])
        database = DatabaseConfig(**raw["database"])
    except KeyError as exc:
        raise ConfigError(f"Missing required configuration section: {exc.args[0]}") from exc
    except TypeError as exc:
        raise ConfigError(f"Invalid configuration structure: {exc}") from exc

    # FR-CFG-02: secrets and connection strings come from the environment.
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise ConfigError(
            "Missing required environment variable: GEMINI_API_KEY. "
            "Set it in .env (see .env.example); it must never be written to config.yaml."
        )

    db_url = os.environ.get("DATABASE_URL", "").strip() or database.url
    database = DatabaseConfig(url=db_url, collection_name=database.collection_name)

    cfg = Config(
        corpus=corpus,
        chunking=chunking,
        embedding=embedding,
        retrieval=retrieval,
        generation=generation,
        runtime=runtime,
        logging=logging_cfg,
        database=database,
        gemini_api_key=api_key,
    )
    _validate(cfg)
    return cfg
