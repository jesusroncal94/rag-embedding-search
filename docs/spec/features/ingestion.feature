# Traces: FR-ING-01..10, FR-CFG-04
@ingestion
Feature: Document ingestion into the vector store
  As the platform engineer
  I want the documentation corpus chunked, embedded and indexed reproducibly
  So that retrieval has a trustworthy substrate

  Background:
    Given a configuration file with:
      | key                   | value |
      | chunking.min_chunk_chars | 200   |
      | chunking.max_chunk_chars | 1500  |
      | embedding.embed_batch_size | 32  |
      | runtime.max_retries   | 3     |
    And a reachable vector database
    And a valid Gemini API key in the environment

  # FR-ING-01
  Scenario: All supported documents under the source directory are ingested
    Given a source directory containing 4 markdown documents and 1 ".png" file
    When I run the ingestion script
    Then 4 documents are reported as processed
    And the ".png" file is skipped without failing the run

  # FR-ING-02
  Scenario: Documents are split on paragraph boundaries
    Given a document with 3 paragraphs separated by blank lines, each 400 characters
    When I run the ingestion script
    Then 3 chunks are stored
    And no chunk contains text from more than one paragraph

  # FR-ING-03
  Scenario: Short paragraphs are merged forward
    Given a document whose paragraphs are 50, 60 and 900 characters
    When I run the ingestion script
    Then 1 chunk is stored
    And that chunk contains the text of all 3 paragraphs in original order

  # FR-ING-03 — boundary case exposed by this pass
  Scenario: A trailing short paragraph with nothing to merge into is still stored
    Given a document whose paragraphs are 900 and 50 characters
    When I run the ingestion script
    Then 2 chunks are stored
    And the 50-character chunk is stored rather than discarded

  # FR-ING-04
  Scenario: Oversized paragraphs are split at sentence boundaries
    Given a document with a single paragraph of 4000 characters containing sentence terminators
    When I run the ingestion script
    Then 3 chunks are stored
    And every chunk is at most 1500 characters
    And every chunk ends at a sentence terminator

  # FR-ING-04 — degenerate case exposed by this pass
  Scenario: An oversized paragraph with no sentence boundary is hard-split
    Given a document with a single 4000-character paragraph containing no sentence terminator
    When I run the ingestion script
    Then chunks are produced by hard character split at "max_chunk_chars"
    And a warning is logged naming the source document

  # FR-ING-05, FR-ING-06
  Scenario: Every chunk is embedded and carries provenance metadata
    Given a source directory containing 1 document of 5 paragraphs
    When I run the ingestion script
    Then every stored chunk has a non-empty embedding vector
    And every stored chunk carries "source_path", "chunk_ordinal", "start_offset" and "end_offset"
    And the chunk ordinals are contiguous starting at 0

  # FR-ING-07
  Scenario: Embedding requests are batched
    Given a corpus that produces 100 chunks
    And "embedding.embed_batch_size" is 32
    When I run the ingestion script
    Then exactly 4 embedding API calls are made

  # FR-ING-08
  Scenario: Transient embedding failures are retried
    Given the embedding API fails with a 503 on the first 2 attempts and succeeds on the third
    When I run the ingestion script
    Then ingestion completes successfully
    And 2 retry events are logged with increasing backoff

  # FR-ING-08a — the quota ran out twice because 429s were retried on the backoff schedule
  Scenario: A rate limit waits as long as the provider asks
    Given the embedding API answers 429 once, asking for a retry after "0.05s"
    When I run the ingestion script
    Then ingestion completes successfully
    And 1 retry event is logged, waiting 50 ms

  # FR-ING-08a
  Scenario: An exhausted daily quota is not retried
    Given the embedding API answers 429 because the daily quota is exhausted
    When I run the ingestion script
    Then the script exits with a non-zero status
    And no retry event is logged
    And exactly 1 embedding API call was attempted

  # FR-ING-08a
  Scenario: A rate limit longer than the allowed wait is not waited out
    Given the embedding API answers 429 once, asking for a retry after "3600s"
    When I run the ingestion script
    Then the script exits with a non-zero status
    And no retry event is logged

  # FR-ING-08
  Scenario: Permanent embedding failure aborts the run
    Given the embedding API fails with a 401 on every attempt
    When I run the ingestion script
    Then the script exits with a non-zero status
    And a structured error log names the failing stage as "embed"
    And no partially-embedded collection is left serving queries

  # FR-ING-09
  Scenario: Ingestion is idempotent
    Given the ingestion script has already been run once over the corpus
    When I run the ingestion script a second time over the same corpus
    Then the stored chunk count is unchanged
    And no chunk is duplicated

  # FR-ING-10
  Scenario: A machine-readable summary is emitted
    When I run the ingestion script
    Then a single JSON log line reports "documents_processed", "chunks_created", "embedding_calls" and "duration_ms"

  # FR-CFG-04 / R-02 — exposed by this pass
  Scenario: The embedding model used is recorded with the collection
    When I run the ingestion script
    Then the collection metadata records the embedding model name and dimensionality
