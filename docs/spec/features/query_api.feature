# Traces: FR-API-01..10
@api
Feature: Retrieval-augmented question answering over the API
  As an internal engineer
  I want to ask a natural-language question about our documentation
  So that I get a grounded answer with citations instead of hunting through pages

  Background:
    Given the corpus has been ingested
    And the API service is healthy
    And "retrieval.top_k" is 5
    And "retrieval.min_similarity" is 0.35

  # FR-API-01, FR-API-02, FR-API-03, FR-API-04, FR-API-05
  Scenario: A question answerable from the corpus returns a cited answer
    When I POST to "/api/v1/query" with question "How do I rotate the service credentials?"
    Then the response status is 200
    And the response body contains a non-empty "answer"
    And the response body contains at least 1 citation
    And every citation carries "source_path" and "chunk_ordinal"
    And every citation corresponds to a chunk that was placed in the LLM context

  # FR-API-03 — request-level override
  Scenario: The caller may override top_k within configured bounds
    When I POST to "/api/v1/query" with question "How do I deploy?" and top_k 3
    Then exactly 3 chunks are retrieved

  # FR-API-03 — bound exposed by this pass
  Scenario: An out-of-range top_k is rejected
    When I POST to "/api/v1/query" with question "How do I deploy?" and top_k 500
    Then the response status is 400
    And the error message names the permitted range

  # FR-API-06 — the anti-hallucination guarantee
  Scenario: A question with no supporting context is refused, not fabricated
    When I POST to "/api/v1/query" with question "What is the CEO's home address?"
    Then the response status is 200
    And the "answer" states that the documentation does not contain the information
    And the citation list is empty
    And no generation call is made to the LLM

  # FR-API-07
  Scenario Outline: Malformed requests are rejected before any provider call
    When I POST to "/api/v1/query" with question <question>
    Then the response status is 400
    And no embedding call is made
    And no generation call is made

    Examples:
      | question   |
      | ""         |
      | "   "      |
      | omitted    |

  # FR-API-07 — bound exposed by this pass
  Scenario: An excessively long question is rejected
    When I POST to "/api/v1/query" with a question of 50000 characters
    Then the response status is 400
    And the error message names the maximum question length

  # FR-API-08
  Scenario: Upstream provider failure is surfaced as 502 without leaking internals
    Given the Gemini generation endpoint returns a 500
    When I POST to "/api/v1/query" with question "How do I deploy?"
    Then the response status is 502
    And the response body contains a "correlation_id"
    And the response body contains no stack trace and no API key

  # FR-API-08 — exposed by this pass
  Scenario: A provider call exceeding the configured timeout is aborted
    Given the Gemini generation endpoint does not respond within "runtime.request_timeout_ms"
    When I POST to "/api/v1/query" with question "How do I deploy?"
    Then the response status is 504
    And the failure is logged with stage "generate"

  # FR-API-09
  Scenario: Health reflects dependency reachability
    When I GET "/health"
    Then the response status is 200
    And the body reports the vector database as reachable

  # FR-API-09
  Scenario: Health degrades when the vector database is down
    Given the vector database is unreachable
    When I GET "/health"
    Then the response status is 503

  # FR-API-10
  Scenario: Correlation ids tie response to logs
    When I POST to "/api/v1/query" with question "How do I deploy?"
    Then the response carries a "correlation_id"
    And every log line emitted for that request carries the same "correlation_id"

  # FR-API-10 — exposed by this pass
  Scenario: A caller-supplied correlation id is honoured
    When I POST to "/api/v1/query" with header "X-Correlation-Id" set to "abc-123"
    Then the response "correlation_id" is "abc-123"
