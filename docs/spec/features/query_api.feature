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

  # FR-API-05a
  Scenario: Every citation carries its chunk text and an injection flag
    When I POST to "/api/v1/query" with question "How long do I have to dispute a card transaction?"
    Then the response status is 200
    And every citation carries the text of the chunk it cites
    And every citation says whether its chunk looked like an injection

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

  # FR-API-14 — fail fast and degrade, do not retry harder
  Scenario: Repeated generation failures stop the model calls and return the sources only
    Given "runtime.breaker_failure_threshold" is 3
    And the Gemini generation endpoint returns a 500
    When I POST the question "How do I dispute a transaction?" 4 times
    Then the first 3 responses are 502
    And the last response is 200 with "generated" false and the retrieved sections as citations
    And the model was called 3 times
    And a log line reports that the "generate" circuit opened

  # FR-API-14
  Scenario: After the open period one trial call decides whether the model is back
    Given "runtime.breaker_failure_threshold" is 1
    And "runtime.breaker_open_ms" is 50
    And the Gemini generation endpoint fails once and then recovers
    When I POST the question "How do I dispute a transaction?", wait 60 ms, and POST it again
    Then the last response is 200 with "generated" true
    And a log line reports that the "generate" circuit closed

  # FR-API-14 — with the embedding circuit open nothing can be retrieved, so there is no degraded answer
  Scenario: Repeated embedding failures fail fast without calling the provider
    Given "runtime.breaker_failure_threshold" is 2
    And the Gemini embedding endpoint keeps failing
    When I POST the question "How do I dispute a transaction?" 3 times
    Then every response is 502
    And the embedding provider was called 2 times

  # FR-API-14a — the retry budget is shared by every request to one provider
  Scenario: Retries stop once the shared retry budget is spent
    Given the embedding retry budget holds 4 tokens
    And the Gemini embedding endpoint keeps failing with a 503
    When I POST the question "How do I dispute a transaction?" 3 times
    Then every response is 502
    And the embedding provider was called 4 times
    And a log line reports that the "embed" retry budget is spent

  # FR-API-08b — found by the @live run on 2026-09-29
  Scenario: An answer cut off at the output token limit is not served
    Given the model stops at the output token limit mid-answer
    When I POST to "/api/v1/query" with question "How do I deploy?"
    Then the response status is 502
    And the response body does not contain the truncated text
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
