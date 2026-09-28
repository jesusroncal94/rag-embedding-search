# Traces: FR-CFG-01..04, FR-DOC-01..05, FR-LOG-01..06, NFR-02, NFR-04, NFR-05
@runtime
Feature: Configuration, containerization and operational logging
  As the engineer who has to run this
  I want one command to bring the stack up and one file to tune it
  So that the PoC is reproducible by someone who did not write it

  # FR-CFG-01
  @unit
  Scenario: Hyperparameters are changed without touching code
    Given "retrieval.top_k" is 5 in the configuration file
    When I change it to 8 and restart the service
    Then queries retrieve 8 chunks
    And no source file was modified

  # FR-CFG-02, NFR-04
  @unit
  Scenario: Secrets never enter the config file or the image
    When I inspect the configuration file and the built image layers
    Then no Gemini API key is present
    And the service reads the key from the environment at runtime

  # FR-CFG-03
  @unit
  Scenario: Missing secrets fail fast at startup
    Given the Gemini API key environment variable is unset
    When the service starts
    Then it exits within 10 seconds
    And the log names the missing variable
    And it does not bind the HTTP port

  # FR-CFG-03 — exposed by this pass
  @unit
  Scenario: Invalid configuration values fail fast
    Given "retrieval.top_k" is set to -1
    When the service starts
    Then it exits with a validation error naming "retrieval.top_k"

  # FR-CFG-04 / R-02
  @unit
  Scenario: A query against an index built with a different embedding model is refused
    Given the collection metadata records embedding model "model-A"
    And the service is configured with embedding model "model-B"
    When the service starts
    Then it fails fast with a message naming both models

  # FR-DOC-01, NFR-02
  @docker
  Scenario: Single-command startup
    Given a clean machine with Docker and a Gemini API key in ".env"
    When I run "docker-compose up"
    Then the vector database and the API service both start
    And "/health" returns 200 within 90 seconds
    And no additional manual step was required

  # FR-DOC-02
  @docker
  Scenario: The API waits for its dependency
    Given the vector database takes 20 seconds to become ready
    When I run "docker-compose up"
    Then the API service does not report healthy before the database is reachable
    And the API service does not crash-loop

  # FR-DOC-03
  @docker
  Scenario: Ingestion runs as a one-shot container on the same network
    Given the stack is running
    When I run the ingestion container against the compose network
    Then it connects to the vector database by service name
    And it exits 0 on success

  # FR-DOC-04
  @docker
  Scenario: Vector data survives a restart
    Given the corpus has been ingested
    When I run "docker-compose down" and then "docker-compose up"
    Then queries still return answers without re-ingesting

  # FR-DOC-04 — the negative case
  @docker
  Scenario: Volume removal clears the corpus
    When I run "docker-compose down -v" and then "docker-compose up"
    Then the collection is empty
    And queries return "insufficient context" rather than an error

  # FR-DOC-05
  @docker
  Scenario: The runtime image is lean and secretless
    When I inspect the final image
    Then it contains no build toolchain
    And it contains no API key in any layer
    And the service runs as a non-root user

  # FR-LOG-01
  @unit
  Scenario: Service start logs resolved configuration with secrets redacted
    When the service starts
    Then one JSON log line reports the resolved configuration
    And the Gemini API key appears as "***"

  # FR-LOG-03, FR-LOG-04, FR-LOG-05
  @unit
  Scenario: A successful query emits exactly the critical events
    When I POST a valid question
    Then a log line reports embedding latency
    And a log line reports retrieval outcome with top score
    And a log line reports LLM latency and model name
    And no log line is emitted for non-critical internal steps

  # FR-LOG-02
  @unit
  Scenario: A failed query is logged once with its stage
    Given the LLM provider is failing
    When I POST a valid question
    Then exactly one error log line is emitted
    And it names stage "generate" and the correlation id

  # Privacy posture — A-07: questions ARE logged (they are the documentation-gap
  # signal); answer text never is (it is the larger leakage surface).
  @unit
  Scenario: Answer text is never logged, questions are
    When I POST a question containing the string "SECRET-CANARY"
    Then a log line with event "query.received" contains "SECRET-CANARY"
    And no log line contains the answer text
    And no log line contains the Gemini API key

  # All logs
  @unit
  Scenario: Every log record is machine-parseable
    When I run the full stack through one ingestion and one query
    Then every emitted log line parses as a single JSON object
    And every line carries "timestamp", "level" and "event"

  # NFR-05
  @manual
  Scenario: A newcomer can run the PoC from the README
    Given a reader with Docker and an API key who has not seen the repository
    When they follow the README top to bottom
    Then they reach a successful cited answer in under 10 minutes
