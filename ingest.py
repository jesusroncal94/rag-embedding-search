#!/usr/bin/env python3
"""Entry point for the data ingestion script, as named in the spec.

The implementation lives in `app/ingest.py` so that it shares the chunker, config and
provider adapters with the query path (ADR-001). This shim exists so the deliverable is
runnable exactly as the spec describes it:

    python ingest.py
"""

import sys

from app.ingest import main

if __name__ == "__main__":
    sys.exit(main())
