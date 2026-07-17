# config/env.py

"""Load environment variables from the project's .env file into os.environ.

Call before importing any harness module that reads settings. Kept as an
explicit, in-process step so that every entry point (the server and the
smoke tests) populates the environment the same way.

Real environment variables always win: load_dotenv() does not override a
variable that is already present in os.environ, so container/CI-provided
config takes precedence over the .env file.
"""

from pathlib import Path

from dotenv import load_dotenv

# The .env lives at the repo root, one level above this package.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_secrets() -> None:
    """Load the .env file into os.environ (no-op for vars already set)."""
    load_dotenv(_ENV_PATH)
