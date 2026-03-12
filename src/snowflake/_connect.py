"""
Shared Snowflake connection helper for EY Water Quality scripts.

Optionally loads credentials from a .env file at the project root.
No external dependencies beyond snowflake.connector.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Parse a simple KEY=VALUE .env file and inject into os.environ."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:   # don't override real env vars
                os.environ[key] = value


def load_credentials(project_root: Path | None = None) -> None:
    """
    Attempt to load Snowflake credentials into os.environ.

    Search order (first found wins, real env vars always take priority):
      1. Environment variables already set (always respected)
      2. <project_root>/.snowflake.env
      3. <project_root>/.env
    """
    root = project_root or Path(__file__).parent.parent.parent
    for name in (".snowflake.env", ".env"):
        candidate = root / name
        if candidate.exists():
            _load_dotenv(candidate)
            print(f"[creds] Loaded credentials from {candidate}")
            return


def get_connection(warehouse: str = "COMPUTE_WH", database: str = "EY_WQ",
                   role: str | None = None):
    """Return an active snowflake.connector.SnowflakeConnection."""
    import snowflake.connector

    load_credentials()

    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    user = os.environ.get("SNOWFLAKE_USER")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    resolved_role = role or os.environ.get("SNOWFLAKE_ROLE")

    missing = [k for k, v in {
        "SNOWFLAKE_ACCOUNT": account,
        "SNOWFLAKE_USER": user,
        "SNOWFLAKE_PASSWORD": password,
    }.items() if not v]

    if missing:
        raise EnvironmentError(
            f"Missing Snowflake credentials: {missing}\n"
            "Set them as environment variables or create a .snowflake.env file "
            "at the project root. See src/snowflake/.snowflake.env.template for format."
        )

    return snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=warehouse,
        database=database,
        role=resolved_role,
    )
