"""Secrets for entry points, from a `.env` file at the repository root (`HF_TOKEN`)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ".env"


def load_dotenv_from_repo_root(root: Path | None = None) -> bool:
    """Load `<root>/.env` into the environment if it exists; a variable already set wins."""
    import dotenv

    path = (REPO_ROOT if root is None else root) / ENV_FILE
    if not path.is_file():
        return False
    dotenv.load_dotenv(path, override=False)
    return True
