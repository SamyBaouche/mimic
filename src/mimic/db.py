"""Initialise the storage backend (kept for `python -m mimic.db`; `mimic doctor` does it too)."""

from mimic.storage import get_storage


def init_db() -> None:
    storage = get_storage()
    storage.init()
    print(f"Storage ready: {storage.kind} ({storage.location})")


if __name__ == "__main__":
    init_db()
