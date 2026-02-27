"""Create all tables if they don't already exist."""

from storage.db import engine
from storage.models import Base


def main() -> None:
    Base.metadata.create_all(engine)
    print(f"Tables created: {list(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    main()
