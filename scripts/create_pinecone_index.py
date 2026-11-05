"""Create the Pinecone index for chunk vectors. Safe to run repeatedly.

Uses PINECONE_* settings from the environment / .env and configs/pinecone.yaml:

    python scripts/create_pinecone_index.py                 # Pinecone cloud (PINECONE_API_KEY)
    PINECONE_HOST=http://localhost:5080 python scripts/create_pinecone_index.py   # Pinecone Local

Exit codes: 0 created or already present, 1 configuration problem or shape mismatch.
"""

from __future__ import annotations

import argparse
import sys
from typing import get_args

from ledgerlens.core.config import PineconeConfig, get_config, get_settings
from ledgerlens.storage.pinecone_store import (
    IndexMismatchError,
    PineconeNotConfiguredError,
    create_index_admin,
    ensure_index,
)


def main(argv: list[str] | None = None) -> int:
    settings, config = get_settings(), get_config().pinecone
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=settings.pinecone_index)
    parser.add_argument("--dimension", type=int, default=config.dimension)
    parser.add_argument(
        "--metric",
        default=config.metric,
        choices=get_args(PineconeConfig.model_fields["metric"].annotation),
    )
    parser.add_argument("--cloud", default=settings.pinecone_cloud)
    parser.add_argument("--region", default=settings.pinecone_region)
    args = parser.parse_args(argv)

    if config.spec != "serverless":
        print(f"error: only serverless indexes are supported (spec={config.spec!r})")
        return 1

    target = settings.pinecone_host or "Pinecone cloud"
    try:
        admin = create_index_admin(settings.pinecone_api_key, settings.pinecone_host)
        outcome, info = ensure_index(
            admin,
            args.name,
            dimension=args.dimension,
            metric=args.metric,
            cloud=args.cloud,
            region=args.region,
        )
    except (PineconeNotConfiguredError, IndexMismatchError) as exc:
        print(f"error: {exc}")
        return 1

    verb = "created" if outcome == "created" else "already exists"
    print(
        f"{info.name}: {verb} on {target} "
        f"(dimension={info.dimension}, metric={info.metric}, host={info.host})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
