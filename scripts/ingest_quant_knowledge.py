#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qq_social_agent.rag_retriever import RAGService

DEFAULT_GROUP_ID = 1026813421
DEFAULT_CREATED_BY = 1535071184
DOC_RELATIVE_PATH = Path("docs/career/quant_industry_guide_2026.md")
TITLE = "量化行业求职地图 2026-08"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the quant career guide into the group RAG knowledge base.")
    parser.add_argument("--group", type=int, default=DEFAULT_GROUP_ID, help="QQ group id for the knowledge source")
    parser.add_argument("--created-by", type=int, default=DEFAULT_CREATED_BY, help="QQ user id recorded as creator")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / "config.yaml"
    doc_path = root / DOC_RELATIVE_PATH
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    content = doc_path.read_text(encoding="utf-8")

    service = RAGService(root / "data" / "bot.sqlite3", raw_config.get("rag", {}))
    try:
        chunks = service.ingest_knowledge(
            group_id=args.group,
            kind="file",
            source_identity=str(DOC_RELATIVE_PATH),
            title=TITLE,
            content=content,
            created_by=args.created_by,
        )
        print(f"ingested group={args.group} title={TITLE!r} chunks={chunks}")
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
