from pathlib import Path

from qq_social_agent.meme_library import PrivateMemeLibrary
from qq_social_agent.memory import MemoryStore


def test_private_meme_library_reads_curated_asset(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    asset_dir = data_dir / "meme_library"
    asset_dir.mkdir(parents=True)
    image_path = asset_dir / "a.png"
    image_path.write_bytes(b"png-data")
    memory = MemoryStore(data_dir / "bot.sqlite3")
    asset = memory.upsert_meme_asset(
        sha256="a" * 64,
        source_group_id=1026813421,
        source_user_id=1535071184,
        source_message_id="manual:test",
        file_path=str(image_path),
        mime_type="image/png",
        byte_size=image_path.stat().st_size,
        description="害羞探头",
        tags=("害羞", "可爱"),
        enabled=True,
    )
    library = PrivateMemeLibrary(
        memory,
        {"allowed_private_users": [1903297906], "same_meme_cooldown_seconds": 60},
        data_dir=data_dir,
    )

    candidates = library.candidates(1903297906, query="有点害羞")

    assert [item.id for item in candidates] == [asset.id]
    assert library.image_base64_ref(asset.id).startswith("base64://")
    assert library.mark_sent(1903297906, asset.id)
    assert library.candidates(1903297906, query="害羞") == []
