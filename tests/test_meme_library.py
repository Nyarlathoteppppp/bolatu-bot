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


def test_group_meme_gate_respects_group_cooldown(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    library = PrivateMemeLibrary(
        memory,
        {
            "allowed_group_ids": [1026813421],
            "group_cooldown_seconds": 60,
        },
        data_dir=tmp_path,
    )
    image_path = tmp_path / "meme_library" / "a.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png-data")
    asset = memory.upsert_meme_asset(
        sha256="b" * 64,
        source_group_id=1026813421,
        source_user_id=1535071184,
        source_message_id="manual:group",
        file_path=str(image_path),
        mime_type="image/png",
        byte_size=image_path.stat().st_size,
        description="开心",
        tags=("开心",),
        enabled=True,
    )

    assert library.group_gate(1026813421).allowed
    assert library.mark_group_sent(1026813421, asset.id)
    assert library.group_gate(1026813421).reason == "group_cooldown"


def test_private_meme_turn_gate_can_run_for_an_allowed_user(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    library = PrivateMemeLibrary(
        memory,
        {"allowed_private_users": [1903297906]},
        data_dir=tmp_path,
    )

    result = library.turn_gate(1903297906, received_messages=1)

    assert result.messages_since_last_meme == 100
    assert result.reason in {"eligible", "early_turn_70_percent_skip", "fourth_turn_20_percent_skip"}


def test_meme_cooldown_is_scoped_to_each_conversation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_path = data_dir / "meme_library" / "a.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png-data")
    memory = MemoryStore(data_dir / "bot.sqlite3")
    asset = memory.upsert_meme_asset(
        sha256="c" * 64,
        source_group_id=1026813421,
        source_user_id=1535071184,
        source_message_id="manual:scoped",
        file_path=str(image_path),
        mime_type="image/png",
        byte_size=image_path.stat().st_size,
        description="害羞",
        tags=("害羞",),
        enabled=True,
    )
    library = PrivateMemeLibrary(
        memory,
        {
            "allowed_private_users": [1903297906],
            "allowed_group_ids": [1026813421],
            "same_meme_cooldown_seconds": 3600,
        },
        data_dir=data_dir,
    )

    assert library.mark_sent(1903297906, asset.id)
    assert library.candidates(1903297906, query="害羞") == []
    assert [item.id for item in library.group_candidates(1026813421, query="害羞")] == [asset.id]
