"""Repositories share the store's connection and existing transaction boundary."""
import sqlite3
import subprocess
import sys

import pytest

from qq_social_agent.memory import MemoryStore


def test_repositories_import_without_store_or_plugin():
    script = '''import sys
from qq_social_agent.memory_metrics_repository import MetricsRepository
from qq_social_agent.memory_private_state_repository import PrivateStateRepository
from qq_social_agent.memory_meme_repository import MemeRepository
from qq_social_agent.memory_atom_repository import MemoryAtomRepository
from qq_social_agent.memory_style_repository import StyleRepository
assert "qq_social_agent.memory" not in sys.modules
assert "qq_social_agent.plugin" not in sys.modules
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


def test_repositories_share_connection_and_close_with_store(tmp_path):
    store = MemoryStore(tmp_path / 'bot.sqlite3')
    repositories = (
        store._metrics_repository, store._private_state_repository, store._meme_repository,
        store._atom_repository, store._style_repository,
    )
    assert all(repo.conn is store.conn for repo in repositories)
    assert store.interactions.conn is store.conn
    assert store.images.conn is store.conn
    store.conn.close()
    for read in (
        lambda: store.metric_event_count('test'),
        lambda: store.private_conversation_state(123),
        lambda: store.meme_asset(123),
        lambda: store.memory_atom(123),
        lambda: store.recent_style_rules(100, 2),
    ):
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            read()


@pytest.mark.parametrize('domain', ['metrics', 'private_state', 'meme', 'atom', 'style'])
def test_repository_write_keeps_existing_shared_commit_boundary(tmp_path, domain):
    path = tmp_path / 'bot.sqlite3'
    store = MemoryStore(path)
    with sqlite3.connect(path) as observer:
        store.conn.execute("insert into app_kv(key, value, updated_at) values ('pending', 'value', 1)")
        assert store.conn.in_transaction
        assert observer.execute("select count(*) from app_kv where key = 'pending'").fetchone()[0] == 0
        if domain == 'metrics':
            store.add_metric_event(event_type='test', group_id=100, created_at=1)
        elif domain == 'private_state':
            store.update_private_conversation_state(chat_id=123, user_id=456, display_name='成员')
        elif domain == 'atom':
            store.add_memory_atom(atom_type='fact', group_id=100, content='测试事实', source='manual')
        elif domain == 'style':
            store.add_style_rules(100, [('测试情境', '测试表达', '原话')])
        else:
            store.upsert_meme_asset(
                sha256='asset', source_group_id=100, source_user_id=456,
                source_message_id='1', file_path='asset.png', mime_type='image/png',
                byte_size=1, description='测试素材', enabled=True,
            )
        assert not store.conn.in_transaction
        assert observer.execute("select value from app_kv where key = 'pending'").fetchone()[0] == 'value'
    store.conn.close()
