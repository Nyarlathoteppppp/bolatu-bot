import json
from types import SimpleNamespace

import pytest

from qq_social_agent.discourse_state import Binding
from qq_social_agent.ellipsis_resolver import EllipsisResolution
from qq_social_agent.interaction_state import format_interaction_state
from qq_social_agent.memory import MemoryStore
from qq_social_agent.resolver_result import RESOLVED, UNAVAILABLE


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / 'state.sqlite3')


def incoming(store, source, text, *, user=1, group=100, reply='', addressed=True, at=None, segments=True):
    store.add_message(group, user, str(user), text, source_message_id=source, created_at=at,
                      message_segments_json=json.dumps([{'type': 'text', 'data': {'text': text}}]) if segments else None)
    store.interactions.observe_inbound(group_id=group, source_message_id=source,
                                      reply_source_id=reply, addressed_bot=addressed)
    return store.conn.execute('select id from messages where group_id=? and source_message_id=?', (group, source)).fetchone()[0]


def sent(store, source, trigger, *, at=10, context_at=2):
    store.add_message(100, 9, '风雪', '回了', is_bot=True, source_message_id=source, created_at=at)
    store.interactions.observe_sent(group_id=100, source_message_id=source, trigger_source_id=trigger,
                                   action='reply', context_at=context_at)


def state(store, source, ids=None):
    if ids is None:
        ids = [row[0] for row in store.conn.execute('select id from messages where group_id=100')]
    return store.interactions.for_source(100, source, context_message_ids=ids)


def payload(state):
    return json.loads(format_interaction_state(state).split('\n')[1])


def discourse(*, target=9, status=RESOLVED, source='', kind='CONTINUATION'):
    return SimpleNamespace(addressee=Binding(status=status, target_id=target),
                           continuation=Binding(status=RESOLVED, target=kind, kind=kind),
                           ellipsis=EllipsisResolution(kind=kind, status=RESOLVED,
                                                       source_message_id=source, source_text='原话'))


def test_missing_source_and_draft_cannot_create_evidence(store):
    store.interactions.observe_inbound(group_id=100, source_message_id='missing')
    store.interactions.observe_sent(group_id=100, source_message_id='draft', trigger_source_id='missing',
                                   action='reply', context_at=1)
    assert store.conn.execute('select count(*) from interaction_events').fetchone()[0] == 0
    assert state(store, 'missing') is None
    assert format_interaction_state(None) == ''


def test_unquoted_new_topic_same_user_is_separate(store):
    incoming(store, 'a', '你怎么看？')
    incoming(store, 'b', '吃什么？')
    assert [e.source_message_id for e in state(store, 'b').events] == ['b']


def test_quote_edges_and_group_isolation(store):
    incoming(store, 'a', '怎么做？')
    incoming(store, 'b', '另外一个问题？')
    incoming(store, 'c', '这个呢？', reply='a')
    incoming(store, 'cross', '你太凶', group=200, reply='a')
    assert [e.source_message_id for e in state(store, 'c').events] == ['a', 'c']
    assert store.conn.execute("select parent_message_id from interaction_events where group_id=200").fetchone()[0] is None


def test_only_resolved_continuation_can_link_and_quote_has_priority(store):
    incoming(store, 'a', '怎么做？')
    incoming(store, 'b', '这段代码', addressed=False)
    incoming(store, 'c', '为什么？')
    snapshot = store.interactions.bind_discourse(group_id=100, source_message_id='c', self_id=9,
                                                discourse=discourse(source='a', kind='ITEM_DEIXIS'),
                                                context_message_ids=[1, 2, 3])
    assert [e.source_message_id for e in snapshot.events] == ['c']
    snapshot = store.interactions.bind_discourse(group_id=100, source_message_id='c', self_id=9,
                                                discourse=discourse(source='a'), context_message_ids=[1, 2, 3])
    assert [e.source_message_id for e in snapshot.events] == ['a', 'c']
    incoming(store, 'd', '然后呢？', reply='b')
    snapshot = store.interactions.bind_discourse(group_id=100, source_message_id='d', self_id=9,
                                                discourse=discourse(source='a'), context_message_ids=[1, 2, 3, 4])
    assert [e.source_message_id for e in snapshot.events] == ['b', 'd']


def test_unavailable_resolution_does_not_guess_thread(store):
    incoming(store, 'a', '怎么做？')
    incoming(store, 'b', '然后呢？')
    unresolved = discourse(source='a')
    unresolved.ellipsis = EllipsisResolution(kind='CONTINUATION', status=UNAVAILABLE)
    snapshot = store.interactions.bind_discourse(group_id=100, source_message_id='b', self_id=9,
                                                discourse=unresolved, context_message_ids=[1, 2])
    assert [e.source_message_id for e in snapshot.events] == ['b']


def test_quote_to_preledger_bot_is_kept_and_sent_links_later(store):
    incoming(store, 'q', '怎么做？', at=1)
    store.add_message(100, 9, '风雪', '旧回复', is_bot=True, source_message_id='s', created_at=2)
    incoming(store, 'r', '你说错了', reply='s', addressed=False, at=3)
    assert state(store, 'r').feedback[0].source_message_id == 'r'
    store.interactions.observe_sent(group_id=100, source_message_id='s', trigger_source_id='q', action='reply', context_at=1)
    assert [e.source_message_id for e in state(store, 'r').events] == ['q', 's', 'r']


def test_quote_envelope_and_bystanders_are_not_bot_feedback(store):
    incoming(store, 'a', '认真回答一下，怎么做？')
    assert state(store, 'a').pending_questions[0].kind == 'question'
    incoming(store, 'b', '你说错了？', addressed=False)
    assert state(store, 'b').pending_questions == ()
    assert state(store, 'b').feedback == ()
    incoming(store, 'c', '别怼我？ [回复某人：你说错了]', segments=False)
    assert state(store, 'c').events[0].text == ''
    assert state(store, 'c').events[0].kind == 'message'
    assert state(store, 'c').events[0].text_provenance == 'unavailable'
    incoming(store, 'd', '你太凶')
    snapshot = store.interactions.bind_discourse(group_id=100, source_message_id='d', self_id=9,
                                                discourse=discourse(target=8), context_message_ids=[4])
    assert snapshot.feedback == ()


def test_acknowledged_send_is_delivery_not_solution_and_new_feedback_survives(store):
    incoming(store, 'q', '怎么做？', at=1)
    incoming(store, 'f', '你说错了', reply='q', at=3)
    sent(store, 's', 'q', at=4, context_at=2)
    snapshot = state(store, 'q')
    assert snapshot.pending_questions == ()
    assert [e.source_message_id for e in snapshot.feedback] == ['f']
    data = payload(snapshot)
    assert data['last_bot_send']['context_predates_feedback_ids'] == [snapshot.feedback[0].message_id]
    assert 'solved' not in data and 'emotion' not in data


def test_closure_only_affects_its_speaker_ancestors_and_persists_after_send(store):
    a = incoming(store, 'a', '怎么做？', user=1, at=1)
    b = incoming(store, 'b', '那这个呢？', user=2, reply='a', at=2)
    incoming(store, 'close', '不问了', user=1, reply='b', at=3)
    assert [e.message_id for e in state(store, 'close').pending_questions] == [b]
    sent(store, 's', 'a', at=4)
    assert payload(state(store, 'close'))['last_explicit_closure']['qq_message_id'] == 'close'
    c = incoming(store, 'c', '另一个怎么做？', user=1, reply='close', at=5)
    assert [e.message_id for e in state(store, 'c').pending_questions] == [b, c]


def test_snapshot_reuses_current_context_without_old_sibling_tree(store):
    incoming(store, 'old', '怎么做？', at=1)
    incoming(store, 'old-sibling', '别怼我', reply='old', at=2)
    current = incoming(store, 'new', '这个呢？', reply='old', at=30)
    assert [e.source_message_id for e in state(store, 'new', [current]).events] == ['old', 'new']


def test_restart_dedup_and_reset(store):
    incoming(store, 'q', '怎么做？', at=1)
    incoming(store, 'q', '怎么做？', at=1)
    sent(store, 's', 'q', at=2)
    sent(store, 's', 'q', at=2)
    path = store.db_path
    store.conn.close()
    reopened = MemoryStore(path)
    assert len(state(reopened, 'q').events) == 2
    reopened.reset_group_messages(100)
    assert reopened.conn.execute('select count(*) from interaction_events').fetchone()[0] == 0


def test_sibling_reply_branches_are_separate_even_inside_context(store):
    incoming(store, 'root', '怎么做？')
    incoming(store, 'side', '你说错了', reply='root', user=2)
    incoming(store, 'side-child', '为什么呢？', reply='side', user=2)
    incoming(store, 'current', '这样可以吗？', reply='root', user=1)
    assert [e.source_message_id for e in state(store, 'current').events] == ['root', 'current']


def test_qq_parent_can_have_larger_db_id_after_backfill(store):
    store.add_message(100, 1, 'A', '这个呢？', source_message_id='child', created_at=2,
                      message_segments_json='[{"type":"text","data":{"text":"这个呢？"}}]')
    incoming(store, 'parent', '怎么做？', at=1)
    store.interactions.observe_inbound(group_id=100, source_message_id='child', reply_source_id='parent', addressed_bot=True)
    assert [e.source_message_id for e in state(store, 'child').events] == ['parent', 'child']
