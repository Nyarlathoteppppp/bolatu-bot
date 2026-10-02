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


def sent(store, source, trigger, *, at=10, context_at=2, text='回了', action='reply'):
    store.add_message(100, 9, '风雪', text, is_bot=True, source_message_id=source, created_at=at)
    store.interactions.observe_sent(group_id=100, source_message_id=source, trigger_source_id=trigger,
                                   action=action, context_at=context_at)


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


def own(store, source, ids=None):
    if ids is None:
        ids = [row[0] for row in store.conn.execute('select id from messages where group_id=100')]
    return store.interactions.own_contributions(state(store, source, ids),
        source_message_id=source, context_message_ids=ids)


def test_own_reply_to_ancestor_is_generation_evidence_without_changing_decision_state(store):
    incoming(store, 'q', '列宁给我托梦了', at=1)
    sent(store, 's', 'q', at=2, text='梦里他说啥了？', action='ask_back')
    incoming(store, 'next', '他说你们不行', reply='q', at=3)
    assert [e.source_message_id for e in state(store, 'next').events] == ['q', 'next']
    contributions = own(store, 'next')
    assert [item['qq_message_id'] for item in contributions] == ['s']
    assert contributions[0]['said'] == '梦里他说啥了？'
    assert contributions[0]['trigger_said'] == '列宁给我托梦了'
    assert contributions[0]['expression_action'] == 'ask_back'
    assert contributions[0]['trigger_user_id'] == 1
    assert contributions[0]['question_like']
    assert contributions[0]['direct_reply_message_ids'] == []


def test_own_questions_record_direct_reply_without_claiming_resolution(store):
    incoming(store, 'q', '我想换电脑', at=1)
    sent(store, 's', 'q', at=2, text='你预算多少？')
    current_id = incoming(store, 'next', '八千', reply='s', at=3)
    contribution = own(store, 'next')[0]
    assert contribution['direct_reply_message_ids'] == [current_id]
    assert 'solved' not in contribution and 'emotion' not in contribution


def test_own_context_excludes_other_threads_future_sends_drafts_and_memes(store):
    incoming(store, 'q', '玩这个梗', at=1)
    sent(store, 'valid', 'q', at=2, text='我可不答应', action='take_side')
    incoming(store, 'other', '你怎么看代码？', at=2)
    sent(store, 'unrelated', 'other', at=2.5, text='代码有错误')
    sent(store, 'meme', 'q', at=2.6, text='[图片]', action='meme')
    store.add_message(100, 9, '风雪', '未确认草稿', is_bot=True, source_message_id='draft', created_at=2.7)
    incoming(store, 'next', '那你怎么想', reply='q', at=3)
    sent(store, 'future', 'q', at=4, text='之后才发的')
    assert [item['qq_message_id'] for item in own(store, 'next')] == ['valid']
    assert own(store, 'other') == []


def test_own_context_obeys_snapshot_and_explicit_topic_closure(store):
    incoming(store, 'q', '我说这个', at=1)
    sent(store, 's', 'q', at=2, text='我不同意')
    next_id = incoming(store, 'next', '接着说', reply='q', at=3)
    assert own(store, 'next', [next_id]) == []
    incoming(store, 'close', '换个话题', reply='s', at=4)
    assert own(store, 'close') == []
    incoming(store, 'new', '聊聊代码', reply='close', at=5)
    assert own(store, 'new') == []


def test_own_context_carries_observed_feedback_and_closure_but_not_future_feedback(store):
    incoming(store, 'q', '你怎么看？', at=1)
    sent(store, 's', 'q', at=2, text='我的看法是这样')
    incoming(store, 'correction', '你说错了', reply='s', user=2, at=3)
    incoming(store, 'close', '换个话题', reply='s', at=4)
    incoming(store, 'next', '我有个新信息', reply='q', at=5)
    incoming(store, 'future', '你说错了', reply='s', at=6)
    contribution = own(store, 'next')[0]
    assert [(item['kind'], item['user_id'], item['said']) for item in contribution['subsequent_feedback']] == [
        ('correction', 2, '你说错了'), ('closed', 1, '换个话题')]


def test_exact_quote_to_own_sent_message_survives_snapshot_cutoff(store):
    incoming(store, 'q', '你更喜欢哪个？', at=1)
    sent(store, 's', 'q', at=2, text='我选这个')
    current = incoming(store, 'next', '你刚才为什么选这个', reply='s', at=300)
    assert [item['said'] for item in own(store, 'next', [current])] == ['我选这个']


def test_recent_unprompted_sends_counts_unaddressed_triggers_once(store):
    incoming(store, 'a', '今天编译器又炸了', addressed=False, at=100)
    sent(store, 'a1', 'a', at=101, text='第一段')
    sent(store, 'a2', 'a', at=102, text='第二段')  # one split reply
    incoming(store, 'b', '风雪你怎么看', addressed=True, at=103)
    sent(store, 'b1', 'b', at=104)
    incoming(store, 'c', '早上好', addressed=False, at=10)
    sent(store, 'c1', 'c', at=11)  # outside the window

    assert store.interactions.recent_unprompted_sends(100, since=50) == 1
    assert store.interactions.recent_unprompted_sends(100, since=0) == 2
