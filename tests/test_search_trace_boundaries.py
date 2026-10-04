"""Query and trace consumers must not pull in their runtime orchestration."""
import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize('modules,forbidden', [
    (
        'qq_social_agent.tools.fresh_types, qq_social_agent.tools.fresh_intent, qq_social_agent.tool_router',
        ('qq_social_agent.tools.fresh_context', 'qq_social_agent.tools.fresh_providers'),
    ),
    (
        'qq_social_agent.trace_snapshot, qq_social_agent.trace_render',
        ('qq_social_agent.observability', 'qq_social_agent.plugin'),
    ),
])
def test_independent_consumers_do_not_load_orchestration(modules, forbidden):
    script = f'''import sys
import {modules}
for name in {forbidden!r}:
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


def test_legacy_exports_use_canonical_search_and_trace_implementations():
    for canonical, legacy in (
        ('qq_social_agent.tools.fresh_types', 'qq_social_agent.tools.fresh_context'),
        ('qq_social_agent.tools.fresh_intent', 'qq_social_agent.tools.fresh_context'),
        ('qq_social_agent.tools.fresh_providers', 'qq_social_agent.tools.fresh_context'),
        ('qq_social_agent.tools.fresh_text', 'qq_social_agent.tools.fresh_context'),
        ('qq_social_agent.trace_snapshot', 'qq_social_agent.observability'),
        ('qq_social_agent.trace_render', 'qq_social_agent.observability'),
    ):
        canonical_module = importlib.import_module(canonical)
        legacy_module = importlib.import_module(legacy)
        definitions = [
            (name, value) for name, value in vars(canonical_module).items()
            if getattr(value, '__module__', None) == canonical
        ]
        assert definitions
        for name, value in definitions:
            assert getattr(legacy_module, name) is value, name
