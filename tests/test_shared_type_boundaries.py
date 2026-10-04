"""Shared records must stay usable without provider or database implementations."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize('modules', [
    'qq_social_agent.llm_task_types, qq_social_agent.memory_models',
    'qq_social_agent.decision_gate, qq_social_agent.tool_router, qq_social_agent.private_message_types',
])
def test_shared_types_and_pure_consumers_do_not_load_runtime_implementations(modules):
    script = f'''import sys
import {modules}
for name in (
    "qq_social_agent.deepseek_client", "qq_social_agent.llm_gateway",
    "qq_social_agent.jev_client", "qq_social_agent.memory", "qq_social_agent.plugin",
):
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('canonical,legacy', [
    ('qq_social_agent.llm_task_types', 'qq_social_agent.deepseek_client'),
    ('qq_social_agent.memory_models', 'qq_social_agent.memory'),
])
def test_existing_public_imports_are_the_same_shared_classes(canonical, legacy):
    types_module = importlib.import_module(canonical)
    old_module = importlib.import_module(legacy)
    classes = [
        (name, value) for name, value in vars(types_module).items()
        if isinstance(value, type) and value.__module__ == canonical
    ]
    assert classes
    for name, value in classes:
        assert getattr(old_module, name) is value, name
