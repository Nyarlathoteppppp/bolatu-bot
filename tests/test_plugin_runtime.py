from pathlib import Path

from qq_social_agent.plugin_runtime import LocalPluginRegistry, load_local_plugin_manifest


def test_local_plugin_manifest_loads_capabilities(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "market_tools"
    plugin_dir.mkdir()
    manifest = plugin_dir / "manifest.yaml"
    manifest.write_text(
        """
id: market_tools
name: 行情工具
version: 0.1.0
enabled: true
description: 查询美股和加密货币。
entrypoint: qq_social_agent.tools.market
permissions:
  - tool.market
capabilities:
  commands:
    - command: 行情
      description: 查询股票或币价
      permission: tool.market
  tools:
    - name: market_lookup
      handler: qq_social_agent.tools.market.MarketTool
  scheduled_tasks: []
  web_routes: []
  event_handlers: []
""".strip(),
        encoding="utf-8",
    )

    loaded = load_local_plugin_manifest(manifest, loaded_at=123.0)

    assert loaded.plugin_id == "market_tools"
    assert loaded.enabled is True
    assert loaded.permissions == ("tool.market",)
    assert loaded.capability_counts()["commands"] == 1
    assert loaded.capabilities["tools"][0].target == "qq_social_agent.tools.market.MarketTool"
    assert loaded.loaded_at == 123.0


def test_local_plugin_registry_keeps_disabled_and_enabled_separate(tmp_path: Path) -> None:
    (tmp_path / "enabled").mkdir()
    (tmp_path / "enabled" / "manifest.yaml").write_text(
        "id: enabled\nname: Enabled\ncapabilities:\n  commands: [ping]\n",
        encoding="utf-8",
    )
    (tmp_path / "disabled").mkdir()
    (tmp_path / "disabled" / "manifest.yaml").write_text(
        "id: disabled\nname: Disabled\nenabled: false\ncapabilities:\n  tools: [hidden_tool]\n",
        encoding="utf-8",
    )

    registry = LocalPluginRegistry(tmp_path)
    registry.reload()

    assert [plugin.plugin_id for plugin in registry.plugins()] == ["disabled", "enabled"]
    assert [plugin.plugin_id for plugin in registry.enabled_plugins()] == ["enabled"]
    payload = registry.status_payload()
    assert payload["total"] == 2
    assert payload["enabled"] == 1
    assert payload["disabled"] == 1
    assert payload["capability_counts"]["commands"] == 1
    assert payload["capability_counts"]["tools"] == 0


def test_local_plugin_registry_records_bad_manifest_without_crashing(tmp_path: Path) -> None:
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.yaml").write_text("name: Missing id\n", encoding="utf-8")
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "manifest.yaml").write_text("id: good\nname: Good\n", encoding="utf-8")

    registry = LocalPluginRegistry(tmp_path)
    registry.reload()

    assert [plugin.plugin_id for plugin in registry.plugins()] == ["good"]
    assert len(registry.errors) == 1
    assert "missing required field: id" in registry.errors[0].error
