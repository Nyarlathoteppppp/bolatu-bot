# Local Plugin Manifests

This directory declares the bot capabilities that should gradually move out of `qq_social_agent/plugin.py`.

Current scope is metadata-only: manifests are loaded by `qq_social_agent.plugin_runtime.LocalPluginRegistry`, but plugin code is not dynamically imported or sandboxed yet. This keeps startup safe while giving commands, tools, scheduled tasks, web routes, and event handlers a clear home.

## Manifest shape

```yaml
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
settings: {}
```

## Migration rule

1. Add or update the manifest first.
2. Expose the capability in `/admin/plugins`.
3. Move implementation code out of `qq_social_agent/plugin.py` only after tests exist.
4. Keep runtime execution explicit; do not dynamically import arbitrary plugin code until a permission model and lifecycle hooks are ready.
