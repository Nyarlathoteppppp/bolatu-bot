# Local Plugin Manifests

This directory declares the bot capabilities that should gradually move out of `qq_social_agent/plugin.py`.

Current scope is manifest-driven but allowlisted: manifests are loaded by `qq_social_agent.plugin_runtime.LocalPluginRegistry`, the registry builds an execution plan, and `qq_social_agent.plugin` binds known targets to local implementations. Plugin code is not dynamically imported or sandboxed yet, so startup remains safe while commands, tools, scheduled tasks, web routes, and event handlers have a clear home.

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
3. Runtime registration must check the manifest capability before enabling tools or scheduled tasks.
4. Move implementation code out of `qq_social_agent/plugin.py` only after tests exist.
5. Keep runtime execution explicit; do not dynamically import arbitrary plugin code until a permission model and lifecycle hooks are ready.
