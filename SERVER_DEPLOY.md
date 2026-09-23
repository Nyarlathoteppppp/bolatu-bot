# Server Deploy

Ubuntu 22.04 server deployment uses Docker Compose.

The production server lives at:

```bash
/opt/qq-social-agent
```

Keep the Compose project name fixed as `qq-social-agent`; NapCat connects to the bot with
`ws://bot:8080/onebot/v11/ws`, so changing the project/network name can break service discovery.

## Branches

The production checkout was on `live-hotfix-20260917` on 2026-09-23. Check the branch and working tree before making changes:

```bash
cd /opt/qq-social-agent
git branch --show-current
git status --short
```

Commit and push the branch that is actually deployed. Do not switch a running checkout to `main` or `dev` as part of a routine update.

Runtime files stay on the server and must not be committed:

```text
.env
data/
logs/
server-data/
```

The repository had a tracked `data/bot.sqlite3` in older commits. Removing it from the current tree does not remove those historical copies.

## Code-only update

The Python package is bind-mounted into the bot container. Run tests on the Ubuntu host, then recreate only `bot`:

```bash
cd /opt/qq-social-agent
PYTHONPATH=. python3 -m pytest -q --tb=short
napcat_before=$(docker inspect -f '{{.State.StartedAt}}' napcat)
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --no-deps --force-recreate bot
curl -fsS http://127.0.0.1:8080/readyz
test "$napcat_before" = "$(docker inspect -f '{{.State.StartedAt}}' napcat)"
```

## Start

```bash
cd /opt/qq-social-agent
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build
```

## Stop

```bash
cd /opt/qq-social-agent
docker compose -p qq-social-agent -f docker-compose.server.yml down
```

## Logs

```bash
cd /opt/qq-social-agent
docker compose -p qq-social-agent -f docker-compose.server.yml logs -f bot
docker compose -p qq-social-agent -f docker-compose.server.yml logs -f napcat
```

## NapCat WebUI

The compose file binds NapCat WebUI to `127.0.0.1:6099`.

Use SSH tunnel from your Mac:

```bash
ssh -L 6099:127.0.0.1:6099 qqbot-server
```

Then open:

```text
http://127.0.0.1:6099/webui
```

Configure OneBot v11 reverse WebSocket:

```text
ws://bot:8080/onebot/v11/ws
```
