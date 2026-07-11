# Server Deploy

Ubuntu 22.04 server deployment uses Docker Compose.

The production server lives at:

```bash
/opt/qq-social-agent
```

Keep the Compose project name fixed as `qq-social-agent`; NapCat connects to the bot with
`ws://bot:8080/onebot/v11/ws`, so changing the project/network name can break service discovery.

## Branches

```text
main  Stable branch used by the running server.
dev   Daily development branch. Merge into main after tests pass.
```

Server defaults to `main`:

```bash
cd /opt/qq-social-agent
git switch main
git pull --ff-only origin main
```

For server-side development:

```bash
cd /opt/qq-social-agent
git switch dev
git pull --ff-only origin dev
# edit files, run tests, commit
git push origin dev
```

Runtime files stay on the server and must not be committed:

```text
.env
data/
logs/
server-data/
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

## Update

```bash
cd /opt/qq-social-agent
git switch main
git pull --ff-only origin main
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build bot
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
