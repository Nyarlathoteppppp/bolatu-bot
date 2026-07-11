# Server Deploy

Ubuntu 22.04 server deployment uses Docker Compose.

## Start

```bash
cd /opt/qq-social-agent
docker compose -f docker-compose.server.yml up -d --build
```

## Stop

```bash
cd /opt/qq-social-agent
docker compose -f docker-compose.server.yml down
```

## Logs

```bash
cd /opt/qq-social-agent
docker compose -f docker-compose.server.yml logs -f bot
docker compose -f docker-compose.server.yml logs -f napcat
```

## Update

```bash
cd /opt/qq-social-agent
git pull
docker compose -f docker-compose.server.yml up -d --build
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

