"""Small HTTP request parsing helpers shared by admin resource controllers."""

from __future__ import annotations

from urllib.parse import parse_qs

from starlette.requests import Request


async def admin_form_data(request: Request) -> dict[str, str]:
    body = await request.body()
    parsed = parse_qs(body.decode("utf-8", errors="ignore"), keep_blank_values=True)
    return {
        key: (",".join(values) if key == "frozen_fields" else (values[-1] if values else ""))
        for key, values in parsed.items()
    }


def admin_form_int(form: dict[str, str], key: str) -> int | None:
    value = form.get(key, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
