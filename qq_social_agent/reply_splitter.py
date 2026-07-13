from __future__ import annotations

import re


def split_reply_messages(text: str, *, max_messages: int) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    if max_messages <= 1:
        return [clean]

    sentences = _split_sentences(clean)
    if len(sentences) <= 1:
        return _split_long_single_sentence(clean, max_messages=max_messages)

    if len(sentences) == 2:
        if len(clean) >= 28 and all(_meaningful_piece(sentence) for sentence in sentences):
            return sentences
        return [clean]

    if max_messages == 2:
        return [sentences[0], "".join(sentences[1:])]

    return [
        *sentences[: max_messages - 1],
        "".join(sentences[max_messages - 1 :]),
    ]


def _split_long_single_sentence(text: str, *, max_messages: int) -> list[str]:
    if len(text) <= 80:
        return [text]

    chunks = [chunk.strip() for chunk in re.split(r"(?<=[，,；;：:、])", text) if chunk.strip()]
    if len(chunks) <= 1:
        return [text[:180].rstrip("，,；;：:、 ") + ("。" if len(text) > 180 else "")]

    parts: list[str] = []
    current = ""
    target_len = 72
    for chunk in chunks:
        if current and len(current) + len(chunk) > target_len and len(parts) < max_messages - 1:
            parts.append(current)
            current = chunk
            continue
        current += chunk

    if current:
        parts.append(current)

    if len(parts) > max_messages:
        head = parts[: max_messages - 1]
        tail = "".join(parts[max_messages - 1 :])
        parts = [*head, tail]

    return [part.strip() for part in parts if part.strip()]


def _split_sentences(text: str) -> list[str]:
    chunks = re.findall(r".+?(?:[。！？!?]+|…{2,}|\.{3,})|.+$", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _meaningful_piece(text: str) -> bool:
    compact = re.sub(r"[\s。！？!?，,；;：:、]+", "", text)
    return len(compact) >= 8
