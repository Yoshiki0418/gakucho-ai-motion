"""Keep malformed motion-tool output out of an explicit turn's speech."""

import re

_TOOL_MARKER = re.compile(
    r"<functions\.generate_motion\b|"
    r"assistant\s+to=functions\.generate_motion\b(?:\s+in\s+commentary)?"
)


def _object_end(text: str, start: int) -> int | None:
    """Find the end of an object, including escaped and nested JSON strings."""
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def clean_motion_tool_artifacts(text: str, *, final: bool = False) -> tuple[str, bool]:
    """Return cleaned speech and whether a tool block is still streaming.

    Recognize only the observed internal generate_motion notation. Ordinary
    JSON and dialogue remain intact. The caller retains the original buffer
    while a block is incomplete so its payload cannot become a new sentence.
    """
    parts = []
    cursor = 0
    while match := _TOOL_MARKER.search(text, cursor):
        parts.append(text[cursor : match.start()])
        object_start = text.find("{", match.end())
        if object_start < 0:
            return "".join(parts), not final
        object_end = _object_end(text, object_start)
        if object_end is None:
            return "".join(parts), not final
        end = object_end
        if match.group().startswith("<"):
            closing = text.find(">", object_end)
            if closing < 0:
                return "".join(parts), not final
            end = closing + 1
        else:
            # This literal suffix appeared after the malformed tool JSON.
            # Remove it only here, rather than removing Korean dialogue.
            suffix = "\ub2c8\ub2e4"
            if text.startswith(suffix, end):
                end += len(suffix)
            elif text[end:] and suffix.startswith(text[end:]):
                return "".join(parts), not final
        if end == len(text) and not final:
            return "".join(parts), True
        cursor = end
    remaining = text[cursor:]
    for marker, final_minimum in (
        ("<functions.generate_motion", len("<functions")),
        ("assistant to=functions.generate_motion in commentary", len("assistant to=")),
    ):
        minimum = final_minimum if final else 1
        for length in range(min(len(marker), len(remaining)), minimum - 1, -1):
            if remaining.endswith(marker[:length]):
                parts.append(remaining[:-length])
                return "".join(parts), not final
    parts.append(remaining)
    return "".join(parts), False
