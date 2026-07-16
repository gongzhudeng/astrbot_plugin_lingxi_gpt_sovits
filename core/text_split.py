def parse_custom_symbols(value: str) -> set[str]:
    """Parse configured symbols, supporting ``\n`` as a newline escape."""
    symbols: set[str] = set()
    index = 0

    while index < len(value):
        if value[index : index + 2] == "\\n":
            symbols.add("\n")
            index += 2
            continue
        symbols.add(value[index])
        index += 1

    return symbols


def split_text_with_symbols(text: str, configured_symbols: str) -> str:
    """Convert custom split boundaries to newlines for GPT-SoVITS cut0."""
    symbols = parse_custom_symbols(configured_symbols)
    if not symbols:
        return text

    parts: list[str] = []
    for char in text:
        parts.append(char)
        if char != "\n" and char in symbols:
            parts.append("\n")

    return "".join(parts)
