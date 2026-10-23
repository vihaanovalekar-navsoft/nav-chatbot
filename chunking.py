"""Fixed-size token windows for the NavAI documents."""


def chunk_text(text: str, tokenizer, size: int, overlap: int):
    """Return `(token_start, token_count, chunk)` windows over `text`.

    Each window holds `size` text tokens and shares `overlap` tokens with the
    previous window. Offsets map tokens back to the original characters, so
    casing, punctuation, and paragraph breaks stay intact. Raises ValueError
    when `size` is not positive or `overlap` is outside `[0, size)`.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be smaller")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    step = size - overlap

    windows = []
    start = 0
    while start < len(offsets):
        window = offsets[start:start + size]
        chunk = text[window[0][0]:window[-1][1]].strip()
        if chunk:
            windows.append((start, len(window), chunk))
        if start + size >= len(offsets):
            break
        start += step
    return windows
