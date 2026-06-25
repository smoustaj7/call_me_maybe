

def bytes_to_unicode():
    """
    returns a dict mapping every byte value (0-255) to a printable unicode
    character.
    printable ASCII/Latin-1 ranges map to themselves; everything else gets
    pushed into a free range starting at 256.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def text_to_bpe_alphabet(text: str, byte_encoder: dict) -> str:
    """
    converts a string into a sequence of BPE alphabet characters.
    """
    bpe_alphabet = ""
    for byte in text.encode("utf-8"):
        bpe_alphabet += byte_encoder[byte]
    return bpe_alphabet


def load_merges(merges_path: str) -> dict:
    """
    loads the merges file and returns a dictionary of merges.
    """
    merges = {}
    with open(merges_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            merges[(parts[0], parts[1])] = i
    return merges


def apply_bpe_merges(symbols: list[str], merges: dict) -> list[str]:
    while True:
        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)]
        candidates = [(merges[p], p) for p in pairs if p in merges]
        if not candidates:
            break
        best_rank, best_pair = min(candidates)
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and \
                    (symbols[i], symbols[i+1]) == best_pair:
                new_symbols.append(symbols[i] + symbols[i+1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols


def split_into_chunks(bpe_text: str) -> list[str]:
    chunks = []
    current = ""
    for char in bpe_text:
        if char == 'Ġ' and current:
            chunks.append(current)
            current = char
        else:
            current += char
    if current:
        chunks.append(current)
    return chunks


def encode(text: str, byte_encoder: dict, merges: dict, vocab: dict)\
        -> list[int]:
    bpe_text = text_to_bpe_alphabet(text, byte_encoder)
    chunks = split_into_chunks(bpe_text)

    token_ids = []
    for chunk in chunks:
        symbols = apply_bpe_merges(list(chunk), merges)
        for symbol in symbols:
            token_ids.append(vocab[symbol])

    return token_ids


def decode(token_ids: list[int], inv_vocab: dict, byte_decoder: dict) -> str:
    bpe_text = "".join(inv_vocab[i] for i in token_ids)
    byte_values = bytes(byte_decoder[char] for char in bpe_text)
    return byte_values.decode("utf-8")
