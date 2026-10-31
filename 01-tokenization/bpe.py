"""Step 1: a byte-level BPE tokenizer, written from scratch.

Idea: start from raw bytes (vocab 0-255), then repeatedly merge the most
frequent adjacent pair of tokens into a new token.

Run the demo:  uv run python 01-tokenization/bpe.py
"""

Pair = tuple[int, int]


def get_pair_counts(ids: list[int]) -> dict[Pair, int]:
    """Count how often each adjacent pair (a, b) occurs in ids.

    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in ids with `new_id`, left to right.

    Example: merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
    """
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2  # skip both items of the pair
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges: dict[Pair, int] = {}  # (a, b) -> new token id, in learned order

    def train(self, text: str, vocab_size: int) -> None:
        """Learn (vocab_size - 256) merges from text."""
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            counts = get_pair_counts(ids)
            best = max(counts, key=counts.get)
            new_id = 256 + i
            ids = merge(ids, best, new_id)
            self.merges[best] = new_id

    def encode(self, text: str) -> list[int]:
        """Text -> token ids. Apply merges in the order they were learned."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # the pair learned earliest has the lowest merge id; unknown pairs rank last
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing left to merge
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text."""
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():  # dict keeps learned order
            vocab[new_id] = vocab[a] + vocab[b]
        data = b"".join(vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


if __name__ == "__main__":
    text = open(__file__).read() * 5  # tiny toy corpus: this file itself
    tok = BPETokenizer()
    tok.train(text, vocab_size=300)

    sample = "def merge(ids): return ids"
    ids = tok.encode(sample)
    print(f"{len(sample.encode())} bytes -> {len(ids)} tokens")
    assert tok.decode(ids) == sample, "round trip failed"
    print("round trip OK")
