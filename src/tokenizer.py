"""
Character-level tokenizer.

No libraries, no magic. We just:
1. Find every unique character in the text
2. Sort them (so the mapping is deterministic)
3. Build char → int and int → char lookup tables

That's it. This is what GPT would use if its vocabulary
were individual characters instead of BPE tokens.
"""


class CharTokenizer:
    def __init__(self, text: str):
        # sorted() gives us a deterministic ordering
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # The two lookup tables — this IS the tokenizer
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integers."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        """Convert a list of integers back to a string."""
        return "".join(self.idx_to_char[i] for i in indices)


if __name__ == "__main__":
    # Quick sanity check
    with open("data/input.txt", "r") as f:
        text = f.read()

    tok = CharTokenizer(text)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Characters: {''.join(tok.idx_to_char[i] for i in range(tok.vocab_size))}")

    # Round-trip test: encode then decode should give back the original
    sample = text[:100]
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    assert sample == decoded, "Round-trip failed!"
    print(f"\nSample: {repr(sample)}")
    print(f"Encoded: {encoded[:50]}...")
    print(f"Decoded: {repr(decoded[:50])}...")
    print("\nRound-trip test passed!")