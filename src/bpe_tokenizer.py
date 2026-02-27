"""
Byte-Pair Encoding (BPE) tokenizer, built from scratch.

This is the same algorithm used by GPT-2/3/4. The idea:

1. Start with a vocabulary of individual characters (or bytes)
2. Count every adjacent pair of tokens in the training text
3. Merge the most frequent pair into a single new token
4. Repeat steps 2-3 for N merges
5. You now have a vocabulary of characters + N subword tokens

Example with 3 merges on "aaabdaaabac":
  Start:  [a, a, a, b, d, a, a, a, b, a, c]
  Merge 1: "aa" → "Z"  →  [Z, a, b, d, Z, a, b, a, c]   (most frequent pair was "aa")
  Merge 2: "Za" → "Y"  →  [Y, b, d, Y, b, a, c]          (most frequent pair was "Za")
  Merge 3: "Yb" → "X"  →  [X, d, X, a, c]                 (most frequent pair was "Yb")

After training, encoding new text applies these merges in the same order.

Why BPE over character-level?
- Shorter sequences (fewer tokens per word → faster training, longer effective context)
- Meaningful subword units ("ing", "tion", "the" become single tokens)
- Handles rare/unknown words by falling back to characters
- This is what real LLMs use
"""

import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (tok_a, tok_b) → new_token_id
        self.vocab = {}           # token_id → bytes
        self.inverse_vocab = {}   # bytes → token_id
        self.merge_list = []      # ordered list of merges for encoding

    def _get_pair_counts(self, token_lists):
        """
        Count frequency of every adjacent pair across all token sequences.

        Args:
            token_lists: list of lists of token ids
        Returns:
            Counter of (tok_a, tok_b) → count
        """
        counts = Counter()
        for tokens in token_lists:
            for i in range(len(tokens) - 1):
                counts[(tokens[i], tokens[i + 1])] += 1
        return counts

    def _merge_pair(self, token_lists, pair, new_id):
        """
        Replace every occurrence of `pair` in all token sequences
        with `new_id`.

        This is the core BPE operation — we scan through each sequence
        and wherever we see (pair[0], pair[1]) adjacent, we replace
        them with a single new_id token.
        """
        result = []
        for tokens in token_lists:
            new_tokens = []
            i = 0
            while i < len(tokens):
                # Check if current position matches the pair
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_id)
                    i += 2  # Skip both tokens
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            result.append(new_tokens)
        return result

    def train(self, text, num_merges=256, verbose=True):
        """
        Train BPE on a text corpus.

        Args:
            text: training text (string)
            num_merges: how many merge operations to learn
                        final vocab_size = 256 (bytes) + num_merges
            verbose: print progress
        """
        # Step 1: Convert text to bytes, then to list of byte values
        # We work at the byte level (like GPT-2) so we can handle ANY text
        text_bytes = text.encode("utf-8")

        # Split into "words" (sequences separated by spaces, keeping the space)
        # This prevents merges across word boundaries, which is what GPT-2 does
        # For simplicity, we split on newlines and spaces
        chunks = []
        current = []
        for b in text_bytes:
            current.append(b)
            # Split after spaces and newlines to create natural boundaries
            if b in (ord(" "), ord("\n")):
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        # Initialize vocabulary with all 256 possible bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        if verbose:
            print(f"Training BPE with {num_merges} merges...")
            print(f"Initial chunks: {len(chunks):,}")
            total_tokens = sum(len(c) for c in chunks)
            print(f"Initial tokens: {total_tokens:,}")

        # Step 2: Iteratively find and merge the most frequent pair
        for i in range(num_merges):
            # Count all adjacent pairs
            pair_counts = self._get_pair_counts(chunks)

            if not pair_counts:
                print(f"No more pairs to merge at step {i}")
                break

            # Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0]
            pair, count = best_pair

            if count < 2:
                print(f"No pair appears more than once at step {i}")
                break

            # Create new token by concatenating the bytes of the pair
            new_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab[next_id] = new_bytes

            # Record this merge
            self.merges[pair] = next_id
            self.merge_list.append(pair)

            # Apply the merge to all chunks
            chunks = self._merge_pair(chunks, pair, next_id)

            if verbose and (i + 1) % 50 == 0:
                total_tokens = sum(len(c) for c in chunks)
                pair_str = self.vocab[pair[0]] + self.vocab[pair[1]]
                print(
                    f"  Merge {i+1:4d}: "
                    f"{repr(pair_str.decode('utf-8', errors='replace')):>10s} "
                    f"(count={count:,}) → token {next_id} | "
                    f"total tokens: {total_tokens:,}"
                )

            next_id += 1

        # Build inverse vocab for encoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        total_tokens = sum(len(c) for c in chunks)
        if verbose:
            print(f"\nDone! Vocab size: {self.vocab_size}")
            print(f"Final token count: {total_tokens:,}")
            ratio = len(text_bytes) / total_tokens
            print(f"Compression ratio: {ratio:.2f}x")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        """
        Encode text to token ids by applying learned merges in order.

        The key insight: we apply merges in the SAME ORDER they were learned.
        This ensures consistent tokenization.
        """
        # Start with raw bytes
        tokens = list(text.encode("utf-8"))

        # Apply each merge in order
        for pair in self.merge_list:
            new_id = self.merges[pair]
            # Scan and merge (same logic as training)
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, tokens):
        """Decode token ids back to text."""
        byte_list = b"".join(self.vocab[t] for t in tokens)
        return byte_list.decode("utf-8", errors="replace")

    def save(self, path):
        """Save the tokenizer to a JSON file."""
        data = {
            "merge_list": self.merge_list,
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved tokenizer to {path}")

    @classmethod
    def load(cls, path):
        """Load a tokenizer from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.merge_list = [tuple(p) for p in data["merge_list"]]
        tok.merges = {}
        next_id = 256
        for pair in tok.merge_list:
            tok.merges[pair] = next_id
            next_id += 1
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        return tok


# -----------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------

if __name__ == "__main__":
    with open("data/input.txt", "r") as f:
        text = f.read()

    # Train with 256 merges → vocab size = 512 (256 bytes + 256 merges)
    tok = BPETokenizer()
    tok.train(text, num_merges=256)

    # Round-trip test
    sample = text[:500]
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    assert decoded == sample, f"Round-trip failed!\nOriginal: {repr(sample[:100])}\nDecoded:  {repr(decoded[:100])}"
    print(f"\nRound-trip test passed!")

    # Show compression
    char_tokens = len(sample)
    bpe_tokens = len(encoded)
    print(f"Sample: {char_tokens} chars → {bpe_tokens} BPE tokens ({char_tokens/bpe_tokens:.2f}x compression)")

    # Show what some tokens look like
    print(f"\nSample token examples:")
    for t in encoded[:30]:
        print(f"  Token {t:4d} → {repr(tok.decode([t]))}")

    # Save for later use
    tok.save("data/bpe_tokenizer.json")

    # Test loading
    tok2 = BPETokenizer.load("data/bpe_tokenizer.json")
    encoded2 = tok2.encode(sample)
    assert encoded == encoded2, "Load/save round-trip failed!"
    print(f"\nSave/load test passed!")