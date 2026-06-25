"""
Tokenizer for v2: byte-level BPE via the HuggingFace `tokenizers` library.

The from-scratch BPE in src/bpe_tokenizer.py implements the same algorithm
and remains the reference for HOW it works (course chapter 7). This wrapper
exists because pure-Python BPE cannot tokenize ~30GB of pretraining data —
the Rust implementation is ~1000x faster.

The four special tokens are reserved AT TRAINING TIME and never change:
the tokenizer is frozen once data shards are built, because retraining it
would silently invalidate every shard and checkpoint. <|im_start|>/<|im_end|>
exist from day one so Phase 3 (chat SFT) needs no embedding resize.
"""

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

ENDOFTEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD = "<|pad|>"
SPECIAL_TOKENS = [ENDOFTEXT, IM_START, IM_END, PAD]


class Tokenizer:
    """Thin wrapper: encode/decode plus named special-token ids."""

    def __init__(self, tok: HFTokenizer):
        self.tok = tok
        self.eot_id = tok.token_to_id(ENDOFTEXT)
        self.im_start_id = tok.token_to_id(IM_START)
        self.im_end_id = tok.token_to_id(IM_END)
        self.pad_id = tok.token_to_id(PAD)
        assert None not in (self.eot_id, self.im_start_id, self.im_end_id, self.pad_id), \
            "tokenizer is missing reserved special tokens"

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        ids = self.tok.encode(text).ids
        if add_eot:
            ids.append(self.eot_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(path)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        return cls(HFTokenizer.from_file(path))


def train_tokenizer(text_iterator, vocab_size: int = 32768) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer (GPT-2 style: 256 byte symbols as the
    base alphabet, so ANY string round-trips losslessly — no <unk> needed).

    text_iterator: yields strings (e.g. documents from a HF dataset).
    """
    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iterator, trainer=trainer)
    return Tokenizer(tok)
