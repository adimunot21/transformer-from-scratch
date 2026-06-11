"""Tokenizer tests: round-trip fidelity and special-token reservation."""

import pytest

from src.llm.tokenizer import (ENDOFTEXT, IM_END, IM_START, SPECIAL_TOKENS,
                               Tokenizer, train_tokenizer)

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs!",
    "Transformers process sequences with attention.",
    "café naïve über — unicode soup 🦊🍜",
    "def main():\n    print('hello')\n",
] * 50


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_tokenizer(iter(CORPUS), vocab_size=600)


def test_special_tokens_reserved(tok):
    # All four specials exist and occupy the first ids (trained first).
    ids = [tok.eot_id, tok.im_start_id, tok.im_end_id, tok.pad_id]
    assert ids == [0, 1, 2, 3]


def test_roundtrip_ascii(tok):
    text = "The quick brown fox."
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_unicode(tok):
    # Byte-level BPE must round-trip ANY string, even unseen scripts.
    text = "héllo wörld 日本語 🦊 \t tabs\nnewlines"
    assert tok.decode(tok.encode(text)) == text


def test_add_eot(tok):
    ids = tok.encode("hello", add_eot=True)
    assert ids[-1] == tok.eot_id


def test_save_load_roundtrip(tok, tmp_path):
    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = Tokenizer.load(path)
    text = "Pack my box with five dozen liquor jugs!"
    assert tok2.encode(text) == tok.encode(text)
    assert tok2.eot_id == tok.eot_id


def test_specials_encode_as_single_tokens(tok):
    # Chat template text must tokenize to the reserved ids, not byte soup.
    text = f"{IM_START}user\nhi{IM_END}"
    ids = tok.encode(text)
    assert tok.im_start_id in ids and tok.im_end_id in ids
