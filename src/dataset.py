"""
Dataset for character-level language modeling.

The key idea: for next-character prediction, every position
in the text gives us a training example.

If block_size=8 and the text is "First Citizen", then one
training sample might be:

  x = "First Ci"  (input)
  y = "irst Cit"  (target — shifted by one)

Each character in x should predict the next character in y.
The model learns from ALL 8 predictions simultaneously.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from src.tokenizer import CharTokenizer


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        """
        Args:
            data: 1D tensor of encoded characters (the full text as integers)
            block_size: number of characters the model sees at once (context window)
        """
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # Every position (except the last block_size chars) is a valid start
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_datasets(text: str, block_size: int, train_split: float = 0.9):
    """
    Encode text and split into train/val datasets.

    Returns:
        tokenizer, train_dataset, val_dataset
    """
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    # Simple split — first 90% train, last 10% val
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    print(f"Total characters: {len(data):,}")
    print(f"Train: {len(train_data):,} chars → {len(train_dataset):,} samples")
    print(f"Val:   {len(val_data):,} chars → {len(val_dataset):,} samples")
    print(f"Vocab size: {tok.vocab_size}")

    return tok, train_dataset, val_dataset


if __name__ == "__main__":
    with open("data/input.txt", "r") as f:
        text = f.read()

    block_size = 256
    batch_size = 64

    tok, train_ds, val_ds = create_datasets(text, block_size)

    # Create a DataLoader and inspect one batch
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    x_batch, y_batch = next(iter(loader))

    print(f"\nBatch shapes: x={x_batch.shape}, y={y_batch.shape}")
    print(f"Expected: x=({batch_size}, {block_size}), y=({batch_size}, {block_size})")

    # Decode the first sample to verify it makes sense
    print(f"\n--- First sample in batch (x) ---")
    print(tok.decode(x_batch[0].tolist()))
    print(f"\n--- First sample in batch (y) ---")
    print(tok.decode(y_batch[0].tolist()))
    print(f"\n(y should be x shifted by one character)")