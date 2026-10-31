# transformer-from-scratch

The Transformer written by hand in PyTorch, without `nn.Transformer` or HuggingFace. The repo has two stages: a small educational version with a byte-pair-encoding tokenizer and a Decision Transformer, and a second version with a modern Llama-style language model.

## Version 1: educational models

- A character-level GPT trained on Shakespeare, with multi-head causal attention, pre-norm residual blocks, learned positional embeddings and a cosine learning-rate schedule with warmup.
- A byte-pair-encoding (BPE) tokenizer with 512 merges, about 2x compression over characters, and a BPE language model.
- A Decision Transformer trained offline on CartPole episodes, conditioned on return-to-go.
- Attention-map plots and an ablation study.

### Results

| Language model | Parameters | Train loss | Val loss |
|---|---|---|---|
| Character level | 841K | 1.43 | 1.62 |
| BPE, before regularisation | 3M | 2.11 | 3.51 |
| BPE, after dropout and weight decay with fewer layers | 2.1M | 2.92 | 3.54 |

The regularisation cut the BPE model's train/validation gap from 1.41 to 0.62. The character model trained for about 2.8 hours on CPU.

Ablation, 1500 steps, validation loss: baseline (4 layers, 4 heads) 1.85, one head 1.81, one layer 1.97. Removing depth hurt more than removing heads at this scale.

Decision Transformer on CartPole, return achieved for each target return:

| Target | Achieved (mean and std) | Max |
|---|---|---|
| 10 | 10.1 +/- 0.3 | 11 |
| 50 | 45.3 +/- 9.0 | 66 |
| 200 | 71.5 +/- 39.1 | 198 |
| 500 | 76.0 +/- 70.2 | 491 |
| Random policy | 22.2 +/- 14.1 | |

Low targets are followed closely. High targets are reached only occasionally.

### Weights

Published at [adimunot/transformer-from-scratch](https://huggingface.co/adimunot/transformer-from-scratch), as inference-only step-5000 checkpoints with a `config.json` each:

| Folder | Tokenizer | Vocabulary | d_model | Heads | Layers | Context |
|---|---|---|---|---|---|---|
| `char/` | character | 65 | 128 | 4 | 4 | 256 |
| `bpe/` | BPE | 768 | 192 | 6 | 6 | 256 |

    cfg = json.load(open(hf_hub_download(repo, "bpe/config.json")))
    model.load_state_dict(load_file(hf_hub_download(repo, "bpe/model.safetensors")))

## Version 2: Llama-style language model (`src/llm/`)

A 101M-parameter language model with a modern architecture, planned for pretraining on 10B tokens of FineWeb-Edu and instruction tuning on smol-smoltalk, on a single rented H100.

Status: the code and tests are done (31 tests, CPU, a few seconds). The training runs have not been done, so there are no results yet.

Compared with version 1:

| Component | Version 1 | Version 2 |
|---|---|---|
| Positions | learned embeddings | RoPE |
| Normalisation | LayerNorm | RMSNorm, no biases |
| MLP | GELU | SwiGLU |
| Attention | per-head linears | fused QKV, grouped-query attention (12 query heads, 4 KV heads) |
| Attention kernel | full T x T matrix | `scaled_dot_product_attention`, with the manual path kept and tested to match |
| Generation | recompute everything | static KV cache, tested to give identical tokens |
| Embeddings | separate output head | tied with the token embedding |
| Tokenizer | character or small BPE | 32K byte-level BPE |
| Training | CPU loop | bf16, gradient accumulation, `torch.compile`, resumable checkpoints, throughput and MFU metering |

Planned phases: a 25M model on TinyStories to prove the pipeline, pretraining the 101M model, then supervised fine-tuning with the ChatML template and a loss masked to the response.

    pip install -r requirements-v2.txt
    pytest tests/ -v
    python scripts/train_tokenizer.py --out data/tokenizer_32k.json
    python scripts/prepare_data.py --dataset roneneldan/TinyStories \
        --tokenizer data/tokenizer_32k.json --out data/shards_tinystories --val-tokens 5000000
    python -m src.llm.train configs/tinystories_25m.py

## Setup for version 1

    conda create -n transformer python=3.11 -y
    conda activate transformer
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install numpy matplotlib tqdm
    curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    python -m src.train            # character model
    python -m src.generate         # interactive generation
    python -m src.bpe_tokenizer    # train the BPE tokenizer
    python notebooks/explore.py    # attention maps and ablations

The `course/` folder has nine chapters on the version 1 code.
