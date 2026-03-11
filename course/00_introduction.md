# Chapter 0: Introduction — What We're Building and Why

## What This Course Is

This is a hands-on course where you build a GPT-style language model from scratch in PyTorch. Not "from scratch" meaning "download a library and call `.fit()`" — from scratch meaning you will write every component of the Transformer architecture yourself: the attention mechanism, the feed-forward network, the positional encoding, the training loop, the text generation, all of it.

By the end, you'll have:
1. A working language model that generates Shakespeare-style text
2. A byte-pair encoding (BPE) tokenizer — the same algorithm used by GPT-2/3/4
3. A Decision Transformer that controls a CartPole agent using the same architecture
4. Deep understanding of why each piece exists and what it does

## Who This Is For

You should know Python well — classes, functions, list comprehensions, file I/O. You do **not** need to know machine learning, neural networks, or PyTorch. We'll explain everything as we go.

## The Big Picture: What Is a Language Model?

A language model does one thing: given some text, predict what comes next.

```
Input:  "To be or not to"
Output: "b" (predicting the next character)
```

That's it. If you can predict the next character well, you can generate text by predicting one character at a time, appending it, and repeating:

```
"To be or not to" → "b"
"To be or not to b" → "e"
"To be or not to be" → ","
"To be or not to be," → " "
"To be or not to be, " → "t"
...and so on
```

Every large language model — GPT-4, Claude, Gemini — works on this same principle, just at a much larger scale and with subword tokens instead of characters.

## The Architecture: Transformer

The neural network architecture we'll use is called the **Transformer**. It was introduced in the 2017 paper "Attention Is All You Need" and has since taken over virtually all of modern AI.

Here's the full architecture we'll build, shown as a flow:

```
Input text: "ROMEO:\nO, she doth"
         │
         ▼
┌─────────────────────┐
│  Character Tokenizer │  Convert each character to a number
│  "R"→18, "O"→15,... │  (our vocabulary is ~65 characters)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Token Embedding     │  Look up a vector for each number
│  18 → [0.2, -0.1,   │  (learnable, 128 dimensions)
│        0.5, ...]     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Positional Embedding│  Add position information
│  pos 0 → [0.1, 0.3] │  "Where am I in the sequence?"
│  pos 1 → [0.4, 0.2] │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Transformer Block (repeated 4×)    │
│  ┌───────────────────────────────┐  │
│  │ Layer Norm                    │  │
│  │ → Multi-Head Self-Attention   │  │  "What should I pay attention to?"
│  │ → Add residual connection     │  │
│  ├───────────────────────────────┤  │
│  │ Layer Norm                    │  │
│  │ → Feed-Forward Network        │  │  "Process the gathered info"
│  │ → Add residual connection     │  │
│  └───────────────────────────────┘  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────┐
│  Final Layer Norm    │
│  → Linear Projection │  Convert back to vocabulary size
│  → Softmax           │  Get probability for each character
└─────────────────────┘
              │
              ▼
Output: probability distribution over 65 characters
        "t" has highest probability → predict "t"
```

Don't worry if this doesn't make sense yet. We'll build each box one at a time, and by the end you'll understand every piece.

## Key Concepts We'll Learn

Before we start coding, here's a map of the key ideas you'll encounter. You don't need to understand these now — think of this as a preview that you can refer back to.

### Tensors
A tensor is a multi-dimensional array of numbers. If you know NumPy arrays, tensors are the same thing but optimized for neural network operations.

- A single number: `5` (scalar, 0 dimensions)
- A list of numbers: `[1, 2, 3]` (vector, 1 dimension)
- A grid of numbers: `[[1,2],[3,4]]` (matrix, 2 dimensions)
- A cube of numbers: 3 dimensions (a "batch" of matrices)

In our model, a typical tensor has shape `(batch_size, sequence_length, embedding_dimension)` — for example `(64, 256, 128)` means "64 sequences, each 256 characters long, each character represented by 128 numbers."

### Neural Network Basics
A neural network is a function that takes numbers in, transforms them through a series of **layers**, and produces numbers out. Each layer has **parameters** (numbers the network can adjust) that are learned during training. The key operation in most layers is a **linear transformation** — multiply by a weight matrix and add a bias:

```
output = input × W + b
```

where `W` (weights) and `b` (bias) are the learnable parameters.

### Training: How the Network Learns
Training is an iterative process:

1. **Forward pass**: Feed input through the network, get a prediction
2. **Loss**: Measure how wrong the prediction is (a single number)
3. **Backward pass**: Calculate how each parameter contributed to the error (gradients)
4. **Update**: Adjust each parameter slightly to reduce the error

Repeat millions of times. The network gradually gets better.

The "backward pass" uses an algorithm called **backpropagation**, which is just the chain rule from calculus applied automatically. PyTorch handles this for us — we just call `loss.backward()`.

### Attention: The Core Innovation
The Transformer's key idea is **self-attention**: each position in the sequence can "look at" every other position and decide how much information to gather from it.

In a sentence like "The cat sat on the mat because **it** was tired", the word "it" needs to know that it refers to "cat" (not "mat"). Attention lets "it" assign a high weight to "cat" and a low weight to "mat".

We'll implement this from scratch. The math is elegant and surprisingly simple.

### Embeddings: Turning Characters into Vectors
Neural networks operate on numbers, not characters. An **embedding** converts each character into a vector (list of numbers). The character "A" might become `[0.2, -0.1, 0.5, 0.8, ...]` — a point in 128-dimensional space. Similar characters end up at nearby points: "a" and "A" will have similar vectors.

These vectors are learned during training — the network discovers useful representations.

## Tools You'll Need

### Python 3.11
The programming language. You should already have this.

### PyTorch
The deep learning framework. It provides:
- **Tensors**: GPU-accelerated multi-dimensional arrays
- **Autograd**: Automatic gradient computation (the "backward pass")
- **nn module**: Building blocks for neural networks (Linear, Embedding, etc.)

We use PyTorch as our foundation but build the Transformer architecture ourselves.

### Conda (Miniforge)
Python environment manager. Keeps our project's packages isolated.

### What We DON'T Use
- No `nn.Transformer` (PyTorch's built-in Transformer — we build our own)
- No `nn.MultiheadAttention` (we write attention from scratch)
- No HuggingFace `transformers` library
- No pre-trained models

## Environment Setup

### Step 1: Create a Conda Environment

Open your terminal:

```bash
conda create -n transformer python=3.11 -y
conda activate transformer
```

This creates an isolated Python environment. Everything we install goes here, not in your system Python.

### Step 2: Install PyTorch

For **CPU-only** (most laptops without a dedicated NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For **GPU** (NVIDIA GPU with CUDA):
```bash
pip install torch torchvision torchaudio
```

The `--index-url` flag for CPU ensures you get a smaller, CPU-only build.

### Step 3: Install Other Packages

```bash
pip install numpy matplotlib tqdm
```

- **numpy**: Array operations (used minimally — PyTorch handles most of this)
- **matplotlib**: Plotting (for loss curves and attention visualizations)
- **tqdm**: Progress bars (optional but nice)

### Step 4: Create the Project Structure

```bash
mkdir -p ~/projects/transformer-from-scratch/{data,src,checkpoints,notebooks,course}
cd ~/projects/transformer-from-scratch
touch src/__init__.py src/tokenizer.py src/model.py src/dataset.py src/train.py src/generate.py
touch README.md requirements.txt
```

This gives us:
```
transformer-from-scratch/
├── data/          ← training data goes here
├── src/           ← our Python modules
├── checkpoints/   ← saved model weights
├── notebooks/     ← experiments and visualizations
├── course/        ← these course files
└── requirements.txt
```

### Step 5: Download Training Data

```bash
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

This is ~1.1MB of Shakespeare — all of his plays concatenated into one text file. It looks like:

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

### Step 6: Freeze Requirements and Initialize Git

```bash
pip freeze > requirements.txt
git init
cat << 'EOF' > .gitignore
__pycache__/
*.pyc
.DS_Store
checkpoints/*.pt
data/input.txt
*.egg-info/
.ipynb_checkpoints/
EOF
git add .
git commit -m "Phase 0: project scaffold and environment"
```

### Step 7: Sanity Check

Run this to verify everything works:

```bash
python -c "
import torch
import numpy as np
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
x = torch.randn(2, 3)
print(f'Test tensor:\n{x}')
print('All good!')
"
```

You should see:
- A PyTorch version (2.x)
- `CUDA available: False` (if CPU) or `True` (if GPU)
- A random 2×3 tensor
- "All good!"

If you get a NumPy compatibility warning, run `pip install "numpy<2"` and try again.

## What's Next

In [Chapter 1](01_data_pipeline.md), we'll build the data pipeline: a character-level tokenizer that converts text to numbers and back, and a PyTorch Dataset that feeds training examples to our model.