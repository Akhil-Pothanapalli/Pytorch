# Transformer from Scratch using PyTorch

This repository contains an implementation of a Transformer model from scratch using PyTorch. The implementation includes modules for self-attention, transformer blocks, encoder, and decoder.

## Motivation

This implementation was undertaken as a learning milestone to understand the inner workings of the Transformer architecture.

## Usage

To use the Transformer model, follow these steps:

1. **Instantiate the Model:** Initialize the Transformer model by specifying parameters such as vocabulary sizes, embedding size, number of layers, etc.

2. **Prepare Data:** Prepare your source and target sequences as tensors. Ensure that padding indices are correctly identified.

3. **Train the Model:** Train the instantiated model using your dataset.

4. **Inference:** Use the trained model for inference tasks.

### Example

```python
import torch
from transformer import Transformer

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample input
x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

# Model parameters
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10

# Instantiate the model
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)

# Forward pass
out = model(x, trg[:, :-1])
print(out.shape)
