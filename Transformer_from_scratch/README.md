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

## Modules
**SelfAttention**
This module implements self-attention mechanism

**TransformerBlock**
Represents a single transformer block consisting of multi-head self-attention and feed-forward neural network

**Encoder**
The Encoder module processes the input sequence and generates context vectors

**DecoderBlock**
A single block of the decoder part of the Transformer model, consisting of self-attention, encoder-decoder attention and feed-forward layers.

**Decoder**
The Decoder module generatesv the output sequence based on the context vectors from the encoder.

**Transformer**
The main Transformer model that encapsulates the encoder and decoder.

## Requirements

1. Pytorch
2. TorchText(for data preprocessing)

## Acknowledgements

This implementation is inspired by the Transformer architecture proposed in the paper "Attention is All You Need" by Vaswani et al.

I'm thankful to Aladdin Persson and his tutorial - https://www.youtube.com/watch?v=U0s0f995w14 ,helped me on this learning journey.
