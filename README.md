# Shakespeare Transformer Model Training from scratch

Training a small GPT model on the works of Shakespeare using PyTorch, inspired by [Andrej Karpathy's YouTube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY). This project includes data preprocessing, model training, and text generation tools.

## Introduction

### Transformer Models

Transformer models, introduced in the groundbreaking paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), have transformed the field of natural language processing (NLP). These models are designed to handle sequential data, making them ideal for tasks such as language modeling, translation, and text generation. Here’s a deeper dive into how transformers work and their key components.

#### Architecture of Transformers

Transformers consist of two main parts: the **encoder** and the **decoder**. However, for models like GPT (Generative Pretrained Transformer), only the decoder part is used. Here’s a breakdown of the key components:

1. **Self-Attention Mechanism**: This is the core innovation of transformers. Self-attention allows the model to weigh the importance of different words in a sentence relative to each other. This mechanism helps the model to focus on relevant parts of the input when making predictions.

2. **Positional Encoding**: Since transformers do not process data sequentially like RNNs or LSTMs, they need a way to incorporate the order of words in a sentence. Positional encodings are added to the input embeddings to give the model information about the position of each word.

3. **Multi-Head Attention**: Instead of performing a single self-attention operation, transformers use multiple attention heads. Each head can focus on different parts of the sentence, allowing the model to capture various aspects of word relationships.

4. **Feed-Forward Neural Network**: After the attention mechanism, the output is passed through a feed-forward neural network, applied separately and identically to each position.

5. **Layer Normalization and Residual Connections**: These techniques help stabilize and improve the training of deep networks by normalizing the output of each layer and adding the input of a layer to its output (residual connection).

6. **Stacked Layers**: Transformers consist of multiple layers (blocks) of attention and feed-forward networks. Each layer helps the model learn more complex representations of the input data.

#### Detailed Explanation of Hyperparameters

Here are the critical hyperparameters that define a transformer model's architecture and performance:

- **Number of Attention Heads**: This is the number of separate attention mechanisms in the multi-head attention module. More heads allow the model to focus on different parts of the input simultaneously.
  
- **Number of Layers (Attention Blocks)**: This refers to the number of stacked encoder or decoder layers in the model. More layers typically mean the model can learn more complex patterns but also require more computational resources.

- **Embedding Dimension**: This is the size of the vector in which words are represented. Higher dimensions can capture more intricate details about word relationships but increase the model size.

- **Context Size (Sequence Length)**: This is the maximum length of input sequences the model can handle. A larger context size allows the model to consider longer dependencies in the text.

- **Dropout Rate**: Dropout is a regularization technique where a fraction of neurons is randomly set to zero during training to prevent overfitting. The dropout rate controls how many neurons are dropped.

- **Learning Rate**: This determines how quickly the model's parameters are updated during training. The learning rate needs to be carefully chosen to balance between quick convergence and stable learning.
  
<div style="text-align: center;">
  <img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="Attention Research" width="500">
</div>

### Comparing with GPT-3 Hyperparameters

To put things in perspective, here are some of the hyperparameters for GPT-3, one of the most advanced transformer models developed by OpenAI:

- **Number of Parameters**: 175 billion
- **Number of Attention Heads**: 96
- **Number of Layers (Attention Blocks)**: 96
- **Context Size (Sequence Length)**: 2048 tokens
- **Embedding Dimension**: 12,288
- **Dropout Rate**: While not explicitly specified, it is typically a small fraction (e.g., 0.1)
- **Learning Rate**: Dynamic, often starting around 1e-4 and adjusted during training
GPT-3’s immense size allows it to perform a wide range of tasks with impressive accuracy and fluency. However, training such a large model requires significant computational resources and data.

## Project Structure
- **data_preprocessing**: Scripts and tools for preprocessing Shakespeare's texts.
- **Bigram Language Model training**: Training a Bigram Language Model which only predicts the next token based on the previous token
- **Transformer model_training**: Code for training the Transformer model from scratch using PyTorch.
- **text_generation**: Tools for generating text using the trained model.
- **result Comparing**: Comparing the text generated by Bigram language model and Transformer model
