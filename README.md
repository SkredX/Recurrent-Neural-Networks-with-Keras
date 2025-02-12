LSTM-Based Sequence-to-Sequence Addition Model

**Overview**

This project implements a sequence-to-sequence numerical addition model using Long Short-Term Memory (LSTM) networks. The model takes in two numbers represented as strings (e.g., "25+38") and predicts their sum (e.g., "63"). It is designed to improve upon traditional RNN-based approaches by leveraging LSTM's ability to handle long-term dependencies in sequential data.

**Features**

LSTM-Based Encoder-Decoder Architecture – Enables efficient sequence learning and generalization.

Embedding Layer for Feature Representation – Reduces sparsity from one-hot encoding.

TimeDistributed Dense Layer – Ensures character-level predictions at each time step.

RepeatVector for Decoder Input – Maintains context from the encoder to decoder.

Categorical Crossentropy Loss with Softmax Output – Helps model learn probability distributions for character sequences.

Learning Rate Scheduler – Dynamically adjusts learning rate for optimal training.

Early Stopping Callback – Prevents overfitting by stopping training when validation loss stagnates.

**Data Representation**

Vocabulary: 0-9 (digits) and + (operator) → Total 11 unique characters.

Max Input Length: 5 (e.g., "99+99" → 5 characters).

Max Output Length: 5 (e.g., "198" is zero-padded to "00198").

One-Hot Encoding: Input and output are represented using 11-dimensional vectors.
