import pickle

import cupy as cp
import numpy as np
from tqdm import tqdm

from mytorch import layers, optimizers
from mytorch.tensor import Tensor
from mytorch.tokenizers import NaiveBPE

with open("/home/scott/Downloads/alice_in_wonderland.txt", "r") as f:
    text = f.read()

text = text.encode("utf-8")

vocab_size = 2048
embedding_size = 256
lr = 3e-4
seq_length = 64
batch_size = 16
epochs = 1000

loss_func = layers.CrossEntropyLoss()
network = [
    layers.Embedding(vocab_size, embedding_size),
    layers.TransformerLayer(embedding_size, 8),
    # layers.TransformerLayer(embedding_size, 8),
    # layers.TransformerLayer(embedding_size, 8),
    layers.Linear(embedding_size, vocab_size, True),
]

optimizer = optimizers.Adam(lr=lr)
network[-1].weights = network[0].weights.transpose_last()

if False:
    tokenizer = NaiveBPE(vocab_size)
    tokenizer.fit([text])
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
else:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

tokenized_text = tokenizer.tokenize(text)

X = []
for i in range(0, len(tokenized_text), seq_length):
    X.append(tokenized_text[i : i + seq_length])

# chuck the last row, it won't have the right shape
X = cp.array(X[:-1], dtype=cp.int32)
best_loss = float("inf")

for e in range(epochs):
    losses = []
    for i in tqdm(range(0, X.shape[0], batch_size), total=X.shape[0] // batch_size):
        X_batch = Tensor(X[i : i + batch_size, :-1])
        y_batch = Tensor(X[i : i + batch_size, 1:])

        processed = X_batch
        for j, layer in enumerate(network):
            processed = layer.forward(processed)

        loss = loss_func.forward(processed, y_batch)
        losses.append(float(loss.value))
        optimizer.step(loss)
    mean_loss = np.mean(losses)
    print(f"mean train loss for epoch {e} was {mean_loss}")
    if mean_loss < best_loss:
        old_ops = network[-1].weights.operations
        network[-1].weights.operations = []
        with open("network.pkl", "wb") as f:
            pickle.dump(network, f)
        network[-1].weights.operations = old_ops
    print(tokenizer.untokenize(X[20].tolist()))

    processed = Tensor(X[20][None, :])
    for j, layer in enumerate(network):
        processed = layer.forward(processed)
    pred = cp.argmax(processed.value, axis=-1)[0]
    print(processed.value.max())
    print(tokenizer.untokenize(pred.tolist()))
