import pickle
import random
import time

import cupy as cp
import numpy as np
from tqdm import tqdm

from mytorch import layers, optimizers
from mytorch.tensor import Tensor
from mytorch.tokenizers import BPE

with open("processed_text.txt", "r") as f:
    text = f.read()

text = text.encode("utf-8")

vocab_size = 32768 // 2
embedding_size = 256
lr = 2e-4
seq_length = 128
batch_size = 16
epochs = 100

loss_func = layers.CrossEntropyLoss()
network = [
    layers.Embedding(vocab_size, embedding_size),
    layers.TransformerLayer(embedding_size, 8),
    layers.TransformerLayer(embedding_size, 8),
    # layers.TransformerLayer(embedding_size, 8),
    layers.Linear(embedding_size, vocab_size, True),
]


network[-1].weights = network[0].weights.transpose_last()
params = [p for layer in network for p in layer.params]
optimizer = optimizers.Adam([(params, lr, 0.01)], warmup_steps=1000)

redo_vocab = False

if redo_vocab:
    tokenizer = BPE(vocab_size)
    start = time.time()
    tokenizer.fit(text)
    print("took", time.time() - start)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    tokenized_text = []
    chunk_size = 1000000
    for i in range(0, len(text), chunk_size):
        tokenized_subtext = tokenizer.tokenize(text[i : i + chunk_size])
        tokenized_text.extend(tokenized_subtext)
    with open("tokenized.pkl", "wb") as f:
        pickle.dump(tokenized_text, f)
else:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("tokenized.pkl", "rb") as f:
        tokenized_text = pickle.load(f)

X = []
for i in range(0, len(tokenized_text), seq_length):
    X.append(tokenized_text[i : i + seq_length])
# last row is unlikely to have the right length
X = X[:-1]
# TODO: I really should seed all the randomness

monitor_toks = tokenizer.tokenize(
    b"This is a simple sentence, nothing too weird going on here."
)
monitor_tensor = Tensor(cp.array(monitor_toks)[None, :])

random.shuffle(X)
best_loss = float("inf")
iterations = 0
for e in range(epochs):
    losses = []
    for i in tqdm(range(0, len(X), batch_size), total=len(X) // batch_size):
        full_batch = cp.array(X[i : i + batch_size], dtype=cp.int32)
        X_batch = Tensor(full_batch[:, :-1])
        y_batch = Tensor(full_batch[:, 1:])

        processed = X_batch
        for j, layer in enumerate(network):
            processed = layer.forward(processed)

        loss = loss_func.forward(processed, y_batch)
        losses.append(float(loss.value))
        loss.compute_gradient()
        optimizer.step()
        if iterations % 100 == 0:
            mean_loss = np.mean(losses)
            print("mean loss", mean_loss)
            losses = []
            if mean_loss < best_loss:
                old_ops = network[-1].weights.operations
                network[-1].weights.operations = []
                with open("network.pkl", "wb") as f:
                    pickle.dump(network, f)
                network[-1].weights.operations = old_ops

            processed = monitor_tensor
            for j, layer in enumerate(network):
                processed = layer.forward(processed)
            pred = cp.argmax(processed.value, axis=-1)[0]
            print(tokenizer.untokenize(pred.tolist()))
        iterations += 1
