import pickle
import random

import cupy as cp
import numpy as np
import torch
from tqdm import trange

from mytorch import layers as mt_layers
from mytorch import optimizers as mt_optimizers
from mytorch.tensor import Tensor
from mytorch.tokenizers import BPE

vocab_size = 32768 // 2
embedding_size = 256
head_count = embedding_size // 32
lr = 2e-4
seq_length = 128
batch_size = 8
warmup_steps = 1000
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("tokenizer.pkl", "rb") as f:
    tokenizer: BPE = pickle.load(f)
with open("tokenized.pkl", "rb") as f:
    tokenized_text: list = pickle.load(f)

X = [
    tokenized_text[i : i + seq_length]
    for i in range(0, len(tokenized_text), seq_length)
]
X = X[:-1]
X = X[:1000]

random.seed(42)
random.shuffle(X)


mt_network = [
    mt_layers.Embedding(vocab_size, embedding_size),
    mt_layers.TransformerLayer(embedding_size, head_count, 2),
    mt_layers.TransformerLayer(embedding_size, head_count, 2),
    mt_layers.TransformerLayer(embedding_size, head_count, 2),
    mt_layers.TransformerLayer(embedding_size, head_count, 2),
    mt_layers.TransformerLayer(embedding_size, head_count, 2),
    mt_layers.TransformerLayer(embedding_size, head_count, 2),
    mt_layers.TransformerLayer(embedding_size, head_count, 2),
    mt_layers.TransformerLayer(embedding_size, head_count, 2),
    mt_layers.Linear(embedding_size, vocab_size, True),
]

mt_params = [p for layer in mt_network for p in layer.params]
mt_optimizer = mt_optimizers.Adam(
    [(mt_params, lr, 0.0)],
    warmup_steps=warmup_steps,
)


monitor_toks = tokenizer.tokenize(
    b"this is a simple sentence, nothing too weird going on here."
)
monitor_tensor = Tensor(cp.array(monitor_toks)[None, :])

step = 0
best_loss = float("inf")

for e in range(epochs):
    losses = []
    for batch_start in trange(0, len(X), batch_size):
        batch = X[batch_start : batch_start + batch_size]
        if len(batch) < batch_size:
            continue

        batch_np = np.array(batch, dtype=np.int32)
        x_np = batch_np[:, :-1]
        y_np = batch_np[:, 1:]

        # mytorch step
        mt_x = Tensor(cp.array(x_np, dtype=cp.int32))
        mt_y = Tensor(cp.array(y_np, dtype=cp.int32))
        mt_processed = mt_x
        for layer in mt_network:
            mt_processed = layer.forward(mt_processed)
        mt_loss_obj = mt_layers.CrossEntropyLoss().forward(mt_processed, mt_y)
        mt_loss_val = float(mt_loss_obj.value)
        mt_loss_obj.compute_gradient()
        mt_optimizer.step()
        losses.append(mt_loss_val)
    mean_loss = np.mean(losses)
    if mean_loss < best_loss:
        old_ops = mt_network[-1].weights.operations
        mt_network[-1].weights.operations = []
        with open("network.pkl", "wb") as f:
            pickle.dump(mt_network, f)
        mt_network[-1].weights.operations = old_ops
    print(np.mean(losses), losses[-1])
    processed = monitor_tensor
    for j, layer in enumerate(mt_network):
        processed = layer.forward(processed)
    pred = cp.argmax(processed[0].value, axis=-1)
    print(tokenizer.untokenize(pred.tolist()))
