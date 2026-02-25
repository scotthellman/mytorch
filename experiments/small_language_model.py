import pickle

import cupy as cp
import numpy as np
from tqdm import tqdm

from mytorch import gpu_layers, optimizers
from mytorch.gpu_tensor import GpuTensor
from mytorch.tokenizers import NaiveBPE

with open("/home/scott/Downloads/alice_in_wonderland.txt", "r") as f:
    text = f.read()

text = text.encode("utf-8")

vocab_size = 1024
embedding_size = 256
lr = 5e-5
seq_length = 64
batch_size = 8
epochs = 10

# FIXME: need at least a start special tok

loss_func = gpu_layers.CrossEntropyLoss()
network = [
    gpu_layers.Embedding(vocab_size, embedding_size),
    gpu_layers.TransformerLayer(embedding_size, 8),
    gpu_layers.TransformerLayer(embedding_size, 8),
    gpu_layers.Linear(embedding_size, vocab_size, True),
]

optimizer = optimizers.Adam(lr=lr)
network[-1].weights = network[0].weights.transpose_last()

tokenizer = NaiveBPE(vocab_size)
tokenizer.fit([text])

tokenized_text = tokenizer.tokenize(text)

X = []
for i in range(0, len(tokenized_text), seq_length):
    X.append(tokenized_text[i : i + seq_length])

# chuck the last row, it won't have the right shape
X = cp.array(X[:-1], dtype=cp.int32)

for e in range(epochs):
    losses = []
    for i in tqdm(range(0, X.shape[0], batch_size), total=X.shape[0] // batch_size):
        X_batch = GpuTensor(X[i : i + batch_size, :-1])
        y_batch = GpuTensor(X[i : i + batch_size, 1:])

        processed = X_batch
        for j, layer in enumerate(network):
            processed = layer.forward(processed)

        loss = loss_func.forward(processed, y_batch)
        losses.append(float(loss.value))
        optimizer.step(loss)
    print(f"mean train loss for epoch {e} was {np.mean(losses)}")

network[-1].weights.operations = []
network[0].weights.operations = []
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("network.pkl", "wb") as f:
    pickle.dump(network, f)
