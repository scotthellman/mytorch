import cupy as cp
import numpy as np

from mytorch import gpu_layers, optimizers
from mytorch.gpu_tensor import GpuTensor

lr = 0.002
batchsize = 8
hiddensize = 32
vocabsize = 4
epochs = 30
n_samples = 200
seq_length = 2
network = [
    gpu_layers.Embedding(vocabsize, hiddensize),
    gpu_layers.SelfAttention(hiddensize, 2),
    gpu_layers.Linear(hiddensize, hiddensize, True),
    gpu_layers.Sigmoid(),
    gpu_layers.LayerNorm(),
    gpu_layers.Linear(hiddensize, vocabsize, True),
]
optimizer = optimizers.Adam(lr=lr)

loss_func = gpu_layers.CrossEntropyLoss()

# tie the in and out embeddings together
# FIXME: make sure this works - T is just a view right?
# network[-1].weights.value = network[0].weights.value.T

X = []
y = []
for i in range(n_samples):
    vec = cp.random.randint(0, vocabsize, size=seq_length)
    X.append(vec)
    # y.append(vec)
    # this objective is trivial _if_ we're mixing information across seq steps
    # impossible otherwise
    y.append(cp.roll(vec, shift=1))
    y[-1][0] = 0
X = cp.vstack(X)
y = cp.vstack(y, dtype=cp.int32)


for e in range(epochs):
    losses = []
    for i in range(0, X.shape[0], batchsize):
        X_batch = GpuTensor(X[i : i + batchsize])
        y_batch = GpuTensor(y[i : i + batchsize])

        processed = X_batch
        for j, layer in enumerate(network):
            processed = layer.forward(processed)

        if cp.any(cp.isnan(processed.value)):
            print(cp.any(cp.isnan(network[1].weights.value)))
            print(processed.value)
            1 / 0
        loss = loss_func.forward(processed, y_batch)
        losses.append(float(loss.value))
        optimizer.step(loss)
    print(f"mean train loss for epoch {e} was {np.mean(losses)}")
print(cp.argmax(processed.value, axis=-1))
print(y_batch.value)
