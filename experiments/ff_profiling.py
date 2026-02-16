import cupy as cp

from mytorch import gpu_layers, optimizers
from mytorch.gpu_tensor import GpuTensor

# do a series of linear + sigmoids, then we can see what's slow in that

insize = 10
hiddensize = 200
outsize = 2
batchsize = 4
steps = 1000
network = [
    gpu_layers.Linear(insize, hiddensize, True),
    # gpu_layers.Sigmoid(),
    gpu_layers.Linear(hiddensize, hiddensize, True),
    # gpu_layers.Sigmoid(),
    gpu_layers.Linear(hiddensize, hiddensize, True),
    gpu_layers.Sigmoid(),
    gpu_layers.Linear(hiddensize, outsize, True),
    gpu_layers.Sigmoid(),
]

start_vector = GpuTensor(cp.random.random((batchsize, insize), dtype=cp.float32))
expected = GpuTensor(
    value=cp.array(
        [[0.86, 0.23], [0.6, 0.3], [0.8, 0.2], [0.16, 0.73]],
    )
)
for i in range(steps):
    current = start_vector
    for layer in network:
        current = layer.forward(current)
    loss = current - expected
    squared_loss = (loss * loss).sum()
    optimizers.sgd_step(squared_loss, 0.02)
    if i % 100 == 0:
        print(current.value)
