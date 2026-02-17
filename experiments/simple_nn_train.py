import cupy as cp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mytorch import gpu_layers, optimizers
from mytorch.gpu_tensor import GpuTensor

lr = 0.001
batchsize = 8
hiddensize = 1000
insize = 50
outsize = 1
epochs = 20
network = [
    gpu_layers.Linear(insize, hiddensize, True),
    gpu_layers.Sigmoid(),
    gpu_layers.Linear(hiddensize, hiddensize, True),
    gpu_layers.Sigmoid(),
    gpu_layers.Linear(hiddensize, outsize, True),
    gpu_layers.Sigmoid(),
]

X, y = make_classification(
    n_samples=300, n_features=insize, n_informative=insize // 2, n_classes=outsize
)

X = cp.asarray(X, dtype=cp.float32)
y = cp.asarray(y, dtype=cp.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


for e in range(epochs):
    losses = []
    for i in range(0, X_train.shape[0], batchsize):
        X_batch = GpuTensor(X_train[i : i + batchsize])
        y_batch = GpuTensor(y_train[i : i + batchsize])

        processed = X_batch
        for layer in network:
            processed = layer.forward(processed)
        loss = processed - y_batch
        squared_loss = (loss * loss).sum()
        losses.append(float(squared_loss.value))
        optimizers.sgd_step(squared_loss, lr)
        if cp.any(cp.isnan(processed.value)):
            1 / 0
    test_losses = []
    for i in range(0, X_test.shape[0], batchsize):
        X_batch = GpuTensor(X_test[i : i + batchsize])
        y_batch = GpuTensor(y_test[i : i + batchsize])
        processed = X_batch
        for layer in network:
            processed = layer.forward(processed)
        loss = processed - y_batch
        squared_loss = (loss * loss).sum()
        test_losses.append(float(squared_loss.value))
    print(f"mean train loss for epoch {e} was {np.mean(losses)}")
    print(f"mean test loss for epoch {e} was {np.mean(test_losses)}")
