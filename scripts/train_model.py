import pickle
import random

import click
import cupy as cp
import numpy as np
from tqdm import trange

from mytorch import layers as layers
from mytorch import optimizers as mt_optimizers
from mytorch.tensor import Tensor
from mytorch.tokenizers.tokenizers import BPE


@click.command()
@click.option(
    "--tokenizer", "tokenizer_path", default="small_tokenizer.pkl", show_default=True
)
@click.option(
    "--tokenized", "tokenized_path", default="small_tokenized.pkl", show_default=True
)
@click.option("--output", "output_path", default="network.pkl", show_default=True)
@click.option("--embedding-size", default=256, show_default=True)
@click.option("--lr", default=2e-4, show_default=True)
@click.option("--seq-length", default=128, show_default=True)
@click.option("--batch-size", default=8, show_default=True)
@click.option("--warmup-steps", default=10000, show_default=True)
@click.option("--epochs", default=5, show_default=True)
def main(
    tokenizer_path,
    tokenized_path,
    output_path,
    embedding_size,
    lr,
    seq_length,
    batch_size,
    warmup_steps,
    epochs,
):
    head_count = embedding_size // 32

    with open(tokenizer_path, "rb") as f:
        tokenizer: BPE = pickle.load(f)
    with open(tokenized_path, "rb") as f:
        tokenized_text: list = pickle.load(f)

    vocab_size = tokenizer.vocab_size
    X = [
        tokenized_text[i : i + seq_length]
        for i in range(0, len(tokenized_text), seq_length)
    ]
    X = X[:-1]

    random.seed(12)
    random.shuffle(X)

    mt_network = [
        layers.Embedding(vocab_size, embedding_size),
        layers.TransformerLayer(embedding_size, head_count, 2),
        layers.TransformerLayer(embedding_size, head_count, 2),
        layers.TransformerLayer(embedding_size, head_count, 2),
        layers.TransformerLayer(embedding_size, head_count, 2),
        layers.TransformerLayer(embedding_size, head_count, 2),
        layers.TransformerLayer(embedding_size, head_count, 2),
        layers.TransformerLayer(embedding_size, head_count, 2),
        layers.TransformerLayer(embedding_size, head_count, 2),
        layers.Linear(embedding_size, vocab_size, True),
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

    best_loss = float("inf")

    for e in range(epochs):
        losses = []
        pbar = trange(0, len(X), batch_size)
        for batch_start in pbar:
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
            mt_loss_obj = layers.CrossEntropyLoss().forward(mt_processed, mt_y)
            mt_loss_val = float(mt_loss_obj.value)
            mt_loss_obj.compute_gradient()
            mt_optimizer.step()
            losses.append(mt_loss_val)
            pbar.set_postfix(loss=f"{mt_loss_val:.4f}")

        mean_loss = np.mean(losses)
        if mean_loss < best_loss:
            old_ops = mt_network[-1].weights.operations
            mt_network[-1].weights.operations = []
            with open(output_path, "wb") as f:
                pickle.dump(mt_network, f)
            mt_network[-1].weights.operations = old_ops
        print(np.mean(losses), losses[-1])
        processed = monitor_tensor
        for j, layer in enumerate(mt_network):
            processed = layer.forward(processed)
        pred = cp.argmax(processed[0].value, axis=-1)
        print(tokenizer.untokenize(pred.tolist()))


if __name__ == "__main__":
    main()
