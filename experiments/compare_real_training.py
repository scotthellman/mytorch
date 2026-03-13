"""
Side-by-side training comparison using real data and full model size.

Yes, this one is Claude-authored. Seemed like exactly the sort of tedious
and error prone script that was better left to an LLM.

Creates a mytorch model, transfers its weights to an equivalent PyTorch model,
then trains both on identical batches and reports loss + weight divergence at
every step. Stops early if divergence exceeds threshold.

Run from the repo root:
    python experiments/compare_real_training.py [--steps N] [--thresh T]
"""

import argparse
import pickle
import random

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mytorch import layers as layers
from mytorch import optimizers as mt_optimizers
from mytorch.tensor import Tensor
from mytorch.tokenizers.tokenizers import BPE

# ── hyperparameters (must match both training scripts) ────────────────────────

with open("small_tokenizer.pkl", "rb") as f:
    tokenizer: BPE = pickle.load(f)
with open("small_tokenized.pkl", "rb") as f:
    tokenized_text: list = pickle.load(f)

vocab_size = tokenizer.vocab_size
embedding_size = 256
head_count = embedding_size // 32
lr = 2e-4
seq_length = 128
batch_size = 8
warmup_steps = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── pytorch model (copied from small_language_model_pytorch.py) ───────────────


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    # x: (b, h, s, d) — interleaved pairs: (0,1), (2,3), ...
    # matches mytorch rope kernel: theta_i = 10000^(-2i/d), shared by element 2i and 2i+1
    b, h, s, d = x.shape
    half = d // 2
    pos = torch.arange(1, s + 1, device=x.device, dtype=torch.float32)
    thetas = 10000.0 ** (
        -2.0 * torch.arange(half, device=x.device, dtype=torch.float32) / d
    )
    angles = pos[:, None] * thetas[None, :]  # (s, half)
    cos, sin = torch.cos(angles), torch.sin(angles)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_odd * cos + x_even * sin
    return out


class SoftmaxAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.key_size = d // n_heads
        self.qkv_proj = nn.Linear(d, d * 3, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d, dim=-1)

        def split_heads(t):
            return t.view(b, s, self.n_heads, self.key_size).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q, k = apply_rope(q), apply_rope(k)
        scale = 1.0 / (self.key_size**0.5)
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        result = (attn @ v).transpose(1, 2).reshape(b, s, self.d)
        return self.out_proj(result)


class TransformerLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, expansion_factor: int = 2):
        super().__init__()
        self.attention = SoftmaxAttention(d, n_heads)
        self.attention_norm = nn.LayerNorm(d)
        self.linear_expand = nn.Linear(d, d * expansion_factor)
        self.linear_contract = nn.Linear(d * expansion_factor, d)
        self.linear_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        interim = self.attention_norm(attended + x)
        processed = self.linear_contract(F.elu(self.linear_expand(interim)))
        return self.linear_norm(processed + interim)


class SmallLM(nn.Module):
    def __init__(self, vocab_size: int, d: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList(
            [TransformerLayer(d, n_heads) for _ in range(n_layers)]
        )
        self.output = nn.Linear(d, vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output(h)


# ── weight transfer helpers ───────────────────────────────────────────────────


def cp_to_pt(cp_arr: cp.ndarray) -> torch.Tensor:
    return torch.tensor(cp.asnumpy(cp_arr), dtype=torch.float32).to(device)


def transfer_weights(mt_network: list, pt_model: SmallLM):
    mt_emb = mt_network[0]
    mt_layers_list = mt_network[1:-1]
    mt_out = mt_network[-1]

    pt_model.embedding.weight.data = cp_to_pt(mt_emb.weights.value)

    for pt_layer, mt_layer in zip(pt_model.layers, mt_layers_list):
        pt_layer.attention.qkv_proj.weight.data = cp_to_pt(
            mt_layer.attention.weights.value.T
        )
        pt_layer.attention.out_proj.weight.data = cp_to_pt(
            mt_layer.attention.out_weights.value.T
        )
        pt_layer.attention_norm.weight.data = cp_to_pt(mt_layer.attention_norm.w.value)
        pt_layer.attention_norm.bias.data = cp_to_pt(mt_layer.attention_norm.b.value)
        pt_layer.linear_expand.weight.data = cp_to_pt(
            mt_layer.linear_expand.weights.value.T
        )
        pt_layer.linear_expand.bias.data = cp_to_pt(
            mt_layer.linear_expand.bias.value.squeeze(0)
        )
        pt_layer.linear_contract.weight.data = cp_to_pt(
            mt_layer.linear_contract.weights.value.T
        )
        pt_layer.linear_contract.bias.data = cp_to_pt(
            mt_layer.linear_contract.bias.value.squeeze(0)
        )
        pt_layer.linear_norm.weight.data = cp_to_pt(mt_layer.linear_norm.w.value)
        pt_layer.linear_norm.bias.data = cp_to_pt(mt_layer.linear_norm.b.value)

    pt_model.output.weight.data = cp_to_pt(mt_out.weights.value.T)
    pt_model.output.bias.data = cp_to_pt(mt_out.bias.value.squeeze(0))


# ── weight comparison ─────────────────────────────────────────────────────────


def spot_check_weights(mt_network: list, pt_model: SmallLM) -> tuple[str, float]:
    """Return (name, max_diff) for the single most-diverged parameter."""
    mt_emb = mt_network[0]
    mt_layers_list = mt_network[1:-1]
    mt_out = mt_network[-1]

    checks = [("emb", mt_emb.weights.value, pt_model.embedding.weight)]
    for i, (mt_layer, pt_layer) in enumerate(zip(mt_layers_list, pt_model.layers)):
        checks += [
            (
                f"L{i}.qkv",
                mt_layer.attention.weights.value,
                pt_layer.attention.qkv_proj.weight.T,
            ),
            (
                f"L{i}.out",
                mt_layer.attention.out_weights.value,
                pt_layer.attention.out_proj.weight.T,
            ),
            (
                f"L{i}.expand",
                mt_layer.linear_expand.weights.value,
                pt_layer.linear_expand.weight.T,
            ),
            (
                f"L{i}.contract",
                mt_layer.linear_contract.weights.value,
                pt_layer.linear_contract.weight.T,
            ),
            (
                f"L{i}.anorm",
                mt_layer.attention_norm.w.value,
                pt_layer.attention_norm.weight,
            ),
            (f"L{i}.lnorm", mt_layer.linear_norm.w.value, pt_layer.linear_norm.weight),
        ]
    checks.append(("out", mt_out.weights.value, pt_model.output.weight.T))

    worst_name, worst_d = "", 0.0
    for name, mt_w, pt_w in checks:
        d = float(np.abs(cp.asnumpy(mt_w) - pt_w.detach().cpu().numpy()).max())
        if d > worst_d:
            worst_d = d
            worst_name = name
    return worst_name, worst_d


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=200, help="steps to run (default: 200)"
    )
    parser.add_argument(
        "--thresh", type=float, default=0.1, help="weight diff threshold for early stop"
    )
    args = parser.parse_args()

    # ── data ─────────────────────────────────────────────────────────────────

    X = [
        tokenized_text[i : i + seq_length]
        for i in range(0, len(tokenized_text), seq_length)
    ]
    X = X[:-1]
    X = X[:1000]

    random.seed(42)
    random.shuffle(X)

    # ── mytorch network ───────────────────────────────────────────────────────

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

    # ── pytorch network (initialised from mytorch weights) ────────────────────

    pt_model = SmallLM(vocab_size, embedding_size, head_count, n_layers=8).to(device)
    transfer_weights(mt_network, pt_model)

    pt_optimizer = torch.optim.Adam(
        pt_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    def warmup_lambda(step: int) -> float:
        return (step + 1) / (warmup_steps + 1) if step < warmup_steps else 1.0

    pt_scheduler = torch.optim.lr_scheduler.LambdaLR(pt_optimizer, warmup_lambda)
    pt_loss_fn = nn.CrossEntropyLoss()

    # ── lockstep training ─────────────────────────────────────────────────────

    print(f"Running {args.steps} steps on real data  |  stop threshold: {args.thresh}")
    print(
        f"{'step':>5}  {'mt_loss':>9}  {'pt_loss':>9}  {'loss_diff':>10}  {'worst_param':>12}  {'weight_diff':>12}"
    )
    print("-" * 75)

    batches = list(range(0, len(X), batch_size))
    step = 0

    for batch_start in batches:
        if step >= args.steps:
            break

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

        # pytorch step
        pt_x = torch.tensor(x_np, dtype=torch.long, device=device)
        pt_y = torch.tensor(y_np, dtype=torch.long, device=device)
        pt_optimizer.zero_grad()
        pt_logits = pt_model(pt_x)
        b, t, v = pt_logits.shape
        pt_loss = pt_loss_fn(pt_logits.reshape(b * t, v), pt_y.reshape(b * t))
        pt_loss_val = pt_loss.item()
        pt_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(pt_model.parameters()), 1.0)
        pt_optimizer.step()
        pt_scheduler.step()

        step += 1

        worst_name, worst_d = spot_check_weights(mt_network, pt_model)
        loss_diff = mt_loss_val - pt_loss_val
        print(
            f"{step:>5}  {mt_loss_val:>9.4f}  {pt_loss_val:>9.4f}  {loss_diff:>+10.4f}"
            f"  {worst_name:>12}  {worst_d:>12.4e}"
        )

        if worst_d > args.thresh:
            print(f"\nWeights diverged beyond threshold {args.thresh} at step {step}.")
            break
    else:
        print(f"\nCompleted {step} steps without exceeding threshold {args.thresh}.")


if __name__ == "__main__":
    main()
