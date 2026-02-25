import pickle

import cupy as cp

from mytorch.gpu_tensor import GpuTensor

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("network.pkl", "rb") as f:
    network = pickle.load(f)

prefix = b"Hello world"
toks = tokenizer.tokenize(prefix)


def get_next_tok(in_vec, network):
    processed = in_vec
    for j, layer in enumerate(network):
        processed = layer.forward(processed)
    out_tok = int(cp.argmax(processed.value, axis=-1)[-1][-1])
    return out_tok


for i in range(60):
    in_vec = GpuTensor(cp.array(toks, dtype=cp.int32)[None, ...])
    next_tok = get_next_tok(in_vec, network)
    toks.append(next_tok)

print("SLM says:")
print(tokenizer.untokenize(toks))
