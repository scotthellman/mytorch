import cupy as cp

from mytorch.layers import TransformerLayer
from mytorch.tensor import Tensor


class Decoder:
    def __init__(self, network, tokenizer):
        self.network = network
        self.tokenizer = tokenizer

    def get_next_tok(self, in_vec, use_cache, temperature=0.1):
        processed = in_vec
        for j, layer in enumerate(self.network):
            if isinstance(layer, TransformerLayer):
                processed = layer.forward(processed, use_cache=use_cache)
            else:
                processed = layer.forward(processed)
        relevant = processed.value[0][-1]
        if temperature == 0:
            g = cp.zeros_like(relevant)
        else:
            g = cp.random.gumbel(size=relevant.shape)

        return int(cp.argmax(relevant / temperature + g))

    def sample_text(
        self,
        start: bytes,
        num_tokens: int,
        use_cache: bool,
        temperature: float,
        seed: int = 2,
    ) -> bytes:
        cp.random.seed(seed)
        toks = self.tokenizer.tokenize(start)
        next_input = list(toks)

        while len(toks) < num_tokens:
            tens = Tensor(cp.array(next_input))[None, :]
            new_tok = self.get_next_tok(tens, use_cache, temperature=temperature)
            toks.append(new_tok)
            if use_cache:
                next_input = [new_tok]
            else:
                next_input.append(new_tok)

        for layer in self.network:
            if isinstance(layer, TransformerLayer):
                layer.reset_cache()
        return self.tokenizer.untokenize(toks)
