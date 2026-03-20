import pickle
import time

import click
from mytorch import layers
from mytorch.inference import Decoder
from mytorch.tensor import DisableGradManager
from mytorch.tokenizers.tokenizers import BPE
from pathlib import Path

def generate_text(prefix: bytes, tokenizer_fn: Path, network_fn: Path, temperature: float = 0.1):
    with open(tokenizer_fn, "rb") as f:
        tokenizer: BPE = pickle.load(f)
    with open(network_fn, "rb") as f:
        network: BPE = pickle.load(f)
    decoder = Decoder(network, tokenizer)

    with DisableGradManager():
        decoder.sample_text(prefix, 200, True, temperature=temperature)


@click.command()
@click.argument("prefix")
@click.option("--tokenizer", "tokenizer_fn", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--network", "network_fn", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--temperature", default=0.1, show_default=True)
def main(prefix: str, tokenizer_fn: Path, network_fn: Path, temperature: float):
    generate_text(prefix.encode(), tokenizer_fn, network_fn, temperature)


if __name__ == "__main__":
    main()
