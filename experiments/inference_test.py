import pickle
import time

from mytorch import layers
from mytorch.inference import Decoder
from mytorch.tensor import DisableGradManager
from mytorch.tokenizers.tokenizers import BPE

if __name__ == "__main__":
    with open("small_tokenizer.pkl", "rb") as f:
        tokenizer: BPE = pickle.load(f)
    vocab_size = tokenizer.vocab_size
    embedding_size = 256
    head_count = 4
    network = [
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
    decoder = Decoder(network, tokenizer)

    text = b"this is a  "
    with DisableGradManager():
        start = time.time()
        print(decoder.sample_text(text, 200, False, temperature=1))
        print(time.time() - start)
        start = time.time()
        print(decoder.sample_text(text, 200, True, temperature=1))
        print(time.time() - start)
