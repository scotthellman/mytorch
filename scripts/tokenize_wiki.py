import bz2
import pickle
import re
import time
from pathlib import Path

import click
import mwxml
from tqdm import tqdm, trange

from mytorch.tokenizers.tokenizers import BPE


@click.command()
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--vocab-size", required=True, type=int)
@click.option("--tokenizer-out", required=True, type=click.Path(path_type=Path))
@click.option("--tokenized-out", required=True, type=click.Path(path_type=Path))
@click.option("--max-docs", default=10000, type=int, show_default=True)
def build_tokens(
    data_dir: Path, vocab_size: int, tokenizer_out: Path, tokenized_out: Path, max_docs: int
):
    tqdm.set_lock(tqdm.get_lock())

    ref_pattern = re.compile(r"\{\{(?:[^{}]|\{\{[^{}]*\}\})*\}\}", flags=re.DOTALL)
    bad_start_pattern = re.compile(r"^[=|*!{<].*", re.MULTILINE)
    empty_pattern = re.compile(r"^\s*$", re.MULTILINE)

    # TODO: if i changed my tokenier to take an iterable of texts,
    # we could work with the full dump. But for now, just grab the first
    # few thousand articles
    # 100 articles is roughly 2 megs

    print("Pulling data from bzip")
    texts = []
    for fn in data_dir.glob("*.bz2"):
        with bz2.open(fn) as f:
            dump = mwxml.Dump.from_file(f)
            for page in dump:
                if page.redirect is not None:
                    continue
                for revision in page:
                    text = revision.text
                    text = re.sub(ref_pattern, "", text)
                    text = re.sub(bad_start_pattern, "", text)
                    text = re.sub(empty_pattern, "", text)
                    text = text.replace("[[", "")
                    text = text.replace("]]", "")

                    texts.append(text)
                if len(texts) > max_docs:
                    break

    final_text = "\n".join(texts)
    final_text = final_text.encode("utf-8")
    print("final text is of length", len(final_text))
    del texts

    tokenizer = BPE(vocab_size)
    start = time.time()
    tokenizer.fit(final_text)
    print("vocab learning took", time.time() - start)
    with open(tokenizer_out, "wb") as f:
        pickle.dump(tokenizer, f)

    chunk_size = 500000
    tokenized_text = []
    for i in trange(0, len(final_text), chunk_size, position=0):
        tokenized_subtext = tokenizer.tokenize(
            final_text[i : i + chunk_size], pbar_pos=1
        )
        tokenized_text.extend(tokenized_subtext)
    with open(tokenized_out, "wb") as f:
        pickle.dump(tokenized_text, f)


if __name__ == "__main__":
    build_tokens()
