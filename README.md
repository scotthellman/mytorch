MyTorch
=====

A minimal implementation of all of the pieces necessary to train a transformer on a GPU without using PyTorch. 

This is a learning project whose goal was to implement transformers "from scratch." Specifically, I stopped at using CuPy to handle invoking custom CUDA kernels - so e.g. memory management of data to and from the GPU was out of scope.

What's implemented:

* Tokenization with Byte Pair Encoding, both learning the vocabulary and tokenizing a text using that vocabulary.
* Custom CUDA kernels for all GPU operations (modulo a few FIXMEs where I am still relying on CuPy)
* Rotary positional embeddings (RoPE)
* Both Softmax and Linear self attention (though Linear doesn't currently support kv caching)
* Reverse-mode automatic differentiation
* Optimization with AdamW
* Linear, Self Attention, Embedding, and LayerNorm neural network layers
* ELU and Sigmoid activation functions

Equivalent MyTorch and PyTorch networks behave the same (their weights differ by ~1e-5 over ~50 training steps, computed by `experiments/compare_real_training.py`). That said, there's unsurprisingly no reason to ever use MyTorch in practice - I haven't done any formal benchmarking, but PyTorch is at least 4x as fast. I don't know the full cause of that, but I suspect a lot of it comes from ineffecient kernels and excessive memory copying.

Example invocations:

```shell
uv run scripts/tokenize_wiki.py --data-dir ~/Downloads/wiki/ --vocab-size 2048 --tokenizer-out test_tokenizer.pkl --tokenized-out test_tokenized.pkl --max-docs 5000

uv run scripts/train_model.py --tokenizer test_tokenizer.pkl --tokenized test_tokenized.pkl --output test_network.pkl --batch-size 16

uv run scripts/run_inference.py "It was the year when they finally immanentized the Eschaton" --tokenizer test_tokenizer.pkl --network test_network.pkl --temperature 0.3

```

I've only run this code on my machine, but as far as I know, it should work anywhere that can properly run the correct CuPY version.

Resources
-------

An incomplete list of resources that were helpful in finishing this project:

* Efficient matrix multiplication from [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
* Linear self attention from the paper itself, [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
* RoPE also from the paper itself, [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
* Layernorm from [Karpathy's description](https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md)
* Refresher on the basics of reverse mode autograd from [Reverse-mode automatic differentiation from scratch, in Python](https://sidsite.com/posts/autodiff/)
* Byte Pair Encoding from [Efficient BPE Tokenization from Scratch](https://github.com/marta1994/efficient_bpe_explanation)