# FlairTextEncoder

**FlairTextEncoder** is a class that wraps the text embedding functionality using models from the **flair** library.
 
This module provides a subset sentence embedding functionality from the flair library, namely it allows you classical word embeddings, byte-pair embeddings and flair embeddings, and create sentence embeddings from a combtination of these models using document pool embeddings.

Due to different interfaces of all these embedding models, using custom pre-trained models (not part of the library), or other embedding models is not possible. For that, we recommend that you create a custom executor.


## References

- [flair GitHub repository](https://github.com/flairNLP/flair)

<!-- version=v0.2 -->

