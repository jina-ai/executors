# FlairTextEncoder

**FlairTextEncoder** is a class that wraps the text embedding functionality using models from the **flair** library.
 
This module provides a subset sentence embedding functionality from the flair library, namely it allows you classical word embeddings, byte-pair embeddings and flair embeddings, and create sentence embeddings from a combtination of these models using document pool embeddings.

Due to different interfaces of all these embedding models, using custom pre-trained models (not part of the library), or other embedding models is not possible. For that, we recommend that you create a custom executor.

The following parameters can be passed on initialization:
- `embeddings`: the name of the embeddings. Supported models include
    - `word:[ID]`: the classic word embedding model, the `[ID]` are listed at
    https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md
    - `flair:[ID]`: the contextual embedding model, the `[ID]` are listed at
    https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
    - `byte-pair:[ID]`: the subword-level embedding model, the `[ID]` are listed at
    https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md

    Example: `('word:glove', 'flair:news-forward', 'flair:news-backward')`
- `default_batch_size`: Default batch size, used if ``batch_size`` is not provided as a parameter in the request
- `default_traversal_paths`: Default traversal paths, used if `traversal_paths` are not provided as a parameter in the request.
- `device`: The device (cpu or gpu) that the model should be on.
- `pooling`: the strategy to merge the word embeddings into the sentence embedding. Supported strategies are `'mean'`, `'min'` and `'max'`.

### Inputs 

`Document` with `text` to be encoded.

### Returns

`Document` with `embedding` fields filled with a numpy array.

## Usage

Here is an example usage of the **FlairTextEncoder**.

```python
from jina import Flow, Document
f = Flow().add(uses='jinahub+docker://FlairTextEncoder')
with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
```

References

- [flair GitHub repository](https://github.com/flairNLP/flair)

