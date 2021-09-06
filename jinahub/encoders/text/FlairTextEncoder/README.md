# FlairTextEncoder

 **FlairTextEncoder** is a class that wraps the text embedding functionality using models from the **flair** library.
 

This module provides a subset sentence embedding functionality from the flair library, namely it allows you classical word embeddings, byte-pair embeddings and flair embeddings, and create sentence embeddings from a combtination of these models using document pool embeddings.

Due to different interfaces of all these embedding models, using custom pre-trained models (not part of the library), or other embedding models is not possible. For that, we recommend that you create a custom executor.

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

