# FlairTextEncoder

`FlairTextEncoder` encodes `Document` content from an array of string in size `B` into a ndarray in size `B x D`.
 
Internally, `FlairTextEncoder` wraps the DocumentPoolEmbeddings from Flair.

## Usage

Here is an example usage of the **FlairTextEncoder**.

```python
from jina import Flow, Document
f = Flow().add(uses='jinahub+docker://FlairTextEncoder')
with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
```

### Inputs 

`Document` with `text` to be encoded.

### Returns

`Document` with `embedding` fields filled with an `ndarray`  with `dtype=nfloat32`.


