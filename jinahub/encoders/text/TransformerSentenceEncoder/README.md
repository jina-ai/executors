# Executor Sentence Encoder 

**TransformerSentenceEncoder** wraps the [Sentence Transformer](https://www.sbert.net/docs)
library into an `Jina` executor. 

### Inputs 

`Document` with `text` sentences.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (depends on the used model) with `dtype=nfloat32`.

## Usage 

```python
from jina import Flow, Document

f = Flow().add(uses='docker://executor-sentence-transformer:latest')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

## Reference
- [Sentence Transformer Library](https://www.sbert.net/docs)
