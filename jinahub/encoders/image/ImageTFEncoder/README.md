# ImageTFEncoder

`ImageTFEncoder` encodes ``Document`` content from a ndarray, potentially BatchSize x (Height x Width x Channel) into a ndarray of `BatchSize * d`. Internally, :class:`ImageTFEncoder` wraps the models from `tensorflow.keras.applications`. https://keras.io/applications/




## Usage 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://ImageTFEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `blob` of the shape `256`.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (=128, by default) with `dtype=nfloat32`.


