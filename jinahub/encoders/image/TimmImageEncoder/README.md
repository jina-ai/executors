# TimmImageEncoder

**TimmImageEncoder** wraps the models from [timm](https://rwightman.github.io/pytorch-image-models/).

**TimmImageEncoder** encodes `Document` blobs of type a `ndarray` and shape Batch x Height x Width x Channel 
into a `ndarray` of Batch x Dim and stores them in the `embedding` attribute of the `Document`.

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

In case you want to install the dependencies locally run 
```
pip install -r requirements.txt
```

## Usages

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://TimmImageEncoder')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://TimmImageEncoder')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`

## 🎉️ Example 

```python
import numpy as np

from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TimmImageEncoder')

doc = Document(blob=np.ones((224, 224, 3), dtype=np.uint8))

with f:
    resp = f.post(on='/index', inputs=doc, return_results=True)
    print(f'{resp}')
    
    
print('\n\nembedding:\n\n', resp[0].docs[0].embedding)
```

Example using the class the class `TimmImageEncoder` directly

```python
import numpy as np

from jina import Document, DocumentArray
from jinahub.image.encoder.timm_encoder import TimmImageEncoder

doc = Document(blob=np.ones((224, 224, 3), dtype=np.uint8))
encoder = TimmImageEncoder()
doc_array = DocumentArray([doc, doc])
encoder.encode(doc_array, parameters={})
list_embeddings = doc_array.get_attributes('embedding')
list_embeddings[0].shape, list_embeddings[1].shape
```

### Inputs 
If `use_default_preprocessing=True` (recommended):  
`Document` with `blob` of shape `H x W x C` and dtype `uint8`.  

If `use_default_preprocessing=False`:  
`Document` with `blob` of shape `C x H x W` and dtype `float32`.

### Returns
`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (size depends on the model) with `dtype=float32`.

## 🔍️ Reference
- [Timm Models](https://rwightman.github.io/pytorch-image-models/models/)
