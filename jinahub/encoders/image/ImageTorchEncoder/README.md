# ✨ ImageTorchEncoder

**ImageTorchEncoder** wraps the models from [torchvision](https://pytorch.org/vision/stable/index.html).

**ImageTorchEncoder** encodes `Document` blobs of type a `ndarray` and shape Batch x Height x Width x Channel 
into a `ndarray` of Batch x Dim and stores them in the `embedding` attribute of the `Document`.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

In case you want to install the dependencies locally run 
```
pip install -r requirements.txt
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images

Use the prebuilt images from JinaHub in your Python codes.
With the `volumes` argument you can pass model from your local machine into the Docker container.
```python
from jina import Flow

flow = Flow().add(uses='jinahub+docker://ImageTorchEncoder',
                  volumes='/your_home_folder/.cache/torch:/root/.cache/torch')
```
Alternatively, reference the docker image in the `yml` config
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://ImageTorchEncoder'
    volumes: '/your_home_folder/.cache/torch:/root/.cache/torch'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ImageTorchEncoder',
               volumes='/your_home_folder/.cache/torch:/root/.cache/torch')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://ImageTorchEncoder'
    volumes: '/your_home_folder/.cache/torch:/root/.cache/torch'
```

## 🎉️ Example 

```python
import numpy as np

from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://ImageTorchEncoder')

doc = Document(blob=np.ones((224, 224, 3), dtype=np.uint8))

with f:
    resp = f.post(on='/index', inputs=doc, return_results=True)
    print(f'{resp}')
    
    
print('\n\nembedding:\n\n', resp[0].docs[0].embedding)
```

Example using the class the class `ImageTorchEncoder` directly

```python
import numpy as np

from jina import Document, DocumentArray
from jinahub.image.encoder.torch_encoder import ImageTorchEncoder

doc = Document(blob=np.ones((224, 224, 3), dtype=np.uint8))
encoder = ImageTorchEncoder()
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
- [PyTorch TorchVision Transformers Preprocessing](https://sparrow.dev/torchvision-transforms/)
