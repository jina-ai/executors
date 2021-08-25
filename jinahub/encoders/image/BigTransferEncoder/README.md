# ✨ Big Transfer Image Encoder

**Big Transfer Image Encoder** is a class that uses the Big Transfer models presented by Google [here]((https://github.com/google-research/big_transfer)).
It uses a pretrained version of a BiT model to encode an image from an array of shape 
(Batch x (Channel x Height x Width)) into an array of shape (Batch x Encoding) 

The following parameters can be used:

- `model_path` (string, default: "pretrained"): The folder where the downloaded pretrained Big Transfer model is located
- `model_name` (string, default: "R50x1"): The model to be downloaded when the model_path is empty. Choose from ['R50x1', 'R101x1', 'R50x3', 'R101x3', 'R152x4']
- `channel_axis` (int): The axis where the channel of the images needs to be (model-dependent)
- `device` (string, default: 'cpu'): Specify the device to run the model on. Examples are 'cpu', 'cuda', 'cuda:2'.
- `default_traversal_paths` (List[str], defaults to ['r']): Traversal path through the docs
- `default_batch_size` (int): Batch size to be used in the encoder model. If not specified, all the documents are

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

No prerequisites are required to run this executor. The executor automatically
downloads the BiT model specified by `model_name`! Alternatively, you could also 
download the model in advance and use the `model_path` parameter.

In case you want to install the dependencies locally run 
```
pip install -r requirements.txt
```
To verify the installation works:
```
pytest tests
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://BigTransferEncoder')
```

or in the `.yml` config.

You can either use a volume to mount an already downloaded model into the container
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://BigTransferEncoder'
    volumes: '/your_pretrained/path:/big_transfer/pretrained'
    uses_with: 
      model_path: '/big_transfer/pretrained'
```

or specify a model name which will download the model automatically.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://BigTransferEncoder'
    uses_with: 
      model_name: 'R50x1'
```


The prebuilt images do currently not support GPU.  

#### using source code
Use the source code from JinaHub in your Python code:

```python
import numpy as np

from jina import Flow, Document

f = Flow().add(uses='jinahub://BigTransferEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(blob=np.ones((96, 96, 3), dtype=np.float32)), return_results=True)
    print(f'{resp[0].docs[0].embedding.shape}')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://BigTransferEncoder'
```

#### GPU usage

You can use the GPU via the source code. Therefore, you need a matching CUDA version
and GPU drivers installed on your system. 
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://BigTransferEncoder'
    uses_with:
      device: 'cuda'
```
Alternatively, use the jinahub gpu docker container. Therefore, you need GPU
drivers installed on your system and nvidia-docker installed.
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://BigTransferEncoder/gpu'
    docker_kwargs: 
      runtime: 'nvidia'
    uses_with:
      device: 'cuda'
```

## 🎉️ Example

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://BigTransferEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` with image `blob`.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (model-dependent) with `dtype=nfloat32`.


## 🔍️ Reference
- https://github.com/google-research/big_transfer
- https://tfhub.dev/google/collections/bit/1
