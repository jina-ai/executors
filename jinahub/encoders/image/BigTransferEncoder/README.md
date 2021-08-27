# Big Transfer Image Encoder

**Big Transfer Image Encoder** is a class that uses the Big Transfer models presented by Google [here]((https://github.com/google-research/big_transfer)).
It uses a pretrained version of a BiT model to encode an image from an array of shape 
(Batch x (Channel x Height x Width)) into an array of shape (Batch x Encoding) 

The following parameters can be used:

- `model_path` (string, default: "pretrained"): The folder where the downloaded pretrained Big Transfer model is located
- `model_name` (string, default: "R50x1"): The model to be downloaded when the model_path is empty. Choose from ['R50x1', 'R101x1', 'R50x3', 'R101x3', 'R152x4']
- `channel_axis` (int): The axis where the channel of the images needs to be (model-dependent)
- `on_gpu` (bool): Specifies whether the model should be used on GPU or CPU. To use GPU,
  put into one batch (limited by the request_size)
  either the GPU docker container needs to be used or you need to install CUDA 11.3 and cudnn8 (similar versions might also work)
- `default_traversal_paths` (List[str], defaults to ['r']): Traversal path through the docs
- `default_batch_size` (int): Batch size to be used in the encoder model. If not specified, all the documents are




## Usage



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
    uses_with:
      on_gpu: true
```

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


## Reference
- https://github.com/google-research/big_transfer
- https://tfhub.dev/google/collections/bit/1
