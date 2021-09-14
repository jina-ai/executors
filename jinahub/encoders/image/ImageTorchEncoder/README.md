# ImageTorchEncoder

**ImageTorchEncoder** wraps the models from [torchvision](https://pytorch.org/vision/stable/index.html).

**ImageTorchEncoder** encodes `Document` blobs of type `ndarray` and shape Height x Width x Channel 
into an `ndarray` of shape `embedding_dim` and stores them in the `embedding` attribute of the `Document`.


## Usage


Use the prebuilt images from Jina Hub in your Python codes and add it to your Flow:
```python
from jina import Flow

flow = Flow().add(uses='jinahub+docker://ImageTorchEncoder')
```
### Encoding example
After creating a Flow, prepare your Documents to encode. They should have the blob attribute set with shape 
Height x Width x Channel. Then, we can start the Flow and encode the Documents. By default, any endpoint will encode 
the Documents:

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

### Set `volumes`

With the `volumes` attribute, you can map the torch cache directory to your local cache directory, in order to avoid downloading 
the model each time you start the Flow.

```python
from jina import Flow

flow = Flow().add(
    uses='jinahub+docker://ImageTorchEncoder',
    volumes='/your_home_folder/.cache/torch:/root/.cache/torch'
)
```

Alternatively, you can reference the docker image in the `yml` config and specify the `volumes` configuration.

`flow.yml`:

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://ImageTorchEncoder'
    volumes: '/your_home_folder/.cache/torch:/root/.cache/torch'
```

And then use it like so:
```python
from jina import Flow

flow = Flow().add(uses='flow.yml')
```

### Returns
`Document` with `embedding` fields filled with an `ndarray` of shape `embedding_dim` (size depends on the model) with `dtype=float32`.

### Supported models:
You can specify the model to use with the parameter `model_name`:
```python
import numpy as np

from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://ImageTorchEncoder',
    uses_with={'model_name': 'alexnet'}
)

doc = Document(blob=np.ones((224, 224, 3), dtype=np.uint8))

with f:
    resp = f.post(on='/index', inputs=doc, return_results=True)
    print(f'{resp}')

print('\n\nembedding:\n\n', resp[0].docs[0].embedding)
```

`ImageTorchEncoder` supports the following models: 

* `alexnet`
* `squeezenet1_0`
* `vgg16`
* `densenet161`
* `inception_v3`
* `googlenet`
* `shufflenet_v2_x1_0`
* `mobilenet_v2`
* `mnasnet1_0`
* `resnet18`

By default, `resnet18` is the used model.

You can check the models [here](https://pytorch.org/vision/stable/models.html)

### GPU usage:
To enable GPU, you can set the `device` parameter to a cuda device.
Make sure your machine is cuda-compatible.
If you're using a docker container, make sure to add the `gpu` tag and enable 
GPU access to Docker with `gpus='all'`.
Furthermore, make sure you satisfy the prerequisites mentioned in 
[Executor on GPU tutorial](https://docs.jina.ai/tutorials/gpu_executor/#prerequisites).

```python
import numpy as np

from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://ImageTorchEncoder/gpu',
               uses_with={'device': 'cuda'}, gpus='all')

doc = Document(blob=np.ones((224, 224, 3), dtype=np.uint8))

with f:
    resp = f.post(on='/index', inputs=doc, return_results=True)
    print(f'{resp}')

print('\n\nembedding:\n\n', resp[0].docs[0].embedding)
```

## Reference

- [PyTorch TorchVision Transformers Preprocessing](https://sparrow.dev/torchvision-transforms/)
- [PyTorch TorchVision](https://pytorch.org/vision/stable/index.html)
- [TorchVision models](https://pytorch.org/vision/stable/models.html)
