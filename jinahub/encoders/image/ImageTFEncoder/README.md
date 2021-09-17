# ImageTFEncoder

`ImageTFEncoder` encodes ``Document`` content from a ndarray, potentially BatchSize x (Height x Width x Channel) into a ndarray of `BatchSize * d`.
Internally, `ImageTFEncoder` wraps the models from [tensorflow.keras.applications](https://keras.io/applications/).

The following arguments can be passed on initialization:

- `model_name`: name of the pre-trained CLIP model.
- `img_shape`: The shape of the image to be encoded.
- `pool_strategy`: Pooling strategy, default use max pooling, available options are `None`, `mean`, `max`.
- `device`: Pytorch device to put the model on, e.g. 'cpu', 'cuda'.
- `traversal_paths`: traversal path (use `cpu` if not specified in request's parameters).
- `batch_size`: default batch size (use `32` if not specified in request's parameters).

#### Inputs 

`Document`s with the `blob` attribute.

#### Returns

`Document`s with `embedding` fields filled with an `ndarray` of the shape 1024 with `dtype=float32`.

## Usage

Use the prebuilt images from JinaHub in your Python code,

```python
from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://ImageTFEncoder',
)
```

or in the .yml config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://ImageTFEncoder'
```

Note that this way the Executor will download the model every time it starts up. You can
re-use the cached model files by mounting the cache directory that the model is using
into the container. To do this, modify the Flow definition like this

```python
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://ImageTFEncoder',
    volumes='/your/home/dir/.cache/tensorflow:/root/.cache/tensorflow'
)
```

This encoder also offers a GPU version under the `gpu` tag. To use it, make sure to pass `device='cuda'`, as the initialization parameter, and `gpus='all'` when adding the containerized Executor to the Flow. See the [Executor on GPU](https://docs.jina.ai/tutorials/gpu_executor/) section of Jina documentation for more details.

Here's how you would modify the example above to use a GPU

```python
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://ImageTFEncoder/gpu',
    uses_with={'device': 'cuda'},
    gpus='all',
    volumes='/your/home/dir/.cache/tensorflow:/root/.cache/tensorflow' 
)
```

## Reference

- [Tensorflow Applications](https://keras.io/api/applications/)
