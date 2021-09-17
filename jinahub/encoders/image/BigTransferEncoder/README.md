# BigTransferEncoder

``BigTransferEncoder`` uses the Big Transfer models presented by Google [here]((https://github.com/google-research/big_transfer)).
Use the prebuilt images from JinaHub in your Python code,

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://BigTransferEncoder')
```


This encoder also offers a GPU version under the `gpu` tag. To use it, make sure to pass `device='/GPU:0'`, as the initialization parameter, and `gpus='all'` when adding the containerized Executor to the Flow. See the [Executor on GPU](https://docs.jina.ai/tutorials/gpu_executor/) section of Jina documentation for more details.
Here's how you would modify the example above to use a GPU

```python
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://ImageTFEncoder/gpu',
    uses_with={'device': '/GPU:0'},
    gpus='all'
)
```

## Reference

- [BiT github repo](https://github.com/google-research/big_transfer)
- [TensorflowHub](https://tfhub.dev/google/collections/bit/1)
