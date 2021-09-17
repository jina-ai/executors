# BigTransferEncoder

``BigTransferEncoder`` uses the Big Transfer models presented by Google [here]((https://github.com/google-research/big_transfer)).
Use the prebuilt images from JinaHub in your Python code,

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://BigTransferEncoder')
```

For more information, such as run executor on gpu, check out [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
