# ImageTFEncoder

`ImageTFEncoder` encodes ``Document`` content from a ndarray, potentially BatchSize x (Height x Width x Channel) into a ndarray of `BatchSize * d`.
Internally, `ImageTFEncoder` wraps the models from [tensorflow.keras.applications](https://keras.io/applications/).

Use the prebuilt images from JinaHub in your Python code,

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://ImageTFEncoder',)
```

For more information on the `gpu` usage and `volume` cache, please refer to the [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
