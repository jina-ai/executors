# AudioLoader

AudioLoader loads audio files into the Document buffer.
The supported audio types are: MP3, WAV

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://AudioLoader')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://AudioLoader')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
