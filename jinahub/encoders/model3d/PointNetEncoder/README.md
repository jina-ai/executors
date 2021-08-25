# PointNetEncoder

PointNetEncoder embeds point cloud 3 models into vectors

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://PointNetEncoder')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://PointNetEncoder')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
